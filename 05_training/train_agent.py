#!/usr/bin/env python3
"""
ART (Automated Red Teaming) Training Pipeline for Agentic Control

This module implements an automated red teaming framework for training robust
control agents. The ART system:

1. Generates adversarial scenarios automatically
2. Evaluates agent performance on challenging cases  
3. Collects failure modes and success patterns
4. Trains improved agents through iterative refinement
5. Creates datasets for supervised learning approaches

The training pipeline supports:
- Adversarial scenario generation
- Multi-controller benchmarking
- Data collection and curation
- Model fine-tuning and evaluation
- Robustness testing across operating ranges

Usage:
    python train_agent.py --mode generate_data --num_episodes 1000
    python train_agent.py --mode train --model_type supervised
    python train_agent.py --mode evaluate --checkpoint model.pt
"""

import os
import sys
import json
import time
import random
import argparse
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from collections import defaultdict
import logging

# Add parent directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '01_basic_physics'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '02_direct_control'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '03_langgraph_tools'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'slurm'))

from double_integrator import DoubleIntegrator
from llm_integration import VLLMController, OpenAIController
from rollout import control_rollout, Trajectory, ControlScenario
from control_graph import ToolAugmentedController
from experiment_runner_clean import CleanExperimentRunner


@dataclass
class ARTConfig:
    """Configuration for ART training"""
    # Data generation
    num_episodes: int = 1000
    difficulty_levels: List[str] = None
    position_range: Tuple[float, float] = (-2.0, 2.0)  # Extended range
    velocity_range: Tuple[float, float] = (-1.5, 1.5)
    
    # Training
    model_type: str = "supervised"  # "supervised", "reinforcement", "hybrid"
    batch_size: int = 32
    learning_rate: float = 1e-4
    num_epochs: int = 100
    
    # Evaluation
    test_scenarios: int = 500
    success_threshold: float = 0.95
    
    # Paths
    data_dir: str = "art_data"
    model_dir: str = "art_models"
    results_dir: str = "art_results"
    
    def __post_init__(self):
        if self.difficulty_levels is None:
            self.difficulty_levels = ["easy", "medium", "hard", "extreme", "adversarial"]


class AdversarialScenarioGenerator:
    """Generates challenging scenarios for red teaming"""
    
    def __init__(self, config: ARTConfig):
        self.config = config
        self.failure_patterns = []
        self.success_patterns = []
        
    def generate_baseline_scenarios(self, num_scenarios: int) -> List[ControlScenario]:
        """Generate baseline scenarios across difficulty levels"""
        scenarios = []
        
        for i in range(num_scenarios):
            # Sample difficulty level
            difficulty = random.choice(self.config.difficulty_levels)
            
            # Generate scenario based on difficulty
            if difficulty == "easy":
                pos = random.uniform(-0.3, 0.3)
                vel = random.uniform(-0.2, 0.2)
                max_steps = 50
            elif difficulty == "medium":
                pos = random.uniform(-0.7, 0.7)
                vel = random.uniform(-0.5, 0.5)
                max_steps = 100
            elif difficulty == "hard":
                pos = random.uniform(-1.2, 1.2)
                vel = random.uniform(-0.8, 0.8)
                max_steps = 150
            elif difficulty == "extreme":
                pos = random.uniform(-1.8, 1.8)
                vel = random.uniform(-1.2, 1.2)
                max_steps = 200
            else:  # adversarial
                pos = random.uniform(*self.config.position_range)
                vel = random.uniform(*self.config.velocity_range)
                max_steps = 250
                
            scenario = ControlScenario(
                name=f"{difficulty}_art_{i}",
                initial_position=pos,
                initial_velocity=vel,
                target_position=0.0,
                target_velocity=0.0,
                difficulty=difficulty,
                max_steps=max_steps
            )
            scenarios.append(scenario)
            
        return scenarios
    
    def generate_adversarial_scenarios(self, failure_data: List[Dict], num_scenarios: int) -> List[ControlScenario]:
        """Generate scenarios based on observed failure patterns"""
        scenarios = []
        
        if not failure_data:
            # If no failure data, generate challenging scenarios
            return self.generate_baseline_scenarios(num_scenarios)
        
        for i in range(num_scenarios):
            # Sample a failure case to build upon
            failure_case = random.choice(failure_data)
            
            # Add noise and variations around the failure point
            base_pos = failure_case.get('initial_position', 0.0)
            base_vel = failure_case.get('initial_velocity', 0.0)
            
            # Perturb around failure point
            pos_noise = random.uniform(-0.3, 0.3)
            vel_noise = random.uniform(-0.2, 0.2)
            
            pos = np.clip(base_pos + pos_noise, *self.config.position_range)
            vel = np.clip(base_vel + vel_noise, *self.config.velocity_range)
            
            scenario = ControlScenario(
                name=f"adversarial_art_{i}",
                initial_position=pos,
                initial_velocity=vel,
                target_position=0.0,
                target_velocity=0.0,
                difficulty="adversarial",
                max_steps=250
            )
            scenarios.append(scenario)
            
        return scenarios
    
    def analyze_failure_patterns(self, trajectories: List[Dict]) -> Dict[str, Any]:
        """Analyze failure patterns from trajectory data"""
        failures = []
        successes = []
        
        for traj in trajectories:
            if traj.get('success', False):
                successes.append(traj)
            else:
                failures.append(traj)
        
        analysis = {
            'total_episodes': len(trajectories),
            'num_failures': len(failures),
            'num_successes': len(successes),
            'failure_rate': len(failures) / len(trajectories) if trajectories else 0,
            'failure_patterns': self._extract_failure_patterns(failures),
            'success_patterns': self._extract_success_patterns(successes)
        }
        
        return analysis
    
    def _extract_failure_patterns(self, failures: List[Dict]) -> Dict[str, Any]:
        """Extract common patterns from failure cases"""
        if not failures:
            return {}
        
        # Analyze initial conditions of failures
        positions = [f.get('initial_position', 0) for f in failures]
        velocities = [f.get('initial_velocity', 0) for f in failures]
        
        patterns = {
            'position_range': [min(positions), max(positions)],
            'velocity_range': [min(velocities), max(velocities)],
            'common_positions': self._find_clusters(positions),
            'common_velocities': self._find_clusters(velocities),
            'avg_steps': np.mean([f.get('num_steps', 0) for f in failures])
        }
        
        return patterns
    
    def _extract_success_patterns(self, successes: List[Dict]) -> Dict[str, Any]:
        """Extract common patterns from success cases"""
        if not successes:
            return {}
            
        positions = [s.get('initial_position', 0) for s in successes]
        velocities = [s.get('initial_velocity', 0) for s in successes]
        
        # Extract confidence values properly (handle lists)
        confidence_values = []
        for s in successes:
            conf = s.get('confidence', 0)
            if isinstance(conf, list) and len(conf) > 0:
                confidence_values.append(np.mean(conf))  # Average if list
            elif isinstance(conf, (int, float)):
                confidence_values.append(conf)
            else:
                confidence_values.append(0.0)
        
        patterns = {
            'position_range': [min(positions), max(positions)],
            'velocity_range': [min(velocities), max(velocities)],
            'avg_steps': np.mean([s.get('num_steps', 0) for s in successes]),
            'avg_confidence': np.mean(confidence_values) if confidence_values else 0.0
        }
        
        return patterns
    
    def _find_clusters(self, data: List[float], num_clusters: int = 3) -> List[float]:
        """Simple clustering to find common values"""
        if len(data) < num_clusters:
            return data
            
        # Simple binning approach
        min_val, max_val = min(data), max(data)
        if min_val == max_val:
            return [min_val]
        
        bin_width = (max_val - min_val) / num_clusters
        clusters = []
        
        for i in range(num_clusters):
            bin_start = min_val + i * bin_width
            bin_end = min_val + (i + 1) * bin_width
            bin_data = [x for x in data if bin_start <= x < bin_end]
            if bin_data:
                clusters.append(np.mean(bin_data))
                
        return clusters


class ARTDataCollector:
    """Collects and manages training data from control experiments"""
    
    def __init__(self, config: ARTConfig):
        self.config = config
        self.data = {
            'trajectories': [],
            'scenarios': [],
            'controller_performance': defaultdict(list),
            'metadata': {}
        }
        
    async def collect_episode_data(self, scenario: ControlScenario, 
                                 controllers: Dict[str, Any]) -> Dict[str, Any]:
        """Collect data from a single episode with multiple controllers"""
        episode_data = {
            'scenario': asdict(scenario),
            'controller_results': {},
            'timestamp': time.time()
        }
        
        for controller_name, controller in controllers.items():
            try:
                # Run the controller on this scenario
                trajectory = await control_rollout(controller, scenario, verbose=False)
                
                # Convert trajectory to dict
                traj_data = {
                    'positions': trajectory.positions,
                    'velocities': trajectory.velocities,
                    'actions': trajectory.actions,
                    'timestamps': trajectory.timestamps,
                    'success': trajectory.success,
                    'final_error': trajectory.final_error,
                    'num_steps': len(trajectory.positions),
                    'controller_name': controller_name,
                    'scenario_name': scenario.name,
                    'initial_position': scenario.initial_position,
                    'initial_velocity': scenario.initial_velocity,
                    'difficulty': scenario.difficulty
                }
                
                # Add controller-specific data
                if hasattr(trajectory, 'reasoning'):
                    traj_data['reasoning'] = trajectory.reasoning
                if hasattr(trajectory, 'confidence'):
                    traj_data['confidence'] = trajectory.confidence
                if hasattr(trajectory, 'tool_calls'):
                    traj_data['tool_calls'] = trajectory.tool_calls
                    
                episode_data['controller_results'][controller_name] = traj_data
                self.data['trajectories'].append(traj_data)
                
            except Exception as e:
                logging.error(f"Error collecting data for {controller_name}: {e}")
                episode_data['controller_results'][controller_name] = {
                    'error': str(e),
                    'success': False
                }
        
        return episode_data
    
    def save_data(self, filepath: str):
        """Save collected data to file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Add metadata
        self.data['metadata'] = {
            'collection_time': time.time(),
            'num_episodes': len(self.data['trajectories']),
            'config': asdict(self.config)
        }
        
        with open(filepath, 'w') as f:
            json.dump(self.data, f, indent=2, default=str)
            
        logging.info(f"Saved ART data to {filepath}")
    
    def load_data(self, filepath: str):
        """Load previously collected data"""
        with open(filepath, 'r') as f:
            self.data = json.load(f)
        logging.info(f"Loaded ART data from {filepath}")
    
    def get_training_dataset(self) -> Tuple[List[Dict], List[Dict]]:
        """Extract training dataset from collected trajectories"""
        successful_trajectories = []
        failed_trajectories = []
        
        for traj in self.data['trajectories']:
            if traj.get('success', False):
                successful_trajectories.append(traj)
            else:
                failed_trajectories.append(traj)
        
        return successful_trajectories, failed_trajectories


class ARTTrainer:
    """Trains improved control agents using collected data"""
    
    def __init__(self, config: ARTConfig):
        self.config = config
        
    def train_supervised_model(self, training_data: List[Dict]) -> Dict[str, Any]:
        """Train a supervised model from successful trajectories"""
        logging.info("Training supervised model...")
        
        # This is a placeholder for actual model training
        # In practice, this would:
        # 1. Extract state-action pairs from successful trajectories
        # 2. Create training dataset with (state, action) pairs
        # 3. Train neural network to predict actions given states
        # 4. Validate on held-out test set
        
        # For demonstration, we'll analyze the data and return statistics
        if not training_data:
            return {'error': 'No training data provided'}
        
        # Extract features from successful trajectories
        features = []
        targets = []
        
        for traj in training_data:
            if not traj.get('success', False):
                continue
                
            positions = traj.get('positions', [])
            velocities = traj.get('velocities', []) 
            actions = traj.get('actions', [])
            
            # Create state-action pairs
            for i in range(len(positions) - 1):
                state = [positions[i], velocities[i]]
                action = actions[i]
                
                features.append(state)
                targets.append(action)
        
        if not features:
            return {'error': 'No valid state-action pairs found'}
        
        features = np.array(features)
        targets = np.array(targets)
        
        # Analyze the dataset
        training_stats = {
            'num_samples': len(features),
            'state_dim': features.shape[1],
            'action_range': [float(np.min(targets)), float(np.max(targets))],
            'state_ranges': {
                'position': [float(np.min(features[:, 0])), float(np.max(features[:, 0]))],
                'velocity': [float(np.min(features[:, 1])), float(np.max(features[:, 1]))]
            },
            'mean_action': float(np.mean(targets)),
            'std_action': float(np.std(targets))
        }
        
        # Simple linear regression as baseline
        try:
            from sklearn.linear_model import LinearRegression
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import mean_squared_error, r2_score
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                features, targets, test_size=0.2, random_state=42
            )
            
            # Train model
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            training_stats.update({
                'model_type': 'LinearRegression',
                'mse': float(mse),
                'r2_score': float(r2),
                'coefficients': model.coef_.tolist(),
                'intercept': float(model.intercept_)
            })
            
            # Save simple model
            model_path = os.path.join(self.config.model_dir, 'linear_baseline.json')
            os.makedirs(self.config.model_dir, exist_ok=True)
            
            model_data = {
                'type': 'linear_regression',
                'coefficients': model.coef_.tolist(),
                'intercept': float(model.intercept_),
                'training_stats': training_stats
            }
            
            with open(model_path, 'w') as f:
                json.dump(model_data, f, indent=2)
                
            training_stats['model_saved'] = model_path
            
        except ImportError:
            logging.warning("scikit-learn not available, skipping model training")
            training_stats['model_type'] = 'analysis_only'
        
        return training_stats
    
    def evaluate_model(self, model_path: str, test_scenarios: List[ControlScenario]) -> Dict[str, Any]:
        """Evaluate trained model on test scenarios"""
        logging.info(f"Evaluating model from {model_path}")
        
        # Load model
        if not os.path.exists(model_path):
            return {'error': f'Model file not found: {model_path}'}
        
        with open(model_path, 'r') as f:
            model_data = json.load(f)
        
        # Create simple linear controller from saved model
        class LearnedController:
            def __init__(self, coefficients, intercept):
                self.coef_ = np.array(coefficients)
                self.intercept_ = intercept
                
            def get_action(self, position, velocity, target_pos=0.0, target_vel=0.0):
                pos_error = target_pos - position
                vel_error = target_vel - velocity
                state = np.array([pos_error, vel_error])
                action = np.dot(self.coef_, state) + self.intercept_
                action = np.clip(action, -1.0, 1.0)  # Saturate
                
                return {
                    'action': float(action),
                    'confidence': 0.8,
                    'reasoning': f'Learned linear policy: {action:.3f}'
                }
        
        if model_data['type'] == 'linear_regression':
            controller = LearnedController(
                model_data['coefficients'], 
                model_data['intercept']
            )
        else:
            return {'error': f'Unsupported model type: {model_data["type"]}'}
        
        # Evaluate on test scenarios
        successes = 0
        results = []
        
        env = DoubleIntegrator()
        
        for scenario in test_scenarios[:min(100, len(test_scenarios))]:  # Limit for demo
            env.reset(scenario.initial_position, scenario.initial_velocity)
            
            trajectory = {
                'positions': [scenario.initial_position],
                'velocities': [scenario.initial_velocity],
                'actions': []
            }
            
            for step in range(scenario.max_steps):
                pos, vel = env.get_state()
                
                # Get action from learned controller
                result = controller.get_action(pos, vel, 0.0, 0.0)
                action = result['action']
                
                trajectory['actions'].append(action)
                
                # Step environment
                env.step(action)
                new_pos, new_vel = env.get_state()
                
                trajectory['positions'].append(new_pos)
                trajectory['velocities'].append(new_vel)
                
                # Check success
                if abs(new_pos) < 0.1 and abs(new_vel) < 0.1:
                    successes += 1
                    trajectory['success'] = True
                    trajectory['num_steps'] = step + 1
                    break
            else:
                trajectory['success'] = False
                trajectory['num_steps'] = scenario.max_steps
            
            results.append(trajectory)
        
        evaluation_stats = {
            'num_test_scenarios': len(results),
            'success_rate': successes / len(results),
            'avg_steps': np.mean([r['num_steps'] for r in results]),
            'model_path': model_path,
            'model_type': model_data['type']
        }
        
        return evaluation_stats


class ARTManager:
    """Main manager for ART training pipeline"""
    
    def __init__(self, config: ARTConfig):
        self.config = config
        self.scenario_generator = AdversarialScenarioGenerator(config)
        self.data_collector = ARTDataCollector(config)
        self.trainer = ARTTrainer(config)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Create directories
        os.makedirs(config.data_dir, exist_ok=True)
        os.makedirs(config.model_dir, exist_ok=True)
        os.makedirs(config.results_dir, exist_ok=True)
        
    def run_data_generation(self) -> str:
        """Generate training data through adversarial scenarios"""
        logging.info("Starting ART data generation...")
        
        # Create controllers for data collection
        controllers = {
            'pd_baseline': self._create_pd_controller(),
            'tool_augmented': ToolAugmentedController()
        }
        
        # Try to add LQR controller
        try:
            controllers['lqr_optimal'] = self._create_lqr_controller()
        except Exception as e:
            logging.warning(f"Could not create LQR controller: {e}")
        
        # Generate initial scenarios
        scenarios = self.scenario_generator.generate_baseline_scenarios(self.config.num_episodes)
        
        logging.info(f"Generated {len(scenarios)} scenarios")
        
        # Collect data (synchronous version for simplicity)
        collected_episodes = []
        
        for i, scenario in enumerate(scenarios):
            if i % 100 == 0:
                logging.info(f"Processing scenario {i+1}/{len(scenarios)}")
            
            episode_data = self._collect_episode_sync(scenario, controllers)
            collected_episodes.append(episode_data)
            
            # Add to data collector
            for controller_name, result in episode_data['controller_results'].items():
                if 'error' not in result:
                    self.data_collector.data['trajectories'].append(result)
        
        self.data_collector.data['scenarios'] = [asdict(s) for s in scenarios]
        
        # Save collected data
        data_path = os.path.join(self.config.data_dir, f'art_data_{int(time.time())}.json')
        self.data_collector.save_data(data_path)
        
        # Analyze failure patterns
        analysis = self.scenario_generator.analyze_failure_patterns(
            self.data_collector.data['trajectories']
        )
        
        analysis_path = os.path.join(self.config.results_dir, f'failure_analysis_{int(time.time())}.json')
        with open(analysis_path, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        logging.info(f"Data generation complete. Collected {len(self.data_collector.data['trajectories'])} trajectories")
        logging.info(f"Failure rate: {analysis['failure_rate']:.3f}")
        
        return data_path
    
    def run_training(self, data_path: str) -> str:
        """Train models using collected data"""
        logging.info(f"Starting ART training with data from {data_path}")
        
        # Load data
        self.data_collector.load_data(data_path)
        
        # Get training dataset
        successful_trajectories, failed_trajectories = self.data_collector.get_training_dataset()
        
        logging.info(f"Training on {len(successful_trajectories)} successful trajectories")
        logging.info(f"Found {len(failed_trajectories)} failed trajectories for analysis")
        
        # Train supervised model
        training_results = self.trainer.train_supervised_model(successful_trajectories)
        
        # Save training results
        results_path = os.path.join(self.config.results_dir, f'training_results_{int(time.time())}.json')
        with open(results_path, 'w') as f:
            json.dump(training_results, f, indent=2, default=str)
        
        logging.info("Training complete!")
        return results_path
    
    def run_evaluation(self, model_path: str) -> str:
        """Evaluate trained model"""
        logging.info(f"Starting ART evaluation with model {model_path}")
        
        # Generate test scenarios
        test_scenarios = self.scenario_generator.generate_baseline_scenarios(self.config.test_scenarios)
        
        # Evaluate model
        evaluation_results = self.trainer.evaluate_model(model_path, test_scenarios)
        
        # Save evaluation results
        results_path = os.path.join(self.config.results_dir, f'evaluation_results_{int(time.time())}.json')
        with open(results_path, 'w') as f:
            json.dump(evaluation_results, f, indent=2, default=str)
        
        logging.info(f"Evaluation complete! Success rate: {evaluation_results.get('success_rate', 0):.3f}")
        return results_path
    
    def _collect_episode_sync(self, scenario: ControlScenario, controllers: Dict[str, Any]) -> Dict[str, Any]:
        """Synchronous version of episode data collection"""
        episode_data = {
            'scenario': asdict(scenario),
            'controller_results': {},
            'timestamp': time.time()
        }
        
        env = DoubleIntegrator()
        
        for controller_name, controller in controllers.items():
            try:
                env.reset(scenario.initial_position, scenario.initial_velocity)
                
                trajectory = {
                    'positions': [scenario.initial_position],
                    'velocities': [scenario.initial_velocity],
                    'actions': [],
                    'reasoning': [],
                    'confidence': [],
                    'controller_name': controller_name,
                    'scenario_name': scenario.name,
                    'initial_position': scenario.initial_position,
                    'initial_velocity': scenario.initial_velocity,
                    'difficulty': scenario.difficulty
                }
                
                for step in range(scenario.max_steps):
                    pos, vel = env.get_state()
                    
                    # Get action from controller
                    if hasattr(controller, 'get_action'):
                        result = controller.get_action(pos, vel, 0.0, 0.0)
                        if isinstance(result, dict):
                            action = result['action']
                            trajectory['reasoning'].append(result.get('reasoning', ''))
                            trajectory['confidence'].append(result.get('confidence', 0.0))
                        else:
                            action = result
                            trajectory['reasoning'].append('')
                            trajectory['confidence'].append(0.0)
                    else:
                        # Fallback for simple controllers
                        action = controller(pos, vel)
                        trajectory['reasoning'].append('')
                        trajectory['confidence'].append(0.0)
                    
                    trajectory['actions'].append(action)
                    
                    # Step environment
                    env.step(action)
                    new_pos, new_vel = env.get_state()
                    
                    trajectory['positions'].append(new_pos)
                    trajectory['velocities'].append(new_vel)
                    
                    # Check success
                    if abs(new_pos) < 0.1 and abs(new_vel) < 0.1:
                        trajectory['success'] = True
                        trajectory['final_error'] = abs(new_pos)
                        trajectory['num_steps'] = step + 1
                        break
                else:
                    trajectory['success'] = False
                    trajectory['final_error'] = abs(env.get_state()[0])
                    trajectory['num_steps'] = scenario.max_steps
                
                episode_data['controller_results'][controller_name] = trajectory
                
            except Exception as e:
                logging.error(f"Error in episode collection for {controller_name}: {e}")
                episode_data['controller_results'][controller_name] = {
                    'error': str(e),
                    'success': False
                }
        
        return episode_data
    
    def _create_pd_controller(self):
        """Create PD baseline controller"""
        class PDController:
            def get_action(self, position, velocity, target_pos=0.0, target_vel=0.0):
                pos_error = target_pos - position
                vel_error = target_vel - velocity
                action = 1.0 * pos_error + 2.0 * vel_error
                action = np.clip(action, -1.0, 1.0)
                return {
                    'action': float(action),
                    'confidence': 0.9,
                    'reasoning': f'PD control: kp*{pos_error:.3f} + kd*{vel_error:.3f} = {action:.3f}'
                }
        
        return PDController()
    
    def _create_lqr_controller(self):
        """Create LQR optimal controller"""
        try:
            from scipy.linalg import solve_continuous_are
            
            class LQRController:
                def __init__(self, Q_pos=1.0, Q_vel=1.0, R=1.0):
                    # Compute LQR gains
                    A = np.array([[0, 1], [0, 0]])
                    B = np.array([[0], [1]]) 
                    Q_matrix = np.array([[Q_pos, 0], [0, Q_vel]])
                    R_matrix = np.array([[R]])
                    
                    P = solve_continuous_are(A, B, Q_matrix, R_matrix)
                    K_matrix = np.linalg.inv(R_matrix) @ B.T @ P
                    self.K1 = float(K_matrix[0, 0])
                    self.K2 = float(K_matrix[0, 1])
                
                def get_action(self, position, velocity, target_pos=0.0, target_vel=0.0):
                    pos_error = target_pos - position
                    vel_error = target_vel - velocity
                    action = self.K1 * pos_error + self.K2 * vel_error
                    action = np.clip(action, -1.0, 1.0)
                    return {
                        'action': float(action),
                        'confidence': 0.95,
                        'reasoning': f'LQR control: K1*{pos_error:.3f} + K2*{vel_error:.3f} = {action:.3f}'
                    }
            
            return LQRController()
            
        except ImportError:
            raise ImportError("scipy not available for LQR controller")


def main():
    """Main entry point for ART training pipeline"""
    parser = argparse.ArgumentParser(description='ART Training Pipeline for Agentic Control')
    parser.add_argument('--mode', choices=['generate_data', 'train', 'evaluate', 'full_pipeline'], 
                       default='full_pipeline', help='Training pipeline mode')
    parser.add_argument('--num_episodes', type=int, default=200, help='Number of episodes for data generation')
    parser.add_argument('--data_path', type=str, help='Path to training data (for train/evaluate modes)')
    parser.add_argument('--model_path', type=str, help='Path to trained model (for evaluate mode)')
    parser.add_argument('--config_path', type=str, help='Path to config file')
    
    args = parser.parse_args()
    
    # Load or create config
    if args.config_path and os.path.exists(args.config_path):
        with open(args.config_path, 'r') as f:
            config_dict = json.load(f)
        config = ARTConfig(**config_dict)
    else:
        config = ARTConfig(num_episodes=args.num_episodes)
    
    # Create ART manager
    art_manager = ARTManager(config)
    
    if args.mode == 'generate_data':
        data_path = art_manager.run_data_generation()
        print(f"✅ Data generation complete: {data_path}")
        
    elif args.mode == 'train':
        if not args.data_path:
            print("❌ Error: --data_path required for train mode")
            return
        results_path = art_manager.run_training(args.data_path)
        print(f"✅ Training complete: {results_path}")
        
    elif args.mode == 'evaluate':
        if not args.model_path:
            print("❌ Error: --model_path required for evaluate mode")
            return
        results_path = art_manager.run_evaluation(args.model_path)
        print(f"✅ Evaluation complete: {results_path}")
        
    else:  # full_pipeline
        print("🚀 Running full ART pipeline...")
        
        # Generate data
        print("📊 Step 1: Generating training data...")
        data_path = art_manager.run_data_generation()
        print(f"✅ Data generation complete: {data_path}")
        
        # Train model
        print("🧠 Step 2: Training model...")
        training_results_path = art_manager.run_training(data_path)
        print(f"✅ Training complete: {training_results_path}")
        
        # Find the trained model path
        with open(training_results_path, 'r') as f:
            training_results = json.load(f)
        
        model_path = training_results.get('model_saved')
        if model_path and os.path.exists(model_path):
            # Evaluate model
            print("📈 Step 3: Evaluating model...")
            evaluation_results_path = art_manager.run_evaluation(model_path)
            print(f"✅ Evaluation complete: {evaluation_results_path}")
            
            # Print summary
            with open(evaluation_results_path, 'r') as f:
                eval_results = json.load(f)
            
            print("\n🎯 ART Pipeline Summary:")
            print(f"   Data collected: {config.num_episodes} episodes")
            print(f"   Training samples: {training_results.get('num_samples', 'N/A')}")
            print(f"   Model type: {training_results.get('model_type', 'N/A')}")
            print(f"   Test success rate: {eval_results.get('success_rate', 0):.3f}")
            print(f"   Model R² score: {training_results.get('r2_score', 'N/A')}")
            
        else:
            print("⚠️  No model was saved, skipping evaluation")
        
        print("\n🎉 Full ART pipeline complete!")


if __name__ == "__main__":
    main()