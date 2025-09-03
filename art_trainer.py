#!/usr/bin/env python3
"""
ART Training Pipeline for Agentic Control
==========================================

This module implements ART (Agent Reward Training) integration with our clean
agentic control pipeline to fine-tune the Tool-Augmented controller.

Features:
- Converts trajectory data to ART format
- Calculates rewards based on control performance 
- Implements training loop with proper evaluation
- Integrates with existing experiment pipeline
"""

import json
import asyncio
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass

# Import configuration system
from experiment_config import (
    setup_python_path, get_project_paths, get_latest_results_file, 
    get_scenarios, create_results_directories
)

# Setup paths and imports
setup_python_path()
from double_integrator import DoubleIntegrator
from control_graph import ToolAugmentedController

# ART framework
try:
    import art
    from art import TrainableModel, Trajectory, TrajectoryGroup
    ART_AVAILABLE = True
    print("✅ ART library available")
except ImportError:
    print("❌ ART library not found. Install with: pip install openpipe-art")
    ART_AVAILABLE = False
    
    # Mock classes for development
    class TrainableModel:
        def __init__(self, **kwargs):
            self.name = kwargs.get('model_name', 'mock_model')
            print(f"Mock TrainableModel created: {self.name}")
        
        def openai_client(self):
            return None
            
        async def train(self, groups):
            print(f"🎯 Mock training on {len(groups)} trajectory groups")
            return {"status": "mock_training_complete"}
    
    class Trajectory:
        def __init__(self, messages, reward, metadata=None):
            self.messages = messages
            self.reward = reward
            self.metadata = metadata or {}
    
    class TrajectoryGroup:
        def __init__(self, trajectories):
            self.trajectories = trajectories


@dataclass 
class ControlTrajectoryData:
    """Container for control trajectory data"""
    scenario_name: str
    positions: List[float]
    velocities: List[float] 
    controls: List[float]
    times: List[float]
    success: bool
    steps: int
    final_pos_error: float
    final_vel_error: float
    control_effort: float


class ControlRewardCalculator:
    """Calculate rewards for control trajectories"""
    
    def __init__(self, config=None):
        self.config = config or {}
        
        # Reward weights
        self.w_success = self.config.get('w_success', 100.0)      # Success reward
        self.w_efficiency = self.config.get('w_efficiency', -1.0) # Efficiency penalty  
        self.w_smoothness = self.config.get('w_smoothness', -0.1) # Control smoothness
        self.w_final_error = self.config.get('w_final_error', -50.0) # Final error penalty
        
    def calculate_reward(self, trajectory: ControlTrajectoryData) -> float:
        """Calculate reward for a control trajectory"""
        reward = 0.0
        
        # Success reward (most important)
        if trajectory.success:
            reward += self.w_success
        else:
            # Heavy penalty for failure
            reward -= abs(self.w_success)
            
        # Efficiency reward (fewer steps = higher reward)
        efficiency_bonus = self.w_efficiency * trajectory.steps
        reward += efficiency_bonus
        
        # Control smoothness (penalize jerky control)
        if len(trajectory.controls) > 1:
            control_changes = np.diff(trajectory.controls)
            smoothness_penalty = self.w_smoothness * np.sum(np.abs(control_changes))
            reward += smoothness_penalty
            
        # Final error penalty
        total_final_error = trajectory.final_pos_error + trajectory.final_vel_error
        error_penalty = self.w_final_error * total_final_error
        reward += error_penalty
        
        return float(reward)
    
    def get_reward_breakdown(self, trajectory: ControlTrajectoryData) -> Dict[str, float]:
        """Get detailed reward breakdown for analysis"""
        success_reward = self.w_success if trajectory.success else -abs(self.w_success)
        efficiency_reward = self.w_efficiency * trajectory.steps
        
        smoothness_reward = 0.0
        if len(trajectory.controls) > 1:
            control_changes = np.diff(trajectory.controls)  
            smoothness_reward = self.w_smoothness * np.sum(np.abs(control_changes))
            
        error_reward = self.w_final_error * (trajectory.final_pos_error + trajectory.final_vel_error)
        
        return {
            'success': success_reward,
            'efficiency': efficiency_reward, 
            'smoothness': smoothness_reward,
            'final_error': error_reward,
            'total': success_reward + efficiency_reward + smoothness_reward + error_reward
        }


class ARTControlTrainer:
    """Main ART training class for control systems"""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo", config: Dict = None):
        self.model_name = model_name
        self.config = config or {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Initialize components
        self.reward_calculator = ControlRewardCalculator(self.config.get('rewards', {}))
        self.results_dir = get_project_paths()["results"]
        create_results_directories()
        
        # Initialize ART model
        self.art_available = ART_AVAILABLE
        
        if self.art_available:
            self.model = TrainableModel(
                name=f"agentic_control_model_{self.timestamp}",
                project="agentic_control",
                base_model=model_name,
                **self.config.get('art_config', {})
            )
            # Register model with ART backend
            print("📝 Registering model with ART backend...")
            try:
                self.model.register()
                print("✅ Model registered successfully")
            except Exception as e:
                print(f"⚠️  Model registration failed: {e}")
                print("💡 This might be due to API keys or network issues")
                print("🔄 Will use mock training for demonstration")
                self.art_available = False
                self.model = TrainableModel(model_name=model_name)
        else:
            self.model = TrainableModel(model_name=model_name)
            
        print(f"🤖 ART Trainer initialized with model: {model_name}")
        
    def load_trajectory_data(self, file_path: Optional[Path] = None) -> Dict[str, List[ControlTrajectoryData]]:
        """Load trajectory data from JSON files"""
        if file_path is None:
            file_path = get_latest_results_file("pd_vs_tool_trajectories_*.json")
            
        if file_path is None:
            raise FileNotFoundError("No trajectory files found. Run experiments first.")
            
        print(f"📊 Loading trajectory data from: {file_path.name}")
        
        with open(file_path, 'r') as f:
            raw_data = json.load(f)
            
        # Convert to ControlTrajectoryData objects
        processed_data = {}
        
        for controller_name, trajectories in raw_data.items():
            processed_data[controller_name] = []
            
            for traj_dict in trajectories:
                trajectory = ControlTrajectoryData(
                    scenario_name=traj_dict['scenario'],
                    positions=traj_dict['positions'],
                    velocities=traj_dict['velocities'],
                    controls=traj_dict['controls'],
                    times=traj_dict['times'],
                    success=traj_dict['success'],
                    steps=traj_dict['steps'],
                    final_pos_error=traj_dict['final_pos_error'],
                    final_vel_error=traj_dict['final_vel_error'],
                    control_effort=traj_dict['control_effort']
                )
                processed_data[controller_name].append(trajectory)
                
        print(f"✅ Loaded {sum(len(trajs) for trajs in processed_data.values())} trajectories")
        return processed_data
    
    def convert_to_art_format(self, trajectories: List[ControlTrajectoryData]) -> List[Trajectory]:
        """Convert control trajectories to ART format"""
        art_trajectories = []
        
        for traj in trajectories:
            # Create conversation format for ART
            messages = []
            
            # System message with control objective
            messages.append({
                "role": "system",
                "content": "You are an expert control system. Use physics-aware tools to control a double integrator system optimally."
            })
            
            # Create step-by-step conversation based on trajectory
            for step in range(len(traj.controls)):
                # Current state observation
                pos = traj.positions[step]
                vel = traj.velocities[step] 
                
                user_msg = f"Current state: position={pos:.3f}m, velocity={vel:.3f}m/s. Target: position=0.0m, velocity=0.0m/s. What control action should I take?"
                messages.append({"role": "user", "content": user_msg})
                
                # Assistant's action (what the tool-augmented controller actually did)
                control_action = traj.controls[step]
                assistant_msg = f"Based on error analysis, I recommend control force: {control_action:.3f}N"
                messages.append({"role": "assistant", "content": assistant_msg})
            
            # Calculate reward for this trajectory
            reward = self.reward_calculator.calculate_reward(traj)
            
            # Create ART trajectory with metadata (clean up numpy types)
            reward_breakdown = self.reward_calculator.get_reward_breakdown(traj)
            clean_breakdown = {k: float(v) for k, v in reward_breakdown.items()}
            
            metadata = {
                'scenario': traj.scenario_name,
                'success': traj.success,
                'steps': traj.steps,
                'control_effort': float(traj.control_effort)
            }
            
            # Create ART trajectory (format depends on whether we have real ART)
            if self.art_available:
                # Real ART expects messages_and_choices
                art_traj = Trajectory(
                    messages_and_choices=messages, 
                    reward=float(reward), 
                    metadata=metadata,
                    metrics=clean_breakdown
                )
            else:
                # Mock ART uses simpler format
                art_traj = Trajectory(
                    messages=messages, 
                    reward=float(reward), 
                    metadata=metadata
                )
            art_trajectories.append(art_traj)
            
        print(f"🔄 Converted {len(trajectories)} trajectories to ART format")
        return art_trajectories
    
    def create_training_groups(self, art_trajectories: List[Trajectory]) -> List[TrajectoryGroup]:
        """Group trajectories for ART training"""
        # Group by reward (high vs low performance)
        high_reward_trajs = [t for t in art_trajectories if t.reward > 0]
        low_reward_trajs = [t for t in art_trajectories if t.reward <= 0]
        
        groups = []
        
        if high_reward_trajs:
            groups.append(TrajectoryGroup(high_reward_trajs))
            print(f"✅ High-reward group: {len(high_reward_trajs)} trajectories")
            
        if low_reward_trajs:
            groups.append(TrajectoryGroup(low_reward_trajs))
            print(f"⚠️  Low-reward group: {len(low_reward_trajs)} trajectories")
            
        return groups
    
    async def train_model(self, trajectory_data: Dict[str, List[ControlTrajectoryData]]) -> Dict[str, Any]:
        """Run ART training on trajectory data"""
        print("🎯 Starting ART training...")
        
        # Focus on Tool-Augmented trajectories for training
        if "Tool-Augmented" not in trajectory_data:
            raise ValueError("No Tool-Augmented trajectories found for training")
            
        tool_trajectories = trajectory_data["Tool-Augmented"]
        print(f"📊 Training on {len(tool_trajectories)} Tool-Augmented trajectories")
        
        # Convert to ART format
        art_trajectories = self.convert_to_art_format(tool_trajectories)
        
        # Create training groups
        training_groups = self.create_training_groups(art_trajectories)
        
        if not training_groups:
            raise ValueError("No training groups created")
            
        # Run ART training
        print("🔥 Running ART training...")
        training_result = await self.model.train(training_groups)
        
        # Save training results
        training_data = {
            'timestamp': self.timestamp,
            'model_name': self.model_name,
            'num_trajectories': len(tool_trajectories),
            'num_groups': len(training_groups),
            'reward_stats': self._calculate_reward_stats(art_trajectories),
            'training_result': training_result,
            'config': self.config
        }
        
        # Save to file
        training_file = self.results_dir / "data" / f"art_training_{self.timestamp}.json"
        with open(training_file, 'w') as f:
            json.dump(training_data, f, indent=2)
            
        print(f"💾 Training results saved: {training_file}")
        return training_data
    
    def _calculate_reward_stats(self, trajectories: List[Trajectory]) -> Dict[str, float]:
        """Calculate reward statistics"""
        rewards = [t.reward for t in trajectories]
        return {
            'mean_reward': float(np.mean(rewards)),
            'std_reward': float(np.std(rewards)),
            'min_reward': float(np.min(rewards)),
            'max_reward': float(np.max(rewards)),
            'positive_reward_ratio': float(np.mean([r > 0 for r in rewards]))
        }
    
    def generate_training_report(self, training_data: Dict[str, Any]) -> Path:
        """Generate comprehensive training report"""
        report_path = self.results_dir / "reports" / f"art_training_report_{self.timestamp}.md"
        
        with open(report_path, 'w') as f:
            f.write(f"# ART Training Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Model:** {training_data['model_name']}\n")
            f.write(f"**Trajectories:** {training_data['num_trajectories']}\n")
            f.write(f"**Training Groups:** {training_data['num_groups']}\n\n")
            
            f.write("## Reward Statistics\n\n")
            stats = training_data['reward_stats']
            f.write(f"- **Mean Reward:** {stats['mean_reward']:.2f}\n")
            f.write(f"- **Reward Std:** {stats['std_reward']:.2f}\n") 
            f.write(f"- **Reward Range:** [{stats['min_reward']:.2f}, {stats['max_reward']:.2f}]\n")
            f.write(f"- **Positive Reward Ratio:** {stats['positive_reward_ratio']:.1%}\n\n")
            
            f.write("## Training Configuration\n\n")
            f.write(f"```json\n{json.dumps(training_data['config'], indent=2)}\n```\n\n")
            
            f.write("## Next Steps\n\n")
            f.write("1. Evaluate trained model on test scenarios\n")
            f.write("2. Compare performance vs baseline Tool-Augmented controller\n")
            f.write("3. Run comprehensive experiments with ART-trained controller\n")
            
        print(f"📝 Training report saved: {report_path}")
        return report_path


async def main():
    """Main training script"""
    print("🚀 ART Training Pipeline for Agentic Control")
    print("=" * 60)
    
    # Training configuration
    training_config = {
        'art_config': {
            'max_iterations': 10,
            'batch_size': 4,
        },
        'rewards': {
            'w_success': 100.0,
            'w_efficiency': -1.0, 
            'w_smoothness': -0.1,
            'w_final_error': -50.0
        }
    }
    
    # Initialize trainer
    trainer = ARTControlTrainer(
        model_name="gpt-3.5-turbo",
        config=training_config
    )
    
    try:
        # Load existing trajectory data
        trajectory_data = trainer.load_trajectory_data()
        
        # Run training
        training_results = await trainer.train_model(trajectory_data)
        
        # Generate report
        report_path = trainer.generate_training_report(training_results)
        
        print("\n✅ ART Training Completed Successfully!")
        print(f"📊 Results: {len(training_results)} training records")
        print(f"📝 Report: {report_path.name}")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())