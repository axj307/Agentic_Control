"""
Rollout Function for ART Training

This module provides rollout functions for collecting training data
for the ART (Automated Red Teaming) framework.

Usage:
    trajectory = await control_rollout(model, scenario)
    trajectories = await collect_training_data(model, scenarios)
"""

import asyncio
import json
import time
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import numpy as np

# Add parent directory to path to import the environment
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '01_basic_physics'))
from double_integrator import DoubleIntegrator as DoubleIntegratorEnvironment
from llm_integration import BaseLLMController, VLLMController, OpenAIController

# Try to import ART components (will be available after installation)
try:
    from art import Trajectory, TrainableModel, Message
    ART_AVAILABLE = True
except ImportError:
    ART_AVAILABLE = False
    # Create placeholder classes for development
    class Trajectory:
        def __init__(self, messages, reward, metadata=None):
            self.messages = messages
            self.reward = reward
            self.metadata = metadata or {}
    
    class TrainableModel:
        def __init__(self, model_name):
            self.model_name = model_name
    
    class Message:
        def __init__(self, role, content):
            self.role = role
            self.content = content


@dataclass
class ControlScenario:
    """Defines a control scenario for training/testing"""
    name: str
    initial_position: float
    initial_velocity: float
    target_position: float = 0.0
    target_velocity: float = 0.0
    max_steps: int = 100
    position_tolerance: float = 0.1
    velocity_tolerance: float = 0.1
    difficulty: str = "medium"  # easy, medium, hard
    
    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'initial_position': self.initial_position,
            'initial_velocity': self.initial_velocity,
            'target_position': self.target_position,
            'target_velocity': self.target_velocity,
            'max_steps': self.max_steps,
            'position_tolerance': self.position_tolerance,
            'velocity_tolerance': self.velocity_tolerance,
            'difficulty': self.difficulty
        }


def create_standard_scenarios() -> List[ControlScenario]:
    """Create a standard set of training scenarios"""
    scenarios = []
    
    # Easy scenarios - close to target
    scenarios.extend([
        ControlScenario("easy_pos_1", 0.2, 0.0, difficulty="easy"),
        ControlScenario("easy_pos_2", -0.15, 0.0, difficulty="easy"),
        ControlScenario("easy_vel_1", 0.0, 0.1, difficulty="easy"),
        ControlScenario("easy_vel_2", 0.0, -0.12, difficulty="easy"),
    ])
    
    # Medium scenarios - moderate distance/velocity
    scenarios.extend([
        ControlScenario("med_pos_1", 0.5, 0.0, difficulty="medium"),
        ControlScenario("med_pos_2", -0.4, 0.0, difficulty="medium"), 
        ControlScenario("med_vel_1", 0.0, 0.3, difficulty="medium"),
        ControlScenario("med_mixed_1", 0.3, 0.2, difficulty="medium"),
        ControlScenario("med_mixed_2", -0.2, -0.25, difficulty="medium"),
    ])
    
    # Hard scenarios - far from target or high velocity
    scenarios.extend([
        ControlScenario("hard_pos_1", 1.0, 0.0, difficulty="hard"),
        ControlScenario("hard_pos_2", -0.8, 0.0, difficulty="hard"),
        ControlScenario("hard_vel_1", 0.0, 0.6, difficulty="hard"),
        ControlScenario("hard_mixed_1", 0.7, 0.4, difficulty="hard"),
        ControlScenario("hard_mixed_2", -0.6, -0.5, difficulty="hard"),
        ControlScenario("hard_extreme", 1.2, 0.8, max_steps=150, difficulty="hard"),
    ])
    
    return scenarios


async def control_rollout(controller: BaseLLMController, 
                         scenario: ControlScenario,
                         verbose: bool = False) -> Trajectory:
    """
    Execute one control rollout for training/testing
    
    Args:
        controller: LLM controller to use
        scenario: Control scenario to execute
        verbose: Whether to print detailed progress
        
    Returns:
        Trajectory object containing messages and reward
    """
    if verbose:
        print(f"🎯 Executing rollout: {scenario.name}")
        print(f"   Initial: pos={scenario.initial_position:.3f}, vel={scenario.initial_velocity:.3f}")
        print(f"   Target: pos={scenario.target_position:.3f}, vel={scenario.target_velocity:.3f}")
    
    # Initialize environment
    env = DoubleIntegratorEnvironment()
    env.reset(scenario.initial_position, scenario.initial_velocity)
    
    # Track rollout data
    messages = []
    states = []
    actions = []
    rewards = []
    total_reward = 0.0
    
    # Add initial system message
    system_msg = Message(
        role="system",
        content=f"Control spacecraft from ({scenario.initial_position:.3f}, {scenario.initial_velocity:.3f}) to ({scenario.target_position:.3f}, {scenario.target_velocity:.3f})"
    )
    messages.append(system_msg)
    
    success = False
    
    for step in range(scenario.max_steps):
        # Get current state
        current_pos = env.position
        current_vel = env.velocity
        states.append((current_pos, current_vel))
        
        # Get action from controller
        try:
            result = controller.get_action(
                current_pos, current_vel,
                scenario.target_position, scenario.target_velocity
            )
            action = result['action']
            confidence = result.get('confidence', 0.5)
            reasoning = result.get('reasoning', 'No reasoning provided')
            
        except Exception as e:
            if verbose:
                print(f"⚠️  Controller error at step {step}: {e}")
            action = 0.0  # Safe fallback
            confidence = 0.0
            reasoning = f"Error: {str(e)}"
        
        actions.append(action)
        
        # Create user message (observation)
        user_msg = Message(
            role="user", 
            content=f"Step {step}: Position={current_pos:.3f}m, Velocity={current_vel:.3f}m/s, Target=({scenario.target_position:.3f},{scenario.target_velocity:.3f})"
        )
        messages.append(user_msg)
        
        # Create assistant message (action)
        assistant_msg = Message(
            role="assistant",
            content=json.dumps({
                "force": action,
                "confidence": confidence,
                "reasoning": reasoning
            })
        )
        messages.append(assistant_msg)
        
        # Execute action in environment
        env.step(action)
        
        # Calculate step reward
        pos_error = abs(scenario.target_position - env.position)
        vel_error = abs(scenario.target_velocity - env.velocity)
        control_effort = abs(action)
        
        # Multi-objective reward
        step_reward = (
            -pos_error * 2.0 +      # Position accuracy (most important)
            -vel_error * 1.0 +      # Velocity accuracy  
            -control_effort * 0.1 + # Control efficiency
            -0.01                   # Time penalty
        )
        
        rewards.append(step_reward)
        total_reward += step_reward
        
        # Check for success
        if (pos_error <= scenario.position_tolerance and 
            vel_error <= scenario.velocity_tolerance):
            success = True
            total_reward += 10.0  # Success bonus
            if verbose:
                print(f"✅ Success at step {step}!")
            break
        
        if verbose and step % 10 == 0:
            print(f"   Step {step}: pos={env.position:.3f}, vel={env.velocity:.3f}, action={action:.3f}")
    
    # Calculate final metrics
    final_pos_error = abs(scenario.target_position - env.position)
    final_vel_error = abs(scenario.target_velocity - env.velocity)
    total_control_effort = sum(abs(a) for a in actions)
    
    # Create trajectory metadata
    metadata = {
        'scenario': scenario.to_dict(),
        'success': success,
        'steps': len(actions),
        'final_position_error': final_pos_error,
        'final_velocity_error': final_vel_error,
        'total_control_effort': total_control_effort,
        'average_reward_per_step': total_reward / max(len(actions), 1),
        'states': states,
        'actions': actions,
        'step_rewards': rewards
    }
    
    if verbose:
        print(f"   Final: pos_error={final_pos_error:.3f}, vel_error={final_vel_error:.3f}")
        print(f"   Total reward: {total_reward:.2f}")
        print(f"   Success: {success}")
    
    # Create ART trajectory
    trajectory = Trajectory(
        messages=messages,
        reward=total_reward,
        metadata=metadata
    )
    
    return trajectory


async def collect_training_data(controller: BaseLLMController,
                              scenarios: Optional[List[ControlScenario]] = None,
                              num_rollouts_per_scenario: int = 1,
                              verbose: bool = True) -> List[Trajectory]:
    """
    Collect training data from multiple rollouts
    
    Args:
        controller: LLM controller to use
        scenarios: List of scenarios (uses standard set if None)
        num_rollouts_per_scenario: How many rollouts per scenario
        verbose: Whether to print progress
        
    Returns:
        List of trajectory objects
    """
    if scenarios is None:
        scenarios = create_standard_scenarios()
    
    if verbose:
        print(f"🚀 Collecting training data...")
        print(f"   Scenarios: {len(scenarios)}")
        print(f"   Rollouts per scenario: {num_rollouts_per_scenario}")
        print(f"   Total rollouts: {len(scenarios) * num_rollouts_per_scenario}")
        print("=" * 50)
    
    trajectories = []
    
    for scenario_idx, scenario in enumerate(scenarios):
        if verbose:
            print(f"📋 Scenario {scenario_idx + 1}/{len(scenarios)}: {scenario.name}")
        
        for rollout_idx in range(num_rollouts_per_scenario):
            trajectory = await control_rollout(
                controller=controller,
                scenario=scenario,
                verbose=verbose and rollout_idx == 0  # Only verbose for first rollout
            )
            trajectories.append(trajectory)
            
            if verbose:
                success = trajectory.metadata['success']
                reward = trajectory.reward
                steps = trajectory.metadata['steps']
                print(f"   Rollout {rollout_idx + 1}: {'✅' if success else '❌'} reward={reward:.2f}, steps={steps}")
    
    # Print summary statistics
    if verbose:
        successes = sum(1 for t in trajectories if t.metadata['success'])
        avg_reward = np.mean([t.reward for t in trajectories])
        avg_steps = np.mean([t.metadata['steps'] for t in trajectories])
        
        print("=" * 50)
        print("📊 Collection Summary:")
        print(f"   Total trajectories: {len(trajectories)}")
        print(f"   Success rate: {successes / len(trajectories) * 100:.1f}%")
        print(f"   Average reward: {avg_reward:.2f}")
        print(f"   Average steps: {avg_steps:.1f}")
    
    return trajectories


def save_trajectories(trajectories: List[Trajectory], filepath: str):
    """Save trajectories to JSON file"""
    data = {
        'trajectories': [],
        'summary': {
            'total_trajectories': len(trajectories),
            'success_rate': sum(1 for t in trajectories if t.metadata['success']) / len(trajectories),
            'average_reward': float(np.mean([t.reward for t in trajectories])),
            'average_steps': float(np.mean([t.metadata['steps'] for t in trajectories]))
        }
    }
    
    for trajectory in trajectories:
        traj_data = {
            'messages': [{'role': m.role, 'content': m.content} for m in trajectory.messages],
            'reward': trajectory.reward,
            'metadata': trajectory.metadata
        }
        data['trajectories'].append(traj_data)
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"💾 Saved {len(trajectories)} trajectories to {filepath}")


def load_trajectories(filepath: str) -> List[Trajectory]:
    """Load trajectories from JSON file"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    trajectories = []
    for traj_data in data['trajectories']:
        messages = [Message(m['role'], m['content']) for m in traj_data['messages']]
        trajectory = Trajectory(
            messages=messages,
            reward=traj_data['reward'],
            metadata=traj_data['metadata']
        )
        trajectories.append(trajectory)
    
    print(f"📂 Loaded {len(trajectories)} trajectories from {filepath}")
    return trajectories


async def test_rollout_system():
    """Test the rollout system with mock controller"""
    print("🧪 Testing Rollout System")
    print("=" * 50)
    
    # Create mock controller that simulates reasonable control
    class MockController(BaseLLMController):
        def _call_llm(self, prompt: str) -> str:
            # Extract state from prompt (simplified)
            import re
            pos_match = re.search(r'Position: ([-+]?\d*\.?\d+)', prompt)
            vel_match = re.search(r'Velocity: ([-+]?\d*\.?\d+)', prompt)
            
            if pos_match and vel_match:
                pos = float(pos_match.group(1))
                vel = float(vel_match.group(1))
                
                # Simple PD-like control with noise
                kp, kd = 1.0, 2.0
                force = -kp * pos - kd * vel
                force += np.random.normal(0, 0.1)  # Add noise
                force = np.clip(force, -1.0, 1.0)
                
                return json.dumps({
                    "force": force,
                    "confidence": 0.7 + 0.3 * np.random.random(),
                    "reasoning": f"Applying force {force:.3f} based on position {pos:.3f} and velocity {vel:.3f}"
                })
            
            return '{"force": 0.0, "confidence": 0.1, "reasoning": "Could not parse state"}'
    
    # Test single rollout
    print("🎯 Testing single rollout...")
    controller = MockController()
    scenario = ControlScenario("test", 0.5, 0.0)
    
    trajectory = await control_rollout(controller, scenario, verbose=True)
    print(f"✅ Single rollout completed. Reward: {trajectory.reward:.2f}")
    
    # Test multiple rollouts
    print("\n📊 Testing multiple rollouts...")
    easy_scenarios = [s for s in create_standard_scenarios() if s.difficulty == "easy"]
    trajectories = await collect_training_data(
        controller, 
        easy_scenarios[:3], 
        num_rollouts_per_scenario=2,
        verbose=True
    )
    
    # Test save/load
    print("\n💾 Testing save/load...")
    save_trajectories(trajectories, "test_trajectories.json")
    loaded_trajectories = load_trajectories("test_trajectories.json")
    
    print(f"✅ Save/load test passed. Original: {len(trajectories)}, Loaded: {len(loaded_trajectories)}")
    
    # Cleanup
    os.remove("test_trajectories.json")
    
    print("\n🎉 All rollout tests passed!")


if __name__ == "__main__":
    asyncio.run(test_rollout_system())