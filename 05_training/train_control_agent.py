#!/usr/bin/env python3
"""
ART Control Agent Training Script
=================================

Full training pipeline for agentic control using ART (Agent Reinforcement Trainer).
Adapted from ART•E reference implementation for control systems.

Usage:
    python train_control_agent.py --episodes 100 --difficulty medium
    python train_control_agent.py --eval_only --model_path ./models/control-agent-v1
"""

import asyncio
import argparse
import os
import sys
import time
import json
from typing import List, Dict, Any

# Add necessary paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '01_basic_physics'))

# ART imports
import art
from art import TrainableModel, TrajectoryGroup, TrainConfig
from art.local import LocalBackend
from art.utils import iterate_dataset

# Our imports
from art_integration import (
    ARTControlTrainer, ControlScenario, ControlPolicyConfig,
    direct_control_rollout, tool_augmented_rollout
)
from ruler_rewards import ControlRewardManager


class ControlTrainingManager:
    """Manages the full ART training pipeline for control agents"""
    
    def __init__(self, 
                 model_name: str = "control-agent-v1",
                 approach: str = "direct",  # "direct" or "tool_augmented" 
                 training_episodes: int = 200,
                 difficulty_levels: List[str] = None,
                 use_ruler_rewards: bool = True):
        
        self.model_name = model_name
        self.approach = approach
        self.training_episodes = training_episodes
        self.difficulty_levels = difficulty_levels or ["easy", "medium", "hard"]
        self.use_ruler_rewards = use_ruler_rewards
        
        # ART components
        self.model = None
        self.backend = None
        self.reward_manager = None
        
        # Training stats
        self.training_stats = {
            'total_episodes': 0,
            'successful_episodes': 0,
            'average_reward': 0.0,
            'training_steps': 0,
            'best_success_rate': 0.0,
            'reward_method': 'RULER' if use_ruler_rewards else 'Manual'
        }
        
    async def initialize_training(self):
        """Initialize ART model and backend for training"""
        print(f"🚀 Initializing ART Training Pipeline")
        print(f"   Model: {self.model_name}")
        print(f"   Approach: {self.approach}")
        print(f"   Episodes: {self.training_episodes}")
        print()
        
        # Create ART model configuration
        config = ControlPolicyConfig(
            groups_per_step=8,  # More groups for better training signal
            trajectories_per_group=4,  # Multiple trajectories per scenario
            learning_rate=1e-5,
            num_epochs=5,
            eval_steps=20
        )
        
        # Initialize model
        self.model = TrainableModel(
            name=self.model_name,
            project="agentic_control_training",
            base_model="Qwen/Qwen2.5-1.5B-Instruct",
            inference_model_name="Qwen/Qwen2.5-1.5B-Instruct", 
            config=config
        )
        
        # Initialize backend
        self.backend = LocalBackend()
        await self.model.register(self.backend)
        
        # Initialize reward manager
        self.reward_manager = ControlRewardManager(
            use_ruler=self.use_ruler_rewards,
            judge_model="openai/gpt-4o-mini",  # Economical option
            fallback_to_manual=True
        )
        
        print("✅ ART model initialized and registered")
        print(f"✅ Reward system: {self.training_stats['reward_method']}")
        
    def create_training_scenarios(self) -> List[ControlScenario]:
        """Create diverse training scenarios across difficulty levels"""
        scenarios = []
        
        # Easy scenarios - small errors, quick convergence
        if "easy" in self.difficulty_levels:
            scenarios.extend([
                ControlScenario("easy_pos_close", 0.15, 0.05, max_steps=20),
                ControlScenario("easy_neg_close", -0.12, 0.03, max_steps=20),
                ControlScenario("easy_vel_small", 0.08, 0.15, max_steps=25),
                ControlScenario("easy_vel_neg", -0.05, -0.12, max_steps=25),
                ControlScenario("easy_mixed", 0.1, -0.08, max_steps=25),
            ])
            
        # Medium scenarios - moderate complexity
        if "medium" in self.difficulty_levels:
            scenarios.extend([
                ControlScenario("med_pos_1", 0.4, 0.1, max_steps=35, difficulty="medium"),
                ControlScenario("med_pos_2", -0.35, -0.08, max_steps=35, difficulty="medium"),
                ControlScenario("med_vel_1", 0.15, 0.35, max_steps=40, difficulty="medium"),
                ControlScenario("med_vel_2", -0.1, -0.3, max_steps=40, difficulty="medium"),
                ControlScenario("med_mixed_1", 0.3, -0.25, max_steps=45, difficulty="medium"),
                ControlScenario("med_mixed_2", -0.25, 0.3, max_steps=45, difficulty="medium"),
            ])
            
        # Hard scenarios - large errors, high complexity
        if "hard" in self.difficulty_levels:
            scenarios.extend([
                ControlScenario("hard_far_1", 0.8, -0.2, max_steps=60, difficulty="hard"),
                ControlScenario("hard_far_2", -0.7, 0.4, max_steps=60, difficulty="hard"),
                ControlScenario("hard_vel_high", 0.3, 0.6, max_steps=55, difficulty="hard"),
                ControlScenario("hard_vel_neg", -0.2, -0.55, max_steps=55, difficulty="hard"),
                ControlScenario("hard_complex_1", 0.6, -0.45, max_steps=70, difficulty="hard"),
                ControlScenario("hard_complex_2", -0.55, 0.5, max_steps=70, difficulty="hard"),
            ])
        
        print(f"📊 Created {len(scenarios)} training scenarios:")
        difficulty_counts = {}
        for s in scenarios:
            difficulty_counts[s.difficulty] = difficulty_counts.get(s.difficulty, 0) + 1
        for diff, count in difficulty_counts.items():
            print(f"   {diff}: {count} scenarios")
        print()
        
        return scenarios
        
    async def run_rollout(self, scenario: ControlScenario):
        """Run single rollout using selected approach"""
        if self.approach == "direct":
            return await direct_control_rollout(self.model, scenario)
        elif self.approach == "tool_augmented":
            return await tool_augmented_rollout(self.model, scenario)
        else:
            raise ValueError(f"Unknown approach: {self.approach}")
    
    async def training_step(self, scenarios: List[ControlScenario], step: int):
        """Execute one training step with trajectory collection and model update"""
        print(f"🔄 Training Step {step}")
        
        # Generate trajectory groups
        print(f"   📊 Generating trajectories for {len(scenarios)} scenarios...")
        
        trajectory_groups = []
        
        for scenario in scenarios:
            print(f"      🎯 {scenario}")
            
            # Generate multiple trajectories per scenario (group)
            trajectories = []
            for traj_idx in range(self.model.config.trajectories_per_group):
                try:
                    # Run rollout (initial reward will be set by rollout function)
                    trajectory = await self.run_rollout(scenario)
                    
                    # Assign RULER-based reward if enabled
                    if self.reward_manager:
                        trajectory = await self.reward_manager.assign_rewards_to_trajectory(trajectory, scenario)
                    
                    trajectories.append(trajectory)
                    
                    # Log trajectory results
                    success = trajectory.metadata.get('success', False)
                    reward = trajectory.reward
                    steps = trajectory.metadata.get('steps', 0)
                    ruler_score = trajectory.metadata.get('ruler_score', 0)
                    print(f"         Traj {traj_idx+1}: Success={success}, Reward={reward:.3f}, Steps={steps}, RULER={ruler_score:.0f}")
                    
                except Exception as e:
                    print(f"         ❌ Traj {traj_idx+1} failed: {e}")
                    continue
            
            if trajectories:
                # Create group and optionally apply group-level RULER scoring
                group = TrajectoryGroup(trajectories)
                if self.reward_manager and self.use_ruler_rewards:
                    # Apply group-level consistency scoring
                    group = await self.reward_manager.assign_rewards_to_group(group, [scenario])
                
                trajectory_groups.append(group)
                
                # Update stats
                successes = sum(1 for t in trajectories if t.metadata.get('success', False))
                self.training_stats['total_episodes'] += len(trajectories)
                self.training_stats['successful_episodes'] += successes
                avg_reward = sum(t.reward for t in trajectories) / len(trajectories)
                
                print(f"      ✅ Group: {successes}/{len(trajectories)} successful, Avg Reward: {avg_reward:.3f}")
        
        if not trajectory_groups:
            print("   ❌ No successful trajectory groups - skipping training step")
            return False
            
        print(f"   🧠 Training on {len(trajectory_groups)} trajectory groups...")
        
        # Train the model
        try:
            await self.model.train(
                trajectory_groups,
                config=TrainConfig(learning_rate=self.model.config.learning_rate)
            )
            
            self.training_stats['training_steps'] += 1
            
            # Calculate success rate
            success_rate = (self.training_stats['successful_episodes'] / 
                          max(1, self.training_stats['total_episodes']))
            self.training_stats['best_success_rate'] = max(
                self.training_stats['best_success_rate'], success_rate
            )
            
            print(f"   ✅ Training step completed")
            print(f"      Success rate: {success_rate:.3f} (Best: {self.training_stats['best_success_rate']:.3f})")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Training failed: {e}")
            return False
    
    async def evaluate_model(self, test_scenarios: List[ControlScenario]):
        """Evaluate trained model on test scenarios"""
        print(f"🔍 Evaluating model on {len(test_scenarios)} test scenarios")
        
        results = []
        
        for scenario in test_scenarios:
            try:
                trajectory = await self.run_rollout(scenario)
                
                result = {
                    'scenario': scenario.name,
                    'difficulty': scenario.difficulty,
                    'success': trajectory.metadata.get('success', False),
                    'reward': trajectory.reward,
                    'steps': trajectory.metadata.get('steps', 0),
                    'final_pos_error': trajectory.metadata.get('final_pos_error', 0),
                    'final_vel_error': trajectory.metadata.get('final_vel_error', 0),
                    'control_effort': trajectory.metadata.get('control_effort', 0)
                }
                
                results.append(result)
                
                print(f"   {scenario.name}: Success={result['success']}, Reward={result['reward']:.3f}")
                
            except Exception as e:
                print(f"   ❌ {scenario.name}: Failed - {e}")
                results.append({
                    'scenario': scenario.name,
                    'success': False,
                    'error': str(e)
                })
        
        # Calculate evaluation metrics
        successful = [r for r in results if r.get('success', False)]
        success_rate = len(successful) / len(results)
        avg_reward = sum(r.get('reward', 0) for r in results) / len(results)
        
        eval_summary = {
            'total_scenarios': len(results),
            'successful_scenarios': len(successful),
            'success_rate': success_rate,
            'average_reward': avg_reward,
            'results': results
        }
        
        print(f"\n📊 Evaluation Results:")
        print(f"   Success Rate: {success_rate:.3f} ({len(successful)}/{len(results)})")
        print(f"   Average Reward: {avg_reward:.3f}")
        
        # Success rate by difficulty
        difficulty_stats = {}
        for result in results:
            diff = result.get('difficulty', 'unknown')
            if diff not in difficulty_stats:
                difficulty_stats[diff] = {'total': 0, 'successful': 0}
            difficulty_stats[diff]['total'] += 1
            if result.get('success', False):
                difficulty_stats[diff]['successful'] += 1
        
        for diff, stats in difficulty_stats.items():
            rate = stats['successful'] / max(1, stats['total'])
            print(f"   {diff.capitalize()} Success Rate: {rate:.3f} ({stats['successful']}/{stats['total']})")
        
        return eval_summary
    
    async def run_training_pipeline(self):
        """Run the complete training pipeline"""
        print("🚀 Starting ART Control Agent Training Pipeline")
        print("=" * 70)
        
        # Initialize
        await self.initialize_training()
        
        # Create scenarios
        training_scenarios = self.create_training_scenarios()
        
        # Create test scenarios (subset for evaluation)
        test_scenarios = training_scenarios[:6]  # Use first few for quick evaluation
        
        # Training loop
        print("🧠 Starting Training Loop")
        print("-" * 50)
        
        step = 0
        episodes_per_step = len(training_scenarios) * self.model.config.trajectories_per_group
        
        while self.training_stats['total_episodes'] < self.training_episodes:
            step += 1
            
            print(f"\n📈 Training Step {step} (Episodes: {self.training_stats['total_episodes']}/{self.training_episodes})")
            
            # Run training step
            success = await self.training_step(training_scenarios, step)
            
            if not success:
                print("⚠️ Training step failed, continuing...")
                continue
            
            # Evaluate periodically
            if step % self.model.config.eval_steps == 0:
                print(f"\n🔍 Running evaluation at step {step}...")
                eval_results = await self.evaluate_model(test_scenarios)
                
                # Save evaluation results
                eval_path = f"evaluation_step_{step}.json"
                with open(eval_path, 'w') as f:
                    json.dump(eval_results, f, indent=2)
                print(f"   📁 Evaluation results saved: {eval_path}")
        
        print(f"\n🎉 Training Complete!")
        print(f"   Total Episodes: {self.training_stats['total_episodes']}")
        print(f"   Training Steps: {self.training_stats['training_steps']}")
        print(f"   Best Success Rate: {self.training_stats['best_success_rate']:.3f}")
        
        # Final evaluation
        print(f"\n🏁 Final Evaluation")
        final_eval = await self.evaluate_model(test_scenarios)
        
        # Save final results
        final_results = {
            'training_stats': self.training_stats,
            'final_evaluation': final_eval,
            'model_name': self.model_name,
            'approach': self.approach,
            'total_episodes': self.training_episodes
        }
        
        results_path = f"final_results_{self.model_name}_{int(time.time())}.json"
        with open(results_path, 'w') as f:
            json.dump(final_results, f, indent=2)
        
        print(f"📁 Final results saved: {results_path}")
        
        return final_results


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='ART Control Agent Training')
    parser.add_argument('--episodes', type=int, default=100, 
                       help='Number of training episodes (default: 100)')
    parser.add_argument('--approach', choices=['direct', 'tool_augmented'], 
                       default='direct',
                       help='Control approach: direct LLM or tool-augmented')
    parser.add_argument('--difficulty', choices=['easy', 'medium', 'hard', 'all'], 
                       default='all',
                       help='Difficulty levels to train on')
    parser.add_argument('--model_name', type=str, default='control-agent-v1',
                       help='Model name for training')
    parser.add_argument('--eval_only', action='store_true',
                       help='Run evaluation only (no training)')
    
    args = parser.parse_args()
    
    # Setup difficulty levels
    if args.difficulty == 'all':
        difficulty_levels = ['easy', 'medium', 'hard']
    else:
        difficulty_levels = [args.difficulty]
    
    # Create training manager
    trainer = ControlTrainingManager(
        model_name=args.model_name,
        approach=args.approach,
        training_episodes=args.episodes,
        difficulty_levels=difficulty_levels
    )
    
    if args.eval_only:
        # Evaluation only mode
        print("🔍 Running evaluation-only mode")
        await trainer.initialize_training()
        scenarios = trainer.create_training_scenarios()
        results = await trainer.evaluate_model(scenarios[:10])  # Evaluate on subset
        print(f"\n✅ Evaluation complete")
    else:
        # Full training pipeline
        results = await trainer.run_training_pipeline()
        print(f"\n✅ Training pipeline complete")
    
    # Cleanup
    if trainer.backend:
        await trainer.backend.cleanup()


if __name__ == "__main__":
    asyncio.run(main())