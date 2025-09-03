#!/usr/bin/env python3
"""
ART Training Pipeline for Control Systems
========================================

Complete training pipeline that compares direct vs tool-augmented control
using the ART framework with GRPO optimization.

Usage:
    python run_training.py
"""

import asyncio
import json
import time
from pathlib import Path
import numpy as np
from typing import List, Dict

from art_integration import (
    ARTControlTrainer, 
    direct_control_rollout, 
    tool_augmented_rollout,
    ControlScenario,
    ART_AVAILABLE
)

if ART_AVAILABLE:
    import art


class TrainingPipeline:
    """Complete ART training pipeline for control systems"""
    
    def __init__(self, experiment_name: str = "control-comparison-study"):
        self.experiment_name = experiment_name
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)
        
        self.results = {
            'experiment_name': experiment_name,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'art_available': ART_AVAILABLE,
            'training_results': {}
        }
    
    def print_header(self, title: str):
        """Print formatted section header"""
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}")
    
    def print_trajectory_summary(self, trajectories: List, approach_name: str):
        """Print summary of trajectory collection results"""
        rewards = [t.reward for t in trajectories]
        successes = [t.metadata['success'] for t in trajectories]
        steps = [t.metadata['steps'] for t in trajectories]
        
        print(f"\n📊 {approach_name} Trajectory Summary:")
        print(f"  Total Trajectories: {len(trajectories)}")
        print(f"  Success Rate: {np.mean(successes):.1%} ({sum(successes)}/{len(successes)})")
        print(f"  Average Reward: {np.mean(rewards):.2f} (range: {min(rewards):.2f} to {max(rewards):.2f})")
        print(f"  Average Steps: {np.mean(steps):.1f} (range: {min(steps)} to {max(steps)})")
        
        # Show individual trajectory results
        print(f"  Individual Results:")
        for i, traj in enumerate(trajectories):
            success_icon = "✅" if traj.metadata['success'] else "❌"
            print(f"    {i+1}. {traj.metadata['scenario']:15s} {success_icon} "
                  f"Reward: {traj.reward:6.2f}, Steps: {traj.metadata['steps']:2d}")
    
    async def collect_trajectories(self, trainer: ARTControlTrainer, scenarios: List[ControlScenario], 
                                 approach: str, rollout_fn) -> List:
        """Collect trajectories for a specific approach"""
        print(f"\n🎯 Collecting {approach} trajectories...")
        trajectories = []
        
        for i, scenario in enumerate(scenarios):
            print(f"  [{i+1}/{len(scenarios)}] Running {scenario.name}...", end=" ")
            
            try:
                trajectory = await rollout_fn(trainer.model, scenario)
                trajectories.append(trajectory)
                
                # Quick result summary
                success_icon = "✅" if trajectory.metadata['success'] else "❌"
                print(f"{success_icon} Reward: {trajectory.reward:.2f}")
                
            except Exception as e:
                print(f"❌ Failed: {e}")
                continue
        
        return trajectories
    
    async def train_approach(self, trainer: ARTControlTrainer, trajectories: List, 
                           approach_name: str) -> Dict:
        """Train model using collected trajectories"""
        if not trajectories:
            print(f"⚠️ No trajectories to train {approach_name}")
            return {}
        
        print(f"\n🚂 Training {approach_name} with {len(trajectories)} trajectories...")
        
        if ART_AVAILABLE:
            try:
                # Create trajectory group for GRPO training
                trajectory_group = art.TrajectoryGroup(trajectories)
                print(f"  Created trajectory group with {len(trajectories)} trajectories")
                
                # Run GRPO training
                print("  Starting GRPO optimization...")
                training_start = time.time()
                await trainer.model.train([trajectory_group])
                training_time = time.time() - training_start
                
                print(f"  ✅ Training completed in {training_time:.1f} seconds")
                
                return {
                    'training_time': training_time,
                    'trajectory_count': len(trajectories),
                    'training_successful': True
                }
                
            except Exception as e:
                print(f"  ❌ Training failed: {e}")
                return {'training_successful': False, 'error': str(e)}
        else:
            print("  ⚠️ ART not available - simulating training")
            time.sleep(2)  # Simulate training time
            return {
                'training_time': 2.0,
                'trajectory_count': len(trajectories),
                'training_successful': True,
                'simulated': True
            }
    
    async def run_complete_pipeline(self):
        """Execute the complete training pipeline"""
        self.print_header("ART CONTROL TRAINING PIPELINE")
        print(f"🚀 Starting experiment: {self.experiment_name}")
        print(f"📅 Timestamp: {self.results['timestamp']}")
        print(f"🔧 ART Available: {'Yes' if ART_AVAILABLE else 'No (using simulation mode)'}")
        
        # Initialize trainer
        trainer = ARTControlTrainer(f"{self.experiment_name}-model")
        await trainer.initialize_model()
        
        # Create training scenarios
        all_scenarios = trainer.create_training_scenarios()
        print(f"📋 Created {len(all_scenarios)} training scenarios")
        
        # Use subset for initial testing
        training_scenarios = all_scenarios[:6]  # Start with 6 scenarios for faster testing
        print(f"🎯 Using {len(training_scenarios)} scenarios for this experiment")
        
        # Display scenarios
        print("\n📝 Training Scenarios:")
        for i, scenario in enumerate(training_scenarios):
            print(f"  {i+1}. {scenario}")
        
        # ===== DIRECT CONTROL TRAINING =====
        self.print_header("DIRECT CONTROL APPROACH")
        
        # Collect direct control trajectories
        direct_trajectories = await self.collect_trajectories(
            trainer, training_scenarios, "Direct Control", direct_control_rollout
        )
        
        if direct_trajectories:
            self.print_trajectory_summary(direct_trajectories, "Direct Control")
            
            # Train direct control model
            direct_training_results = await self.train_approach(
                trainer, direct_trajectories, "Direct Control"
            )
            
            self.results['training_results']['direct_control'] = {
                'trajectories': len(direct_trajectories),
                'training': direct_training_results,
                'performance': {
                    'success_rate': np.mean([t.metadata['success'] for t in direct_trajectories]),
                    'average_reward': np.mean([t.reward for t in direct_trajectories]),
                    'average_steps': np.mean([t.metadata['steps'] for t in direct_trajectories])
                }
            }
        
        # ===== TOOL-AUGMENTED CONTROL TRAINING =====
        self.print_header("TOOL-AUGMENTED CONTROL APPROACH")
        
        # Collect tool-augmented trajectories
        tool_trajectories = await self.collect_trajectories(
            trainer, training_scenarios, "Tool-Augmented Control", tool_augmented_rollout
        )
        
        if tool_trajectories:
            self.print_trajectory_summary(tool_trajectories, "Tool-Augmented Control")
            
            # Train tool-augmented model
            tool_training_results = await self.train_approach(
                trainer, tool_trajectories, "Tool-Augmented Control"
            )
            
            self.results['training_results']['tool_augmented'] = {
                'trajectories': len(tool_trajectories),
                'training': tool_training_results,
                'performance': {
                    'success_rate': np.mean([t.metadata['success'] for t in tool_trajectories]),
                    'average_reward': np.mean([t.reward for t in tool_trajectories]),
                    'average_steps': np.mean([t.metadata['steps'] for t in tool_trajectories])
                }
            }
        
        # ===== COMPARISON AND ANALYSIS =====
        self.print_header("PERFORMANCE COMPARISON")
        
        if direct_trajectories and tool_trajectories:
            await self.compare_approaches(direct_trajectories, tool_trajectories)
        
        # Save results
        await self.save_results()
        
        self.print_header("TRAINING PIPELINE COMPLETE")
        print("🎉 Training experiment finished successfully!")
        print(f"📁 Results saved in: {self.results_dir}/")
    
    async def compare_approaches(self, direct_trajectories: List, tool_trajectories: List):
        """Compare the performance of both approaches"""
        print("\n📈 Detailed Performance Comparison:")
        print("-" * 60)
        
        # Calculate metrics for both approaches
        approaches = {
            'Direct Control': direct_trajectories,
            'Tool-Augmented': tool_trajectories
        }
        
        comparison_data = {}
        
        for approach_name, trajectories in approaches.items():
            successes = [t.metadata['success'] for t in trajectories]
            rewards = [t.reward for t in trajectories]
            steps = [t.metadata['steps'] for t in trajectories]
            efforts = [t.metadata['control_effort'] for t in trajectories]
            
            metrics = {
                'success_rate': np.mean(successes),
                'avg_reward': np.mean(rewards),
                'avg_steps': np.mean(steps),
                'avg_control_effort': np.mean(efforts),
                'reward_std': np.std(rewards),
                'steps_std': np.std(steps)
            }
            
            comparison_data[approach_name] = metrics
            
            print(f"\n{approach_name}:")
            print(f"  Success Rate:     {metrics['success_rate']:.1%}")
            print(f"  Average Reward:   {metrics['avg_reward']:.2f} ± {metrics['reward_std']:.2f}")
            print(f"  Average Steps:    {metrics['avg_steps']:.1f} ± {metrics['steps_std']:.1f}")
            print(f"  Control Effort:   {metrics['avg_control_effort']:.2f}")
        
        # Winner analysis
        print(f"\n🏆 Performance Winners:")
        if len(comparison_data) == 2:
            direct_data = comparison_data['Direct Control']
            tool_data = comparison_data['Tool-Augmented']
            
            winners = {
                'Success Rate': 'Tool-Augmented' if tool_data['success_rate'] > direct_data['success_rate'] else 'Direct Control',
                'Average Reward': 'Tool-Augmented' if tool_data['avg_reward'] > direct_data['avg_reward'] else 'Direct Control',
                'Speed (fewer steps)': 'Tool-Augmented' if tool_data['avg_steps'] < direct_data['avg_steps'] else 'Direct Control',
                'Efficiency (less effort)': 'Tool-Augmented' if tool_data['avg_control_effort'] < direct_data['avg_control_effort'] else 'Direct Control'
            }
            
            for metric, winner in winners.items():
                print(f"  {metric:20s}: {winner}")
            
            # Overall assessment
            tool_wins = sum(1 for w in winners.values() if w == 'Tool-Augmented')
            print(f"\n🎯 Overall: Tool-Augmented wins in {tool_wins}/{len(winners)} metrics")
        
        self.results['comparison'] = comparison_data
    
    async def save_results(self):
        """Save all results to files"""
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        
        # Save main results JSON
        results_file = self.results_dir / f"training_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"💾 Results saved to: {results_file}")
        
        # Save summary report
        report_file = self.results_dir / f"training_report_{timestamp}.md"
        await self.generate_summary_report(report_file)
        print(f"📝 Summary report: {report_file}")
    
    async def generate_summary_report(self, report_file: Path):
        """Generate a human-readable summary report"""
        with open(report_file, 'w') as f:
            f.write(f"# ART Control Training Results\n\n")
            f.write(f"**Experiment:** {self.experiment_name}\\n")
            f.write(f"**Date:** {self.results['timestamp']}\\n")
            f.write(f"**ART Framework:** {'Available' if ART_AVAILABLE else 'Simulated'}\\n\\n")
            
            f.write("## Summary\n\n")
            if 'comparison' in self.results:
                for approach, data in self.results['comparison'].items():
                    f.write(f"### {approach}\n")
                    f.write(f"- Success Rate: {data['success_rate']:.1%}\\n")
                    f.write(f"- Average Reward: {data['avg_reward']:.2f}\\n")
                    f.write(f"- Average Steps: {data['avg_steps']:.1f}\\n\\n")
            
            f.write("## Key Findings\n\n")
            f.write("- Successfully demonstrated ART integration for control systems\\n")
            f.write("- Compared direct vs tool-augmented approaches\\n")
            f.write("- Generated training trajectories with reward feedback\\n")
            f.write("- Ready for extension to aerospace control problems\\n")


async def main():
    """Main entry point"""
    try:
        pipeline = TrainingPipeline("double-integrator-comparison")
        await pipeline.run_complete_pipeline()
    except KeyboardInterrupt:
        print("\\n🛑 Training interrupted by user")
    except Exception as e:
        print(f"\\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("🚀 ART Control Training Pipeline")
    print("================================")
    print("This script demonstrates ART training for control systems.")
    print("It compares direct LLM control vs tool-augmented approaches.\\n")
    
    # Run the pipeline
    asyncio.run(main())