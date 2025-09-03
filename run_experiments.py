#!/usr/bin/env python3
"""
Clean Experiment Runner for Agentic Control
Simplified, config-driven experiments with enhanced visualization
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
import sys

# Import configuration and setup paths
from experiment_config import (
    DEFAULT_CONFIG, get_scenarios, create_results_directories, 
    setup_python_path, get_project_paths
)

# Setup paths and imports
setup_python_path()
from double_integrator import DoubleIntegrator
from control_graph import ToolAugmentedController

# Set matplotlib backend for headless operation
plt.switch_backend('Agg')

class CleanExperimentRunner:
    """Simplified experiment runner with enhanced visualization"""
    
    def __init__(self):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.config = DEFAULT_CONFIG.copy()
        self.results_dir = get_project_paths()["results"]
        create_results_directories()
        
    def create_pd_controller(self):
        """Create PD controller with configured gains"""
        class PDController:
            def __init__(self, config):
                self.kp = config["pd_gains"]["kp"]
                self.kd = config["pd_gains"]["kd"]
                
            def get_action(self, position, velocity, target_pos, target_vel=0.0):
                pos_error = target_pos - position
                vel_error = target_vel - velocity
                action = self.kp * pos_error + self.kd * vel_error
                return {
                    'action': np.clip(action, -1.0, 1.0),
                    'reasoning': f'PD: P={self.kp*pos_error:.3f}, D={self.kd*vel_error:.3f}'
                }
        
        return PDController(self.config)
    
    def run_single_trajectory(self, controller, scenario_config, max_steps=100):
        """Run a single control trajectory"""
        # Initialize environment
        env = DoubleIntegrator(
            max_force=self.config["max_force"],
            dt=self.config["dt"]
        )
        env.reset(scenario_config["init_pos"], scenario_config["init_vel"])
        
        # Track trajectory
        trajectory = {
            'scenario': scenario_config["name"],
            'positions': [env.position],
            'velocities': [env.velocity],
            'controls': [],
            'times': [0.0],
            'target_pos': scenario_config["target_pos"],
            'target_vel': scenario_config["target_vel"]
        }
        
        # Run simulation
        success = False
        for step in range(max_steps):
            # Get control action
            result = controller.get_action(
                env.position, env.velocity,
                scenario_config["target_pos"], scenario_config["target_vel"]
            )
            action = result['action']
            trajectory['controls'].append(action)
            
            # Step environment
            env.step(action)
            trajectory['positions'].append(env.position)
            trajectory['velocities'].append(env.velocity)
            trajectory['times'].append((step + 1) * env.dt)
            
            # Check success
            pos_error = abs(scenario_config["target_pos"] - env.position)
            vel_error = abs(scenario_config["target_vel"] - env.velocity)
            
            if (pos_error < self.config["position_tolerance"] and 
                vel_error < self.config["velocity_tolerance"]):
                success = True
                break
        
        # Add final metrics
        trajectory.update({
            'success': success,
            'steps': len(trajectory['controls']),
            'final_pos_error': abs(scenario_config["target_pos"] - env.position),
            'final_vel_error': abs(scenario_config["target_vel"] - env.velocity),
            'control_effort': sum(abs(c) for c in trajectory['controls'])
        })
        
        return trajectory
    
    def run_comparison_experiment(self, difficulty="all"):
        """Run PD vs Tool-Augmented comparison"""
        print(f"🚀 Running {difficulty} scenarios comparison...")
        
        scenarios = get_scenarios(difficulty)
        controllers = {
            "PD Baseline": self.create_pd_controller(),
            "Tool-Augmented": ToolAugmentedController()
        }
        
        all_results = {"PD Baseline": [], "Tool-Augmented": []}
        
        for scenario_config in scenarios:
            print(f"📍 Testing scenario: {scenario_config['name']}")
            
            for controller_name, controller in controllers.items():
                trajectory = self.run_single_trajectory(controller, scenario_config)
                all_results[controller_name].append(trajectory)
                
                status = "✅" if trajectory['success'] else "❌"
                print(f"   {controller_name}: {status} {trajectory['steps']} steps")
        
        return all_results, scenarios
    
    def create_enhanced_comparison_plot(self, results, scenarios):
        """Create enhanced comparison visualization"""
        print("📊 Creating enhanced comparison plots...")
        
        config = self.config["plot_config"]
        fig, axes = plt.subplots(2, 2, figsize=config["figsize"])
        fig.suptitle('PD Baseline vs Tool-Augmented Control Comparison', 
                     fontsize=config["fontsize_title"], fontweight='bold')
        
        colors = config["colors"]
        
        # Collect all trajectory data
        pd_trajectories = results["PD Baseline"]
        tool_trajectories = results["Tool-Augmented"]
        
        # Plot 1: Position trajectories
        ax1 = axes[0, 0]
        for i, (pd_traj, tool_traj) in enumerate(zip(pd_trajectories, tool_trajectories)):
            # Convert to numpy arrays and fix time calculation
            pd_time = np.array(pd_traj['times'])
            tool_time = np.array(tool_traj['times'])
            
            ax1.plot(pd_time, pd_traj['positions'], 
                    color=colors["pd"], linewidth=config["linewidth"], 
                    alpha=0.7, label='PD Baseline' if i == 0 else '')
            ax1.plot(tool_time, tool_traj['positions'], 
                    color=colors["tool"], linewidth=config["linewidth"], 
                    alpha=0.7, label='Tool-Augmented' if i == 0 else '')
            
            # Mark target
            ax1.axhline(y=pd_traj['target_pos'], color=colors["target"], 
                       linestyle='--', alpha=0.5, linewidth=2)
        
        ax1.set_xlabel('Time (s)', fontsize=config["fontsize_label"])
        ax1.set_ylabel('Position (m)', fontsize=config["fontsize_label"])
        ax1.set_title('Position Trajectories', fontsize=config["fontsize_label"])
        ax1.legend(fontsize=config["fontsize_tick"])
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(labelsize=config["fontsize_tick"])
        
        # Plot 2: Velocity trajectories  
        ax2 = axes[0, 1]
        for i, (pd_traj, tool_traj) in enumerate(zip(pd_trajectories, tool_trajectories)):
            pd_time = np.array(pd_traj['times'])
            tool_time = np.array(tool_traj['times'])
            
            ax2.plot(pd_time, pd_traj['velocities'],
                    color=colors["pd"], linewidth=config["linewidth"],
                    alpha=0.7, label='PD Baseline' if i == 0 else '')
            ax2.plot(tool_time, tool_traj['velocities'],
                    color=colors["tool"], linewidth=config["linewidth"], 
                    alpha=0.7, label='Tool-Augmented' if i == 0 else '')
            
            # Mark target velocity
            ax2.axhline(y=pd_traj['target_vel'], color=colors["target"], 
                       linestyle='--', alpha=0.5, linewidth=2)
        
        ax2.set_xlabel('Time (s)', fontsize=config["fontsize_label"])
        ax2.set_ylabel('Velocity (m/s)', fontsize=config["fontsize_label"])
        ax2.set_title('Velocity Trajectories', fontsize=config["fontsize_label"])
        ax2.legend(fontsize=config["fontsize_tick"])
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(labelsize=config["fontsize_tick"])
        
        # Plot 3: Control actions
        ax3 = axes[1, 0]
        for i, (pd_traj, tool_traj) in enumerate(zip(pd_trajectories, tool_trajectories)):
            # Control times (one less than positions)
            pd_time = np.array(pd_traj['times'][:-1])
            tool_time = np.array(tool_traj['times'][:-1])
            
            ax3.plot(pd_time, pd_traj['controls'],
                    color=colors["pd"], linewidth=config["linewidth"],
                    alpha=0.7, label='PD Baseline' if i == 0 else '')
            ax3.plot(tool_time, tool_traj['controls'],
                    color=colors["tool"], linewidth=config["linewidth"],
                    alpha=0.7, label='Tool-Augmented' if i == 0 else '')
        
        # Mark control limits
        ax3.axhline(y=self.config["max_force"], color=colors["limits"], 
                   linestyle=':', alpha=0.7, linewidth=2, label='Control Limits')
        ax3.axhline(y=-self.config["max_force"], color=colors["limits"], 
                   linestyle=':', alpha=0.7, linewidth=2)
        
        ax3.set_xlabel('Time (s)', fontsize=config["fontsize_label"])
        ax3.set_ylabel('Control Force (N)', fontsize=config["fontsize_label"])
        ax3.set_title('Control Actions', fontsize=config["fontsize_label"])
        ax3.legend(fontsize=config["fontsize_tick"])
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(labelsize=config["fontsize_tick"])
        
        # Plot 4: Phase portraits
        ax4 = axes[1, 1]
        for i, (pd_traj, tool_traj) in enumerate(zip(pd_trajectories, tool_trajectories)):
            ax4.plot(pd_traj['positions'], pd_traj['velocities'],
                    color=colors["pd"], linewidth=config["linewidth"],
                    alpha=0.7, label='PD Baseline' if i == 0 else '')
            ax4.plot(tool_traj['positions'], tool_traj['velocities'],
                    color=colors["tool"], linewidth=config["linewidth"],
                    alpha=0.7, label='Tool-Augmented' if i == 0 else '')
            
            # Mark start points
            ax4.scatter(pd_traj['positions'][0], pd_traj['velocities'][0],
                       color=colors["pd"], s=100, marker='o', alpha=0.8, zorder=5)
            ax4.scatter(tool_traj['positions'][0], tool_traj['velocities'][0],
                       color=colors["tool"], s=100, marker='s', alpha=0.8, zorder=5)
        
        # Mark target
        ax4.scatter(0, 0, color=colors["target"], s=200, marker='*', 
                   label='Target', zorder=10)
        
        ax4.set_xlabel('Position (m)', fontsize=config["fontsize_label"])
        ax4.set_ylabel('Velocity (m/s)', fontsize=config["fontsize_label"])
        ax4.set_title('Phase Portrait', fontsize=config["fontsize_label"])
        ax4.legend(fontsize=config["fontsize_tick"])
        ax4.grid(True, alpha=0.3)
        ax4.tick_params(labelsize=config["fontsize_tick"])
        
        plt.tight_layout()
        
        # Save plot with standardized naming
        plot_path = self.results_dir / "plots" / f"pd_vs_tool_comparison_{self.timestamp}.png"
        plt.savefig(plot_path, dpi=config["dpi"], bbox_inches='tight')
        print(f"💾 Saved comparison plot: {plot_path}")
        plt.close()
        
        return plot_path
    
    def save_results(self, results, scenarios):
        """Save experimental results"""
        # Prepare data for JSON serialization
        json_results = {}
        for controller_name, trajectories in results.items():
            json_results[controller_name] = []
            for traj in trajectories:
                json_traj = {}
                for key, value in traj.items():
                    if isinstance(value, (list, np.ndarray)):
                        json_traj[key] = [float(x) for x in value]
                    else:
                        json_traj[key] = float(value) if isinstance(value, np.number) else value
                json_results[controller_name].append(json_traj)
        
        # Save trajectory data with standardized naming
        data_path = self.results_dir / "data" / f"pd_vs_tool_trajectories_{self.timestamp}.json"
        with open(data_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        print(f"💾 Saved results: {data_path}")
        
        # Generate summary report with standardized naming
        report_path = self.results_dir / "reports" / f"pd_vs_tool_analysis_{self.timestamp}.md"
        with open(report_path, 'w') as f:
            f.write(f"# Control Comparison Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Scenarios:** {len(scenarios)}\n\n")
            
            f.write("## Performance Summary\n\n")
            f.write("| Controller | Success Rate | Avg Steps | Avg Control Effort |\n")
            f.write("|------------|-------------|-----------|-------------------|\n")
            
            for controller_name, trajectories in results.items():
                successes = sum(1 for t in trajectories if t['success'])
                success_rate = successes / len(trajectories) * 100
                successful_trajs = [t for t in trajectories if t['success']]
                avg_steps = np.mean([t['steps'] for t in successful_trajs]) if successful_trajs else 'N/A'
                avg_effort = np.mean([t['control_effort'] for t in successful_trajs]) if successful_trajs else 'N/A'
                
                steps_str = f"{avg_steps:.1f}" if isinstance(avg_steps, float) else str(avg_steps)
                effort_str = f"{avg_effort:.2f}" if isinstance(avg_effort, float) else str(avg_effort)
                f.write(f"| {controller_name} | {success_rate:.1f}% | {steps_str} | {effort_str} |\n")
            
            f.write(f"\n## Scenarios Tested\n\n")
            for scenario in scenarios:
                f.write(f"- **{scenario['name']}**: Start ({scenario['init_pos']:.1f}, {scenario['init_vel']:.1f}) → Target ({scenario['target_pos']:.1f}, {scenario['target_vel']:.1f})\n")
        
        print(f"📝 Saved report: {report_path}")
        
        return data_path, report_path

def main():
    """Main experiment runner"""
    parser = argparse.ArgumentParser(description="Run clean agentic control experiments")
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard", "all"], 
                       default="all", help="Scenario difficulty level")
    parser.add_argument("--save-plots", action="store_true", default=True, 
                       help="Save visualization plots")
    
    args = parser.parse_args()
    
    print("🚀 Starting Clean Agentic Control Experiments")
    print(f"📋 Difficulty: {args.difficulty}")
    print(f"📊 Save Plots: {args.save_plots}")
    print("")
    
    # Run experiments
    runner = CleanExperimentRunner()
    results, scenarios = runner.run_comparison_experiment(args.difficulty)
    
    # Create visualizations
    if args.save_plots:
        plot_path = runner.create_enhanced_comparison_plot(results, scenarios)
    
    # Save results
    data_path, report_path = runner.save_results(results, scenarios)
    
    # Print summary
    print("\n✅ Experiments Completed Successfully!")
    print("\n📊 Quick Summary:")
    for controller_name, trajectories in results.items():
        successes = sum(1 for t in trajectories if t['success'])
        success_rate = successes / len(trajectories) * 100
        print(f"   {controller_name}: {success_rate:.1f}% success ({successes}/{len(trajectories)})")
    
    print(f"\n📁 Results saved to:")
    print(f"   📊 PD vs Tool comparison plot: {plot_path.name}")
    print(f"   💾 Trajectory data: {data_path.name}")
    print(f"   📝 Analysis report: {report_path.name}")

if __name__ == "__main__":
    main()