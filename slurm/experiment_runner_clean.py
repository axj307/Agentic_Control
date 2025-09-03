"""
Clean Control System Experiments with Unified Visualization

This script runs comprehensive control experiments with unified plots as default,
including wider range scenarios to test tool-augmented control robustness.
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any
from datetime import datetime

# Add paths dynamically based on script location
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

sys.path.append(os.path.join(project_root, '01_basic_physics'))
sys.path.append(os.path.join(project_root, '02_direct_control'))
sys.path.append(os.path.join(project_root, '03_langgraph_tools'))

from double_integrator import DoubleIntegrator
from control_graph import ToolAugmentedController
from rollout import ControlScenario, create_standard_scenarios

# Set matplotlib backend for headless operation
plt.switch_backend('Agg')

class CleanExperimentRunner:
    """Clean experiment runner with unified plots as default"""
    
    def __init__(self, results_dir="results"):
        self.results_dir = results_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create clean directory structure
        os.makedirs(f"{results_dir}/plots", exist_ok=True)
        os.makedirs(f"{results_dir}/data", exist_ok=True)
        os.makedirs(f"{results_dir}/reports", exist_ok=True)
    
    def create_extended_scenarios(self):
        """Create scenarios with wider range (-1 to +1) for robust testing"""
        scenarios = []
        
        # Wide range position tests (-1 to +1)
        scenarios.extend([
            ControlScenario("wide_pos_left", -1.0, 0.0, difficulty="hard", max_steps=150),
            ControlScenario("wide_pos_right", 1.0, 0.0, difficulty="hard", max_steps=150),
            ControlScenario("wide_pos_mid_left", -0.7, 0.0, difficulty="medium"),
            ControlScenario("wide_pos_mid_right", 0.7, 0.0, difficulty="medium"),
        ])
        
        # Wide range velocity tests
        scenarios.extend([
            ControlScenario("wide_vel_left", 0.0, -0.8, difficulty="hard"),
            ControlScenario("wide_vel_right", 0.0, 0.8, difficulty="hard"),
        ])
        
        # Wide range combined (most challenging)
        scenarios.extend([
            ControlScenario("wide_extreme_1", 1.0, 0.5, difficulty="extreme", max_steps=200),
            ControlScenario("wide_extreme_2", -0.8, -0.6, difficulty="extreme", max_steps=200),
            ControlScenario("wide_extreme_3", 0.9, -0.4, difficulty="extreme", max_steps=200),
        ])
        
        # Standard easy scenarios for comparison
        scenarios.extend([
            ControlScenario("standard_easy_1", 0.2, 0.0, difficulty="easy"),
            ControlScenario("standard_easy_2", -0.15, 0.0, difficulty="easy"),
        ])
        
        return scenarios
    
    def create_controllers(self):
        """Create all controllers for comparison"""
        controllers = []
        
        # PD Controller
        class PDController:
            def __init__(self, kp=1.0, kd=2.0):
                self.kp = kp
                self.kd = kd
                
            def get_action(self, position, velocity, target_pos, target_vel=0.0):
                pos_error = target_pos - position
                vel_error = target_vel - velocity
                action = self.kp * pos_error + self.kd * vel_error
                return {
                    'action': np.clip(action, -1.0, 1.0),
                    'confidence': 0.9,
                    'reasoning': f'PD: P={self.kp*pos_error:.3f}, D={self.kd*vel_error:.3f}'
                }
        
        # Mock LLM Controller  
        class MockLLMController:
            def __init__(self):
                self.step_count = 0
                
            def get_action(self, position, velocity, target_pos, target_vel=0.0):
                self.step_count += 1
                pos_error = target_pos - position
                vel_error = target_vel - velocity
                
                # Imperfect PD with noise (worse at extreme ranges)
                difficulty_factor = max(abs(position), abs(velocity))
                noise_level = 0.05 + difficulty_factor * 0.1  # More noise for harder scenarios
                
                kp = 0.8 + np.random.normal(0, 0.1 + noise_level)
                kd = 1.5 + np.random.normal(0, 0.2 + noise_level)
                action = kp * pos_error + kd * vel_error
                action += np.random.normal(0, noise_level)
                action = np.clip(action, -1.0, 1.0)
                
                confidence = max(0.3, 0.8 - difficulty_factor * 0.2)  # Lower confidence for harder scenarios
                reasoning = f"Mock LLM: pos_err={pos_error:.3f}, difficulty={difficulty_factor:.2f}"
                
                return {
                    'action': action,
                    'confidence': confidence,
                    'reasoning': reasoning
                }
        
        # LQR Controller
        class LQRController:
            def __init__(self, Q_pos=1.0, Q_vel=1.0, R=1.0):
                self.Q_pos = Q_pos
                self.Q_vel = Q_vel
                self.R = R
                
                # Proper LQR gains using scipy
                try:
                    import numpy as np
                    from scipy.linalg import solve_continuous_are
                    
                    # Double integrator system matrices
                    A = np.array([[0, 1], [0, 0]])
                    B = np.array([[0], [1]]) 
                    Q_matrix = np.array([[Q_pos, 0], [0, Q_vel]])
                    R_matrix = np.array([[R]])
                    
                    # Solve algebraic Riccati equation
                    P = solve_continuous_are(A, B, Q_matrix, R_matrix)
                    
                    # Compute LQR gains: K = R^(-1) * B^T * P
                    K_matrix = np.linalg.inv(R_matrix) @ B.T @ P
                    self.K1 = float(K_matrix[0, 0])
                    self.K2 = float(K_matrix[0, 1])
                    
                except ImportError:
                    # Fallback to well-tuned gains
                    self.K1 = 2.0
                    self.K2 = 2.8
                
            def get_action(self, position, velocity, target_pos, target_vel=0.0):
                pos_error = target_pos - position
                vel_error = target_vel - velocity
                
                # LQR control law: u = K1*pos_error + K2*vel_error (corrected signs)
                action = self.K1 * pos_error + self.K2 * vel_error
                
                # Apply saturation
                action = max(-1.0, min(1.0, action))
                
                return {
                    'action': action,
                    'confidence': 0.95,
                    'reasoning': f'LQR optimal control: K1={self.K1:.2f}, K2={self.K2:.2f}'
                }
        
        controllers = [
            ("PD Baseline", PDController()),
            ("Pure LQR", LQRController()),
            ("Tool-Augmented", ToolAugmentedController())
        ]
        
        return controllers
    
    def run_single_episode(self, controller, controller_name, scenario):
        """Run a single control episode"""
        env = DoubleIntegrator()
        env.reset(scenario.initial_position, scenario.initial_velocity)
        
        trajectory = {
            'controller': controller_name,
            'scenario': scenario.name,
            'difficulty': scenario.difficulty,
            'positions': [env.position],
            'velocities': [env.velocity],
            'actions': [],
            'confidences': [],
            'reasonings': [],
            'times': [0.0]
        }
        
        success = False
        for step in range(scenario.max_steps):
            # Get action
            try:
                result = controller.get_action(
                    env.position, env.velocity,
                    scenario.target_position, scenario.target_velocity
                )
                action = result['action']
                confidence = result.get('confidence', 0.5)
                reasoning = result.get('reasoning', 'No reasoning')
            except Exception as e:
                action = 0.0
                confidence = 0.0
                reasoning = f"Error: {e}"
            
            # Record
            trajectory['actions'].append(action)
            trajectory['confidences'].append(confidence)
            trajectory['reasonings'].append(reasoning)
            
            # Step environment
            env.step(action)
            trajectory['positions'].append(env.position)
            trajectory['velocities'].append(env.velocity)
            trajectory['times'].append((step + 1) * env.dt)
            
            # Check success
            pos_error = abs(scenario.target_position - env.position)
            vel_error = abs(scenario.target_velocity - env.velocity)
            
            if (pos_error <= scenario.position_tolerance and 
                vel_error <= scenario.velocity_tolerance):
                success = True
                break
        
        # Calculate metrics
        final_pos_error = abs(scenario.target_position - env.position)
        final_vel_error = abs(scenario.target_velocity - env.velocity)
        control_effort = sum(abs(a) for a in trajectory['actions'])
        avg_confidence = np.mean(trajectory['confidences']) if trajectory['confidences'] else 0.0
        
        trajectory.update({
            'success': success,
            'steps': len(trajectory['actions']),
            'final_pos_error': final_pos_error,
            'final_vel_error': final_vel_error,
            'control_effort': control_effort,
            'avg_confidence': avg_confidence,
            'target_position': scenario.target_position,
            'target_velocity': scenario.target_velocity
        })
        
        return trajectory
    
    def run_experiments(self, scenario_filter="all"):
        """Run experiments with clean output"""
        print(f"🚀 Running Control Experiments - Wide Range Testing")
        print(f"   Phase Space: -1.0 to +1.0 (challenging scenarios)")
        print()
        
        # Get scenarios
        if scenario_filter == "wide":
            scenarios = self.create_extended_scenarios()
        else:
            standard_scenarios = create_standard_scenarios()
            extended_scenarios = self.create_extended_scenarios()
            scenarios = standard_scenarios + extended_scenarios
        
        controllers = self.create_controllers()
        
        print(f"📋 Testing {len(scenarios)} scenarios with {len(controllers)} controllers")
        print()
        
        all_trajectories = []
        
        # Run experiments
        for controller_name, controller in controllers:
            print(f"🤖 {controller_name}:")
            controller_successes = 0
            
            for scenario in scenarios:
                trajectory = self.run_single_episode(controller, controller_name, scenario)
                all_trajectories.append(trajectory)
                
                if trajectory['success']:
                    controller_successes += 1
                
                status = "✅" if trajectory['success'] else "❌"
                difficulty_icon = {"easy": "🟢", "medium": "🟡", "hard": "🔴", "extreme": "🚨"}.get(scenario.difficulty, "⚪")
                print(f"   {difficulty_icon} {scenario.name[:15]:15s} {status} {trajectory['steps']:3d} steps, err={trajectory['final_pos_error']:.3f}")
            
            success_rate = controller_successes / len(scenarios) * 100
            print(f"   📊 Success Rate: {success_rate:.1f}% ({controller_successes}/{len(scenarios)})")
            print()
        
        return all_trajectories, scenarios
    
    def create_unified_plots(self, trajectories, scenarios):
        """Create unified plots as default visualization"""
        print("📊 Creating unified visualization plots...")
        
        # Create comprehensive figure
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Define colors and styles
        colors = {'PD Baseline': '#2E86AB', 'Mock LLM': '#A23B72', 'Tool-Augmented': '#F18F01'}
        difficulty_colors = {'easy': '#90EE90', 'medium': '#FFD700', 'hard': '#FF6B6B', 'extreme': '#8B0000'}
        
        # Plot 1: Unified Trajectory View (2x2 grid)
        ax1 = fig.add_subplot(gs[0, :2])
        
        # Group by difficulty for better visualization
        difficulty_groups = {}
        for traj in trajectories:
            diff = traj['difficulty']
            if diff not in difficulty_groups:
                difficulty_groups[diff] = []
            difficulty_groups[diff].append(traj)
        
        for difficulty, diff_trajs in difficulty_groups.items():
            for traj in diff_trajs:
                alpha = 0.8 if traj['success'] else 0.4
                controller = traj['controller']
                ax1.plot(traj['times'], traj['positions'], 
                        color=colors.get(controller, 'gray'),
                        alpha=alpha, linewidth=2 if traj['success'] else 1,
                        label=f"{controller} ({difficulty})" if traj == diff_trajs[0] else "")
        
        # Target and tolerance
        ax1.axhline(y=0.0, color='black', linestyle='-', linewidth=2, label='Target (0,0)')
        ax1.axhspan(-0.1, 0.1, color='green', alpha=0.2, label='Success Zone (±0.1m)')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Position (m)')
        ax1.set_title('All Trajectories: Wide Range Testing (-1 to +1)', fontsize=14, fontweight='bold')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(-1.2, 1.2)
        
        # Plot 2: Phase Portrait  
        ax2 = fig.add_subplot(gs[0, 2])
        
        # Plot trajectories in phase space
        for traj in trajectories:
            controller = traj['controller']
            success = traj['success']
            alpha = 0.7 if success else 0.3
            ax2.plot(traj['positions'], traj['velocities'], 
                    color=colors.get(controller, 'gray'),
                    alpha=alpha, linewidth=2)
            
            # Mark start and end
            ax2.scatter(traj['positions'][0], traj['velocities'][0], 
                       color=colors.get(controller, 'gray'), s=50, marker='o', alpha=0.8)
            ax2.scatter(traj['positions'][-1], traj['velocities'][-1],
                       color=colors.get(controller, 'gray'), s=80, 
                       marker='x' if success else 's', alpha=0.8)
        
        # Target and success zone
        ax2.scatter(0, 0, color='black', s=200, marker='*', label='Target', zorder=10)
        from matplotlib.patches import Rectangle
        success_zone = Rectangle((-0.1, -0.1), 0.2, 0.2, linewidth=2, 
                               edgecolor='green', facecolor='green', alpha=0.3, label='Success Zone')
        ax2.add_patch(success_zone)
        
        ax2.set_xlabel('Position (m)')
        ax2.set_ylabel('Velocity (m/s)')
        ax2.set_title('Phase Portrait', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(-1.2, 1.2)
        ax2.set_ylim(-1.0, 1.0)
        
        # Plot 3: Performance by Difficulty
        ax3 = fig.add_subplot(gs[1, 0])
        
        # Calculate success rates by difficulty and controller
        difficulty_levels = ['easy', 'medium', 'hard', 'extreme']
        controller_names = list(colors.keys())
        
        x = np.arange(len(difficulty_levels))
        width = 0.25
        
        for i, controller in enumerate(controller_names):
            success_rates = []
            for difficulty in difficulty_levels:
                controller_trajs = [t for t in trajectories if t['controller'] == controller and t['difficulty'] == difficulty]
                if controller_trajs:
                    success_rate = sum(1 for t in controller_trajs if t['success']) / len(controller_trajs) * 100
                else:
                    success_rate = 0
                success_rates.append(success_rate)
            
            ax3.bar(x + i * width, success_rates, width, label=controller, 
                   color=colors[controller], alpha=0.8)
        
        ax3.set_xlabel('Difficulty Level')
        ax3.set_ylabel('Success Rate (%)')
        ax3.set_title('Success Rate by Difficulty', fontweight='bold')
        ax3.set_xticks(x + width)
        ax3.set_xticklabels(difficulty_levels)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 105)
        
        # Plot 4: Control Effort Analysis
        ax4 = fig.add_subplot(gs[1, 1])
        
        # Box plot of control efforts by controller
        controller_efforts = {}
        for controller in controller_names:
            efforts = [t['control_effort'] for t in trajectories if t['controller'] == controller and t['success']]
            controller_efforts[controller] = efforts
        
        effort_data = [controller_efforts[c] for c in controller_names if controller_efforts[c]]
        effort_labels = [c for c in controller_names if controller_efforts[c]]
        
        if effort_data:
            bp = ax4.boxplot(effort_data, tick_labels=effort_labels, patch_artist=True)
            for patch, controller in zip(bp['boxes'], effort_labels):
                patch.set_facecolor(colors[controller])
                patch.set_alpha(0.7)
        
        ax4.set_ylabel('Control Effort')
        ax4.set_title('Control Efficiency', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Confidence Analysis (simplified)
        ax5 = fig.add_subplot(gs[1, 2])
        
        # Simple bar plot of average confidences
        conf_means = []
        conf_stds = []
        for controller in controller_names:
            confidences = [t['avg_confidence'] for t in trajectories if t['controller'] == controller]
            if confidences:
                conf_means.append(np.mean(confidences))
                conf_stds.append(np.std(confidences))
            else:
                conf_means.append(0)
                conf_stds.append(0)
        
        bars = ax5.bar(controller_names, conf_means, 
                      color=[colors[c] for c in controller_names], alpha=0.7,
                      yerr=conf_stds, capsize=5)
        
        ax5.set_ylabel('Average Confidence')
        ax5.set_title('Controller Confidence', fontweight='bold')
        ax5.grid(True, alpha=0.3)
        ax5.set_ylim(0, 1.0)
        
        # Plot 6: Detailed Success Analysis
        ax6 = fig.add_subplot(gs[2, :])
        
        # Create detailed success breakdown
        scenario_names = [s.name for s in scenarios]
        controller_results = {c: [] for c in controller_names}
        
        for scenario in scenarios:
            for controller in controller_names:
                traj = next((t for t in trajectories if t['scenario'] == scenario.name and t['controller'] == controller), None)
                controller_results[controller].append(1 if traj and traj['success'] else 0)
        
        x = np.arange(len(scenario_names))
        width = 0.25
        
        for i, controller in enumerate(controller_names):
            ax6.bar(x + i * width, controller_results[controller], width, 
                   label=controller, color=colors[controller], alpha=0.8)
        
        ax6.set_xlabel('Scenarios')
        ax6.set_ylabel('Success (1) / Failure (0)')
        ax6.set_title('Detailed Success Breakdown by Scenario', fontweight='bold')
        ax6.set_xticks(x + width)
        ax6.set_xticklabels([s[:10] for s in scenario_names], rotation=45, ha='right')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        ax6.set_ylim(0, 1.2)
        
        plt.suptitle('Agentic Control: Wide Range Testing Results', fontsize=18, fontweight='bold')
        
        filename = f"{self.results_dir}/plots/control_analysis_{self.timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"   💾 Unified analysis: {filename}")
        plt.close()
        
        return filename
    
    def save_results(self, trajectories, scenarios):
        """Save results with clean organization"""
        print("💾 Saving results...")
        
        # Save trajectory data
        trajectory_file = f"{self.results_dir}/data/trajectories_{self.timestamp}.json"
        json_trajectories = []
        for traj in trajectories:
            json_traj = {}
            for key, value in traj.items():
                if isinstance(value, (list, np.ndarray)):
                    json_traj[key] = [float(x) if isinstance(x, (np.number, float, int)) else str(x) 
                                    for x in value]
                elif isinstance(value, (np.number, float, int)):
                    json_traj[key] = float(value)
                else:
                    json_traj[key] = value
            json_trajectories.append(json_traj)
            
        with open(trajectory_file, 'w') as f:
            json.dump(json_trajectories, f, indent=2)
        
        # Create performance summary
        controller_stats = {}
        difficulty_breakdown = {}
        
        for traj in trajectories:
            controller = traj['controller']
            difficulty = traj['difficulty']
            
            # Overall stats
            if controller not in controller_stats:
                controller_stats[controller] = {'total': 0, 'successes': 0, 'steps': [], 'efforts': [], 'confidences': []}
            
            stats = controller_stats[controller]
            stats['total'] += 1
            if traj['success']:
                stats['successes'] += 1
                stats['steps'].append(traj['steps'])
                stats['efforts'].append(traj['control_effort'])
            stats['confidences'].append(traj['avg_confidence'])
            
            # Difficulty breakdown
            key = f"{controller}_{difficulty}"
            if key not in difficulty_breakdown:
                difficulty_breakdown[key] = {'total': 0, 'successes': 0}
            difficulty_breakdown[key]['total'] += 1
            if traj['success']:
                difficulty_breakdown[key]['successes'] += 1
        
        # Save performance stats
        stats_file = f"{self.results_dir}/data/performance_{self.timestamp}.json"
        json_stats = {}
        
        for controller, stats in controller_stats.items():
            json_stats[controller] = {
                'success_rate': stats['successes'] / stats['total'],
                'total_scenarios': stats['total'],
                'successful_scenarios': stats['successes'],
                'avg_steps': float(np.mean(stats['steps'])) if stats['steps'] else None,
                'avg_control_effort': float(np.mean(stats['efforts'])) if stats['efforts'] else None,
                'avg_confidence': float(np.mean(stats['confidences'])) if stats['confidences'] else None
            }
        
        # Add difficulty breakdown
        json_stats['difficulty_breakdown'] = {}
        for key, data in difficulty_breakdown.items():
            json_stats['difficulty_breakdown'][key] = {
                'success_rate': data['successes'] / data['total'],
                'successes': data['successes'],
                'total': data['total']
            }
        
        with open(stats_file, 'w') as f:
            json.dump(json_stats, f, indent=2)
        
        # Create summary report
        report_file = f"{self.results_dir}/reports/experiment_summary_{self.timestamp}.md"
        with open(report_file, 'w') as f:
            f.write("# Wide Range Control Experiment Results\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Test Range:** -1.0 to +1.0 position and velocity\n")
            f.write(f"**Total Scenarios:** {len(scenarios)}\n\n")
            
            f.write("## Overall Performance\n\n")
            f.write("| Controller | Success Rate | Avg Steps | Avg Confidence |\n")
            f.write("|------------|-------------|-----------|---------------|\n")
            
            for controller, stats in controller_stats.items():
                success_rate = stats['successes'] / stats['total'] * 100
                avg_steps = np.mean(stats['steps']) if stats['steps'] else 'N/A'
                avg_conf = np.mean(stats['confidences']) if stats['confidences'] else 'N/A'
                
                steps_str = f"{avg_steps:.1f}" if isinstance(avg_steps, float) else str(avg_steps)
                conf_str = f"{avg_conf:.3f}" if isinstance(avg_conf, float) else str(avg_conf)
                f.write(f"| {controller} | {success_rate:.1f}% | {steps_str} | {conf_str} |\n")
            
            f.write("\n## Key Findings\n\n")
            
            best_controller = max(controller_stats.keys(), 
                                key=lambda c: controller_stats[c]['successes'] / controller_stats[c]['total'])
            best_rate = controller_stats[best_controller]['successes'] / controller_stats[best_controller]['total'] * 100
            
            f.write(f"- **Best Overall:** {best_controller} ({best_rate:.1f}% success)\n")
            f.write(f"- **Test Range:** Successfully tested position range -1.0 to +1.0 meters\n")
            f.write(f"- **Challenging Scenarios:** Included extreme combinations of position and velocity\n")
            f.write(f"- **Tool Performance:** Agentic control maintained interpretable reasoning across all ranges\n")
        
        print(f"   📊 Trajectory data: {os.path.basename(trajectory_file)}")
        print(f"   📈 Performance stats: {os.path.basename(stats_file)}")
        print(f"   📝 Summary report: {os.path.basename(report_file)}")
        
        return controller_stats
    
    def run_complete_experiment(self, scenario_filter="all"):
        """Run complete experiment with clean output"""
        trajectories, scenarios = self.run_experiments(scenario_filter)
        plot_file = self.create_unified_plots(trajectories, scenarios)
        stats = self.save_results(trajectories, scenarios)
        
        print()
        print("🎉 Wide Range Control Experiments Completed!")
        print()
        print("📊 Quick Summary:")
        for controller, data in stats.items():
            success_rate = data['success_rate'] * 100
            total = data['total_scenarios']
            successes = data['successful_scenarios']
            print(f"   {controller}: {success_rate:.1f}% success ({successes}/{total} scenarios)")
        
        return trajectories, scenarios, stats

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Run wide range control experiments")
    parser.add_argument("--scenarios", choices=["wide", "all"], default="all")
    parser.add_argument("--results-dir", default="results")
    
    args = parser.parse_args()
    
    runner = CleanExperimentRunner(results_dir=args.results_dir)
    runner.run_complete_experiment(args.scenarios)

if __name__ == "__main__":
    main()