"""
Test Tool-Augmented Control System

This script comprehensively tests the tool-augmented control approach
and compares it with direct control methods.

Usage:
    python test_graph.py --scenarios easy
    python test_graph.py --scenarios all --verbose --save results.json
"""

import argparse
import asyncio
import json
import time
from typing import Dict, List, Optional
import numpy as np

# Add parent directories to path
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '01_basic_physics'))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '02_direct_control'))
from double_integrator import DoubleIntegrator as DoubleIntegratorEnvironment
from control_graph import ToolAugmentedController
from rollout import ControlScenario, create_standard_scenarios, control_rollout


class PDController:
    """Simple PD controller for baseline comparison"""
    
    def __init__(self, kp: float = 1.0, kd: float = 2.0):
        self.kp = kp
        self.kd = kd
        
    def get_action(self, position: float, velocity: float,
                   target_pos: float, target_vel: float = 0.0) -> float:
        """Get PD control action"""
        pos_error = target_pos - position
        vel_error = target_vel - velocity
        
        action = self.kp * pos_error + self.kd * vel_error
        return np.clip(action, -1.0, 1.0)


class MockLLMController:
    """Mock LLM controller for comparison (from simple_controller.py)"""
    
    def __init__(self):
        self.step_count = 0
        
    def get_action(self, position: float, velocity: float, 
                   target_pos: float, target_vel: float = 0.0) -> Dict:
        self.step_count += 1
        
        # Simple PD-like control with some noise and imperfection
        pos_error = target_pos - position
        vel_error = target_vel - velocity
        
        # Imperfect gains and some randomness
        kp = 0.8 + np.random.normal(0, 0.1)  # Noisy gain
        kd = 1.5 + np.random.normal(0, 0.2)  # Noisy gain
        
        action = kp * pos_error + kd * vel_error
        action += np.random.normal(0, 0.05)  # Control noise
        action = np.clip(action, -1.0, 1.0)
        
        confidence = 0.6 + 0.3 * np.random.random()
        
        reasoning = f"Position error is {pos_error:.3f}, velocity is {velocity:.3f}. "
        if abs(pos_error) > abs(vel_error):
            reasoning += "Large position error dominates. Focus on positioning."
        elif abs(vel_error) > 0.1:
            reasoning += "Velocity control needed."
        else:
            reasoning += "System is close to target. Applying gentle control."
        
        return {
            'action': action,
            'confidence': confidence,
            'reasoning': reasoning,
            'step': self.step_count
        }


def test_single_scenario(controller, controller_name: str, scenario: ControlScenario, 
                        verbose: bool = False) -> Dict:
    """Test a single controller on one scenario"""
    
    if verbose:
        print(f"Testing {controller_name} on {scenario.name}")
    
    env = DoubleIntegratorEnvironment()
    env.reset(scenario.initial_position, scenario.initial_velocity)
    
    steps = 0
    control_effort = 0.0
    success = False
    trajectory = []
    actions = []
    confidences = []
    reasonings = []
    
    start_time = time.time()
    
    for step in range(scenario.max_steps):
        # Get action from controller
        try:
            result = controller.get_action(
                env.position, env.velocity,
                scenario.target_position, scenario.target_velocity
            )
            action = result['action']
            confidence = result.get('confidence', 0.5)
            reasoning = result.get('reasoning', 'No reasoning provided')
            
        except Exception as e:
            if verbose:
                print(f"   Error at step {step}: {e}")
            action = 0.0
            confidence = 0.0
            reasoning = f"Error: {e}"
        
        # Record data
        trajectory.append((env.position, env.velocity))
        actions.append(action)
        confidences.append(confidence)
        reasonings.append(reasoning)
        
        # Execute action
        env.step(action)
        control_effort += abs(action)
        steps += 1
        
        # Check success
        pos_error = abs(scenario.target_position - env.position)
        vel_error = abs(scenario.target_velocity - env.velocity)
        
        if (pos_error <= scenario.position_tolerance and 
            vel_error <= scenario.velocity_tolerance):
            success = True
            break
    
    end_time = time.time()
    computation_time = end_time - start_time
    
    # Calculate final errors
    final_pos_error = abs(scenario.target_position - env.position)
    final_vel_error = abs(scenario.target_velocity - env.velocity)
    
    result = {
        'controller': controller_name,
        'scenario': scenario.name,
        'success': success,
        'steps': steps,
        'final_pos_error': final_pos_error,
        'final_vel_error': final_vel_error,
        'control_effort': control_effort,
        'computation_time': computation_time,
        'avg_confidence': np.mean(confidences) if confidences else 0.0,
        'trajectory': trajectory,
        'actions': actions,
        'confidences': confidences,
        'reasonings': reasonings[-3:] if reasonings else []  # Last 3 reasonings
    }
    
    if verbose:
        status = "✅" if success else "❌"
        print(f"   {status} Steps: {steps}, Pos err: {final_pos_error:.3f}, Time: {computation_time:.2f}s")
    
    return result


def test_controller_suite(controller, controller_name: str, scenarios: List[ControlScenario],
                         verbose: bool = False) -> Dict:
    """Test a controller on a suite of scenarios"""
    
    if verbose:
        print(f"\n🤖 Testing {controller_name}")
        print("-" * 40)
    
    results = []
    start_time = time.time()
    
    for scenario in scenarios:
        result = test_single_scenario(controller, controller_name, scenario, verbose)
        results.append(result)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Calculate summary statistics
    successful_results = [r for r in results if r['success']]
    
    if successful_results:
        success_rate = len(successful_results) / len(results)
        avg_steps = np.mean([r['steps'] for r in successful_results])
        avg_control_effort = np.mean([r['control_effort'] for r in successful_results])
        avg_computation_time = np.mean([r['computation_time'] for r in successful_results])
        avg_confidence = np.mean([r['avg_confidence'] for r in successful_results])
    else:
        success_rate = 0.0
        avg_steps = float('inf')
        avg_control_effort = float('inf')
        avg_computation_time = float('inf')
        avg_confidence = 0.0
    
    summary = {
        'controller': controller_name,
        'total_scenarios': len(scenarios),
        'success_rate': success_rate,
        'avg_steps': avg_steps,
        'avg_control_effort': avg_control_effort,
        'avg_computation_time': avg_computation_time,
        'avg_confidence': avg_confidence,
        'total_time': total_time,
        'results': results
    }
    
    if verbose:
        print(f"📊 {controller_name} Summary:")
        print(f"   Success Rate: {success_rate*100:.1f}%")
        print(f"   Avg Steps: {avg_steps:.1f}")
        print(f"   Avg Control Effort: {avg_control_effort:.2f}")
        print(f"   Avg Computation Time: {avg_computation_time:.3f}s")
        print(f"   Avg Confidence: {avg_confidence:.3f}")
    
    return summary


def compare_controllers(summaries: List[Dict], verbose: bool = True):
    """Compare multiple controllers"""
    
    if not verbose:
        return
    
    print("\n🏆 CONTROLLER COMPARISON")
    print("=" * 80)
    
    # Sort by success rate, then by steps
    sorted_summaries = sorted(summaries, key=lambda x: (-x['success_rate'], x['avg_steps']))
    
    # Header
    print(f"{'Controller':<20} {'Success %':<10} {'Avg Steps':<10} {'Effort':<8} {'Time (s)':<8} {'Confidence':<10}")
    print("-" * 80)
    
    for summary in sorted_summaries:
        name = summary['controller'][:19]
        success_pct = f"{summary['success_rate']*100:.1f}%"
        steps = f"{summary['avg_steps']:.1f}" if summary['avg_steps'] != float('inf') else "∞"
        effort = f"{summary['avg_control_effort']:.2f}" if summary['avg_control_effort'] != float('inf') else "∞"
        comp_time = f"{summary['avg_computation_time']:.3f}" if summary['avg_computation_time'] != float('inf') else "∞"
        confidence = f"{summary['avg_confidence']:.3f}"
        
        print(f"{name:<20} {success_pct:<10} {steps:<10} {effort:<8} {comp_time:<8} {confidence:<10}")
    
    # Analysis
    if len(sorted_summaries) >= 2:
        best = sorted_summaries[0]
        print(f"\n🥇 Best Overall: {best['controller']}")
        print(f"   Success Rate: {best['success_rate']*100:.1f}%")
        
        # Find best in each category
        best_efficiency = min(summaries, key=lambda x: x['avg_steps'] if x['avg_steps'] != float('inf') else float('inf'))
        best_speed = min(summaries, key=lambda x: x['avg_computation_time'] if x['avg_computation_time'] != float('inf') else float('inf'))
        best_confidence = max(summaries, key=lambda x: x['avg_confidence'])
        
        print(f"\n📈 Category Leaders:")
        print(f"   Most Efficient: {best_efficiency['controller']} ({best_efficiency['avg_steps']:.1f} steps)")
        print(f"   Fastest: {best_speed['controller']} ({best_speed['avg_computation_time']:.3f}s)")
        print(f"   Most Confident: {best_confidence['controller']} ({best_confidence['avg_confidence']:.3f})")


def analyze_tool_usage(tool_augmented_summary: Dict, verbose: bool = True):
    """Analyze how tools were used in the tool-augmented approach"""
    
    if not verbose:
        return
    
    print("\n🔧 TOOL USAGE ANALYSIS")
    print("=" * 50)
    
    results = tool_augmented_summary['results']
    
    # Count tool usage by scenario difficulty
    easy_results = [r for r in results if 'easy' in r['scenario']]
    medium_results = [r for r in results if 'med' in r['scenario']]
    hard_results = [r for r in results if 'hard' in r['scenario']]
    
    categories = [
        ("Easy", easy_results),
        ("Medium", medium_results), 
        ("Hard", hard_results)
    ]
    
    for category_name, category_results in categories:
        if not category_results:
            continue
            
        print(f"\n{category_name} Scenarios:")
        
        # Analyze confidence by category
        confidences = [r['avg_confidence'] for r in category_results]
        success_rate = sum(1 for r in category_results if r['success']) / len(category_results)
        
        print(f"   Success Rate: {success_rate*100:.1f}%")
        print(f"   Avg Confidence: {np.mean(confidences):.3f}")
        print(f"   Confidence Range: {np.min(confidences):.3f} - {np.max(confidences):.3f}")
        
        # Show some example reasonings
        sample_reasonings = []
        for result in category_results[:2]:  # First 2 scenarios
            if result['reasonings']:
                sample_reasonings.extend(result['reasonings'][-1:])  # Last reasoning
        
        if sample_reasonings:
            print(f"   Example Reasoning: {sample_reasonings[0][:100]}...")


def main():
    parser = argparse.ArgumentParser(description="Test Tool-Augmented Control")
    parser.add_argument("--scenarios", 
                       choices=["easy", "medium", "hard", "all"],
                       default="easy",
                       help="Which scenarios to test")
    parser.add_argument("--controllers",
                       choices=["all", "tool-only", "baseline-only"],
                       default="all", 
                       help="Which controllers to test")
    parser.add_argument("--verbose", "-v",
                       action="store_true",
                       help="Verbose output")
    parser.add_argument("--save",
                       type=str,
                       help="Save results to JSON file")
    
    args = parser.parse_args()
    
    print("🚀 Tool-Augmented Control Testing")
    print("=" * 50)
    
    # Create test scenarios
    all_scenarios = create_standard_scenarios()
    if args.scenarios == "easy":
        scenarios = [s for s in all_scenarios if s.difficulty == "easy"]
    elif args.scenarios == "medium":
        scenarios = [s for s in all_scenarios if s.difficulty == "medium"]
    elif args.scenarios == "hard":
        scenarios = [s for s in all_scenarios if s.difficulty == "hard"]
    else:  # all
        scenarios = all_scenarios
    
    print(f"📋 Testing {len(scenarios)} scenarios ({args.scenarios} difficulty)")
    
    # Initialize controllers
    controllers = []
    
    if args.controllers in ["all", "baseline-only"]:
        # PD Baseline
        pd_controller = PDController(kp=1.0, kd=2.0)
        controllers.append(("PD Baseline", pd_controller))
        
        # Mock LLM
        mock_llm = MockLLMController()
        controllers.append(("Mock LLM", mock_llm))
    
    if args.controllers in ["all", "tool-only"]:
        # Tool-Augmented Controller
        tool_controller = ToolAugmentedController()
        controllers.append(("Tool-Augmented", tool_controller))
    
    print(f"🤖 Testing {len(controllers)} controllers")
    
    # Test each controller
    summaries = []
    for controller_name, controller in controllers:
        summary = test_controller_suite(controller, controller_name, scenarios, args.verbose)
        summaries.append(summary)
    
    # Compare results
    compare_controllers(summaries, verbose=True)
    
    # Analyze tool usage if tool-augmented controller was tested
    tool_summary = next((s for s in summaries if s['controller'] == "Tool-Augmented"), None)
    if tool_summary:
        analyze_tool_usage(tool_summary, verbose=args.verbose)
    
    # Save results if requested
    if args.save:
        save_data = {
            'summaries': summaries,
            'scenarios': [s.to_dict() for s in scenarios],
            'test_config': {
                'scenarios': args.scenarios,
                'controllers': args.controllers,
                'timestamp': time.time()
            }
        }
        
        with open(args.save, 'w') as f:
            json.dump(save_data, f, indent=2, default=str)
        print(f"\n💾 Results saved to {args.save}")
    
    print("\n🎉 Testing completed!")
    
    # Summary insights
    print("\n💡 Key Insights:")
    if len(summaries) > 1:
        best_controller = max(summaries, key=lambda x: x['success_rate'])
        print(f"   • Best performer: {best_controller['controller']} ({best_controller['success_rate']*100:.1f}% success)")
        
        if tool_summary:
            baseline_summary = next((s for s in summaries if "PD" in s['controller']), None)
            if baseline_summary:
                improvement = (tool_summary['success_rate'] - baseline_summary['success_rate']) * 100
                if improvement > 0:
                    print(f"   • Tool-augmented shows {improvement:.1f}% improvement over baseline")
                else:
                    print(f"   • Baseline outperforms tool-augmented by {-improvement:.1f}%")


if __name__ == "__main__":
    main()