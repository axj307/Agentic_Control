"""
Test Real LLM Controllers

This script tests real LLM controllers (vLLM, OpenAI) on control tasks
and compares their performance with the PD baseline.

Usage:
    # Test with vLLM (requires running server)
    python test_llm.py --controller vllm
    
    # Test with OpenAI (requires API key)
    python test_llm.py --controller openai
    
    # Test both if available
    python test_llm.py --controller both
"""

import argparse
import asyncio
import json
import time
from typing import Dict, List, Optional
import numpy as np

# Add parent directory to path
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from double_integrator_environment.double_integrator import DoubleIntegratorEnvironment, PDController
from llm_integration import VLLMController, OpenAIController, REQUESTS_AVAILABLE, OPENAI_AVAILABLE
from rollout import ControlScenario, create_standard_scenarios, control_rollout


def test_pd_baseline(scenarios: List[ControlScenario], verbose: bool = False) -> Dict:
    """Test PD controller baseline for comparison"""
    if verbose:
        print("🎮 Testing PD Controller Baseline...")
        print("-" * 40)
    
    pd_controller = PDController(kp=1.0, kd=2.0)
    results = []
    
    for scenario in scenarios:
        env = DoubleIntegratorEnvironment(
            position=scenario.initial_position,
            velocity=scenario.initial_velocity
        )
        
        steps = 0
        control_effort = 0.0
        success = False
        
        for step in range(scenario.max_steps):
            # Get PD action
            action = pd_controller.get_action(
                env.position, env.velocity,
                scenario.target_position, scenario.target_velocity
            )
            
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
        
        result = {
            'scenario': scenario.name,
            'success': success,
            'steps': steps,
            'final_pos_error': abs(scenario.target_position - env.position),
            'final_vel_error': abs(scenario.target_velocity - env.velocity),
            'control_effort': control_effort
        }
        results.append(result)
        
        if verbose:
            status = "✅" if success else "❌"
            print(f"{status} {scenario.name}: steps={steps}, pos_err={result['final_pos_error']:.3f}")
    
    # Calculate summary stats
    success_rate = sum(1 for r in results if r['success']) / len(results)
    avg_steps = np.mean([r['steps'] for r in results if r['success']])
    avg_control_effort = np.mean([r['control_effort'] for r in results if r['success']])
    
    summary = {
        'controller': 'PD Baseline',
        'success_rate': success_rate,
        'avg_steps': avg_steps,
        'avg_control_effort': avg_control_effort,
        'results': results
    }
    
    if verbose:
        print(f"📊 PD Summary: {success_rate*100:.1f}% success, {avg_steps:.1f} avg steps")
    
    return summary


async def test_llm_controller(controller, controller_name: str, 
                            scenarios: List[ControlScenario],
                            verbose: bool = False) -> Dict:
    """Test an LLM controller on scenarios"""
    if verbose:
        print(f"🤖 Testing {controller_name} Controller...")
        print("-" * 40)
    
    results = []
    start_time = time.time()
    
    for scenario_idx, scenario in enumerate(scenarios):
        if verbose:
            print(f"Scenario {scenario_idx + 1}/{len(scenarios)}: {scenario.name}")
        
        try:
            trajectory = await control_rollout(
                controller=controller,
                scenario=scenario,
                verbose=False
            )
            
            result = {
                'scenario': scenario.name,
                'success': trajectory.metadata['success'],
                'steps': trajectory.metadata['steps'],
                'final_pos_error': trajectory.metadata['final_position_error'],
                'final_vel_error': trajectory.metadata['final_velocity_error'],
                'control_effort': trajectory.metadata['total_control_effort'],
                'reward': trajectory.reward
            }
            
        except Exception as e:
            if verbose:
                print(f"❌ Error in {scenario.name}: {e}")
            result = {
                'scenario': scenario.name,
                'success': False,
                'steps': scenario.max_steps,
                'final_pos_error': float('inf'),
                'final_vel_error': float('inf'),
                'control_effort': float('inf'),
                'reward': -1000.0,
                'error': str(e)
            }
        
        results.append(result)
        
        if verbose:
            status = "✅" if result['success'] else "❌"
            print(f"  {status} steps={result['steps']}, pos_err={result['final_pos_error']:.3f}")
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Calculate summary stats (only successful runs)
    successful_results = [r for r in results if r['success']]
    
    if successful_results:
        success_rate = len(successful_results) / len(results)
        avg_steps = np.mean([r['steps'] for r in successful_results])
        avg_control_effort = np.mean([r['control_effort'] for r in successful_results])
        avg_reward = np.mean([r['reward'] for r in successful_results])
    else:
        success_rate = 0.0
        avg_steps = float('inf')
        avg_control_effort = float('inf')
        avg_reward = -1000.0
    
    summary = {
        'controller': controller_name,
        'success_rate': success_rate,
        'avg_steps': avg_steps,
        'avg_control_effort': avg_control_effort,
        'avg_reward': avg_reward,
        'total_time': total_time,
        'avg_time_per_scenario': total_time / len(scenarios),
        'results': results
    }
    
    if verbose:
        print(f"📊 {controller_name} Summary: {success_rate*100:.1f}% success, {avg_steps:.1f} avg steps")
        print(f"⏱️  Total time: {total_time:.1f}s, Avg per scenario: {total_time/len(scenarios):.1f}s")
    
    return summary


def compare_controllers(results: List[Dict], verbose: bool = True):
    """Compare results from different controllers"""
    if not verbose:
        return
        
    print("\n🏆 CONTROLLER COMPARISON")
    print("=" * 60)
    
    # Sort by success rate, then by average steps
    results_sorted = sorted(results, key=lambda x: (-x['success_rate'], x['avg_steps']))
    
    print(f"{'Controller':<15} {'Success Rate':<12} {'Avg Steps':<10} {'Avg Effort':<12} {'Avg Time/Scenario':<15}")
    print("-" * 60)
    
    for result in results_sorted:
        success_pct = f"{result['success_rate']*100:.1f}%"
        steps = f"{result['avg_steps']:.1f}" if result['avg_steps'] != float('inf') else "∞"
        effort = f"{result['avg_control_effort']:.2f}" if result['avg_control_effort'] != float('inf') else "∞"
        time_per = f"{result.get('avg_time_per_scenario', 0):.1f}s"
        
        print(f"{result['controller']:<15} {success_pct:<12} {steps:<10} {effort:<12} {time_per:<15}")
    
    # Best controller analysis
    if results_sorted:
        best = results_sorted[0]
        print(f"\n🥇 Best Controller: {best['controller']}")
        print(f"   Success Rate: {best['success_rate']*100:.1f}%")
        if best['avg_steps'] != float('inf'):
            print(f"   Average Steps: {best['avg_steps']:.1f}")
        if 'avg_reward' in best:
            print(f"   Average Reward: {best['avg_reward']:.2f}")
    
    # Per-scenario breakdown
    print(f"\n📊 Per-Scenario Results:")
    print("-" * 60)
    
    if results:
        scenario_names = [r['scenario'] for r in results[0]['results']]
        
        print(f"{'Scenario':<15}", end="")
        for result in results:
            print(f"{result['controller'][:10]:<12}", end="")
        print()
        print("-" * (15 + 12 * len(results)))
        
        for scenario_name in scenario_names:
            print(f"{scenario_name:<15}", end="")
            for result in results:
                scenario_result = next(r for r in result['results'] if r['scenario'] == scenario_name)
                status = "✅" if scenario_result['success'] else "❌"
                steps = scenario_result['steps']
                print(f"{status} {steps:>3d}      ", end="")
            print()


async def main():
    parser = argparse.ArgumentParser(description="Test Real LLM Controllers")
    parser.add_argument("--controller", 
                       choices=["vllm", "openai", "both"], 
                       default="both",
                       help="Which controller(s) to test")
    parser.add_argument("--scenarios",
                       choices=["easy", "medium", "hard", "all"],
                       default="easy",
                       help="Which difficulty scenarios to test")
    parser.add_argument("--verbose", "-v",
                       action="store_true",
                       help="Verbose output")
    parser.add_argument("--save-results",
                       type=str,
                       help="Save results to JSON file")
    
    args = parser.parse_args()
    
    print("🚀 LLM Controller Testing")
    print("=" * 50)
    
    # Create scenarios
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
    
    # Test PD baseline
    results = []
    pd_result = test_pd_baseline(scenarios, verbose=args.verbose)
    results.append(pd_result)
    
    # Test LLM controllers
    controllers_to_test = []
    
    if args.controller in ["vllm", "both"] and REQUESTS_AVAILABLE:
        try:
            vllm_controller = VLLMController()
            controllers_to_test.append(("vLLM", vllm_controller))
        except Exception as e:
            print(f"⚠️  vLLM not available: {e}")
            if args.controller == "vllm":
                print("💡 To use vLLM:")
                print("   1. Install: pip install vllm requests")
                print("   2. Start server: vllm serve Qwen/Qwen2.5-1.5B-Instruct --port 8000")
                return
    
    if args.controller in ["openai", "both"] and OPENAI_AVAILABLE:
        try:
            openai_controller = OpenAIController()
            controllers_to_test.append(("OpenAI", openai_controller))
        except Exception as e:
            print(f"⚠️  OpenAI not available: {e}")
            if args.controller == "openai":
                print("💡 To use OpenAI:")
                print("   1. Install: pip install openai")
                print("   2. Set API key: export OPENAI_API_KEY=your_key")
                return
    
    # Test each LLM controller
    for name, controller in controllers_to_test:
        result = await test_llm_controller(
            controller=controller,
            controller_name=name,
            scenarios=scenarios,
            verbose=args.verbose
        )
        results.append(result)
    
    # Compare results
    compare_controllers(results, verbose=True)
    
    # Save results if requested
    if args.save_results:
        with open(args.save_results, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to {args.save_results}")
    
    print("\n🎉 Testing completed!")


if __name__ == "__main__":
    asyncio.run(main())