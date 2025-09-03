#!/usr/bin/env python3
"""
Test Double Integrator Environment
================================

Verify that the double integrator physics work correctly and test 
different control strategies as baselines.

Run with:
    conda activate agentic_control && python test_environment.py
"""

import numpy as np
import matplotlib.pyplot as plt
from double_integrator import DoubleIntegrator, EASY_SCENARIOS, MEDIUM_SCENARIOS


def test_basic_physics():
    """Test basic physics simulation"""
    print("🧪 Testing Basic Physics...")
    
    env = DoubleIntegrator(max_force=1.0, dt=0.1)
    
    # Test 1: Zero control (system should maintain velocity)
    print("\nTest 1: Zero Control")
    pos, vel = env.reset(initial_position=0.0, initial_velocity=0.5)
    print(f"Initial: {env.get_state_string()}")
    
    for i in range(5):
        pos, vel = env.step(control_force=0.0)
        print(f"Step {i+1}: {env.get_state_string()}")
    
    # Should move at constant velocity
    expected_pos = 0.5 * 5 * 0.1  # vel * steps * dt
    assert abs(pos - expected_pos) < 1e-10, f"Physics error: expected {expected_pos}, got {pos}"
    assert abs(vel - 0.5) < 1e-10, f"Velocity should be constant: expected 0.5, got {vel}"
    
    print("✅ Zero control test passed")
    
    # Test 2: Constant force
    print("\nTest 2: Constant Force")
    pos, vel = env.reset(initial_position=0.0, initial_velocity=0.0)
    
    for i in range(10):
        pos, vel = env.step(control_force=1.0)
    
    # After time t with constant acceleration a: v = a*t, x = 0.5*a*t^2
    t = 10 * 0.1
    expected_vel = 1.0 * t
    expected_pos = 0.5 * 1.0 * t**2
    
    assert abs(vel - expected_vel) < 1e-6, f"Velocity error: expected {expected_vel}, got {vel}"
    assert abs(pos - expected_pos) < 1e-6, f"Position error: expected {expected_pos}, got {pos}"
    
    print("✅ Constant force test passed")
    
    print("🎉 All physics tests passed!")


def test_pd_controller():
    """Test PD controller baseline"""
    print("\n🎮 Testing PD Controller...")
    
    env = DoubleIntegrator(max_force=1.0, dt=0.1)
    
    # Test different scenarios
    scenarios = EASY_SCENARIOS + MEDIUM_SCENARIOS[:2]  # Test a few scenarios
    
    results = []
    
    for i, scenario in enumerate(scenarios):
        print(f"\nScenario {i+1}: {scenario['description']}")
        
        # Reset to initial state
        initial_pos, initial_vel = scenario['initial_state']
        target_pos, target_vel = scenario['target_state']
        
        env.reset(initial_pos, initial_vel)
        print(f"Initial: {env.get_state_string()}")
        print(f"Target: Position: {target_pos:.3f} m, Velocity: {target_vel:.3f} m/s")
        
        # Run PD controller
        max_steps = 50
        for step in range(max_steps):
            control = env.pd_controller(target_pos, target_vel, kp=1.0, kd=2.0)
            pos, vel = env.step(control)
            
            # Check if reached target
            if env.is_at_target(target_pos, target_vel, pos_tol=0.05, vel_tol=0.05):
                print(f"✅ Reached target in {step+1} steps")
                break
        else:
            print(f"⚠️ Did not reach target in {max_steps} steps")
        
        print(f"Final: {env.get_state_string()}")
        
        # Calculate performance metrics
        pos_error = abs(env.position - target_pos)
        vel_error = abs(env.velocity - target_vel)
        control_effort = sum(abs(u) for u in env.history['control'])
        
        results.append({
            'scenario': i+1,
            'success': pos_error < 0.1 and vel_error < 0.1,
            'final_pos_error': pos_error,
            'final_vel_error': vel_error,
            'control_effort': control_effort,
            'steps': len(env.history['time']) - 1
        })
    
    # Summary
    print("\n📊 PD Controller Results Summary:")
    success_rate = sum(1 for r in results if r['success']) / len(results)
    avg_effort = np.mean([r['control_effort'] for r in results])
    avg_steps = np.mean([r['steps'] for r in results])
    
    print(f"Success Rate: {success_rate:.1%}")
    print(f"Average Control Effort: {avg_effort:.2f}")
    print(f"Average Steps to Target: {avg_steps:.1f}")
    
    return results


def test_visualization():
    """Test visualization capabilities"""
    print("\n📈 Testing Visualization...")
    
    env = DoubleIntegrator(max_force=1.0, dt=0.1)
    
    # Create an interesting trajectory
    env.reset(initial_position=1.0, initial_velocity=-0.5)
    target_pos = 0.0
    
    print("Creating sample trajectory with PD control...")
    for step in range(30):
        control = env.pd_controller(target_pos, kp=1.0, kd=2.0)
        env.step(control)
        
        if step % 10 == 0:
            print(f"Step {step}: {env.get_state_string()}")
    
    # Plot trajectory
    print("Generating plot...")
    env.plot_trajectory(target_pos=target_pos)
    
    print("✅ Visualization test completed")


def run_all_tests():
    """Run all tests"""
    print("🚀 Running All Double Integrator Tests")
    print("=" * 50)
    
    try:
        test_basic_physics()
        test_pd_controller()
        test_visualization()
        
        print("\n🎉 All tests completed successfully!")
        print("\n✅ Double integrator environment is working correctly")
        print("✅ PD controller baseline is functional")
        print("✅ Visualization system works")
        
        print("\n📋 Next Steps:")
        print("1. Move to 02_direct_control/ to implement LLM-based control")
        print("2. Create simple agent that mimics PD controller behavior")
        print("3. Test agent performance vs PD baseline")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise


if __name__ == "__main__":
    run_all_tests()