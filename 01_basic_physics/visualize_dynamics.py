#!/usr/bin/env python3
"""
Visualize Double Integrator Dynamics
===================================

Create interactive visualizations to understand double integrator
behavior under different control strategies.

Run with:
    conda activate agentic_control && python visualize_dynamics.py
"""

import numpy as np
import matplotlib.pyplot as plt
from double_integrator import DoubleIntegrator, EASY_SCENARIOS, MEDIUM_SCENARIOS, HARD_SCENARIOS


def compare_control_strategies():
    """Compare different control strategies visually"""
    print("📊 Comparing Control Strategies...")
    
    # Test scenario: move from position 1.0 to 0.0
    initial_pos, initial_vel = 1.0, 0.0
    target_pos, target_vel = 0.0, 0.0
    
    strategies = {
        'PD Control (kp=1, kd=2)': {'kp': 1.0, 'kd': 2.0},
        'PD Control (kp=2, kd=1)': {'kp': 2.0, 'kd': 1.0},
        'PD Control (kp=0.5, kd=3)': {'kp': 0.5, 'kd': 3.0},
    }
    
    plt.figure(figsize=(15, 10))
    
    for i, (name, params) in enumerate(strategies.items()):
        env = DoubleIntegrator(max_force=1.0, dt=0.1)
        env.reset(initial_pos, initial_vel)
        
        # Run control strategy
        for step in range(50):
            control = env.pd_controller(target_pos, target_vel, 
                                      kp=params['kp'], kd=params['kd'])
            env.step(control)
            
            if env.is_at_target(target_pos, target_vel, pos_tol=0.05, vel_tol=0.05):
                break
        
        # Plot results
        time = np.array(env.history['time'])
        
        # Position subplot
        plt.subplot(3, 1, 1)
        plt.plot(time, env.history['position'], linewidth=2, label=name)
        plt.axhline(y=target_pos, color='red', linestyle='--', alpha=0.5)
        plt.ylabel('Position (m)')
        plt.title('Position vs Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Velocity subplot
        plt.subplot(3, 1, 2)
        plt.plot(time, env.history['velocity'], linewidth=2, label=name)
        plt.axhline(y=target_vel, color='red', linestyle='--', alpha=0.5)
        plt.ylabel('Velocity (m/s)')
        plt.title('Velocity vs Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Control subplot
        plt.subplot(3, 1, 3)
        plt.plot(time, env.history['control'], linewidth=2, label=name)
        plt.axhline(y=1.0, color='red', linestyle=':', alpha=0.5)
        plt.axhline(y=-1.0, color='red', linestyle=':', alpha=0.5)
        plt.xlabel('Time (s)')
        plt.ylabel('Control Force (N)')
        plt.title('Control Force vs Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.suptitle('Comparison of PD Control Strategies', fontsize=14)
    plt.tight_layout()
    plt.show()


def create_phase_portrait():
    """Create phase portrait showing different trajectories"""
    print("🌀 Creating Phase Portrait...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Left plot: Different initial conditions with same control
    ax1.set_title('Phase Portrait: Different Initial Conditions')
    
    initial_conditions = [
        (1.0, 0.0), (0.5, 0.5), (-0.5, 0.3), (0.8, -0.4), 
        (-1.0, 0.0), (0.2, -0.8), (-0.3, 0.6)
    ]
    
    for i, (init_pos, init_vel) in enumerate(initial_conditions):
        env = DoubleIntegrator(max_force=1.0, dt=0.1)
        env.reset(init_pos, init_vel)
        
        # Run PD controller to origin
        for _ in range(40):
            control = env.pd_controller(0.0, 0.0, kp=1.0, kd=2.0)
            env.step(control)
            
            if env.is_at_target(0.0, 0.0, pos_tol=0.05, vel_tol=0.05):
                break
        
        positions = np.array(env.history['position'])
        velocities = np.array(env.history['velocity'])
        
        ax1.plot(positions, velocities, linewidth=2, alpha=0.7, 
                label=f'IC: ({init_pos:.1f}, {init_vel:.1f})')
        ax1.plot(positions[0], velocities[0], 'o', markersize=8)  # Start
        ax1.plot(positions[-1], velocities[-1], 's', markersize=6)  # End
    
    ax1.plot(0, 0, 'r*', markersize=15, label='Target (0,0)')
    ax1.set_xlabel('Position (m)')
    ax1.set_ylabel('Velocity (m/s)')
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Right plot: Different control gains
    ax2.set_title('Phase Portrait: Different Control Gains')
    
    control_gains = [
        {'kp': 0.5, 'kd': 1.0, 'color': 'blue', 'label': 'kp=0.5, kd=1.0'},
        {'kp': 1.0, 'kd': 2.0, 'color': 'green', 'label': 'kp=1.0, kd=2.0'},
        {'kp': 2.0, 'kd': 1.0, 'color': 'orange', 'label': 'kp=2.0, kd=1.0'},
        {'kp': 1.5, 'kd': 3.0, 'color': 'purple', 'label': 'kp=1.5, kd=3.0'}
    ]
    
    initial_pos, initial_vel = 1.0, -0.5
    
    for gains in control_gains:
        env = DoubleIntegrator(max_force=1.0, dt=0.1)
        env.reset(initial_pos, initial_vel)
        
        for _ in range(50):
            control = env.pd_controller(0.0, 0.0, kp=gains['kp'], kd=gains['kd'])
            env.step(control)
            
            if env.is_at_target(0.0, 0.0, pos_tol=0.05, vel_tol=0.05):
                break
        
        positions = np.array(env.history['position'])
        velocities = np.array(env.history['velocity'])
        
        ax2.plot(positions, velocities, color=gains['color'], linewidth=2, 
                alpha=0.8, label=gains['label'])
        ax2.plot(positions[0], velocities[0], 'o', color=gains['color'], markersize=8)
    
    ax2.plot(0, 0, 'r*', markersize=15, label='Target (0,0)')
    ax2.plot(initial_pos, initial_vel, 'ko', markersize=10, label='Start')
    ax2.set_xlabel('Position (m)')
    ax2.set_ylabel('Velocity (m/s)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.show()


def analyze_scenario_difficulty():
    """Analyze different scenario difficulties"""
    print("📈 Analyzing Scenario Difficulty...")
    
    all_scenarios = {
        'Easy': EASY_SCENARIOS,
        'Medium': MEDIUM_SCENARIOS,
        'Hard': HARD_SCENARIOS
    }
    
    results = {}
    
    for difficulty, scenarios in all_scenarios.items():
        print(f"\nTesting {difficulty} scenarios...")
        
        difficulty_results = []
        
        for scenario in scenarios:
            env = DoubleIntegrator(max_force=1.0, dt=0.1)
            initial_pos, initial_vel = scenario['initial_state']
            target_pos, target_vel = scenario['target_state']
            
            env.reset(initial_pos, initial_vel)
            
            # Run PD controller
            for step in range(100):
                control = env.pd_controller(target_pos, target_vel, kp=1.0, kd=2.0)
                env.step(control)
                
                if env.is_at_target(target_pos, target_vel, pos_tol=0.05, vel_tol=0.05):
                    break
            
            # Calculate metrics
            pos_error = abs(env.position - target_pos)
            vel_error = abs(env.velocity - target_vel)
            control_effort = sum(abs(u) for u in env.history['control'])
            settling_time = len(env.history['time']) * env.dt
            
            difficulty_results.append({
                'initial_error': np.sqrt((initial_pos - target_pos)**2 + (initial_vel - target_vel)**2),
                'final_pos_error': pos_error,
                'final_vel_error': vel_error,
                'control_effort': control_effort,
                'settling_time': settling_time,
                'success': pos_error < 0.1 and vel_error < 0.1
            })
        
        results[difficulty] = difficulty_results
    
    # Plot comparison
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    
    difficulties = list(results.keys())
    colors = ['green', 'orange', 'red']
    
    for i, (difficulty, color) in enumerate(zip(difficulties, colors)):
        data = results[difficulty]
        
        # Success rate
        success_rate = sum(r['success'] for r in data) / len(data)
        ax1.bar(difficulty, success_rate, color=color, alpha=0.7)
        
        # Average control effort
        avg_effort = np.mean([r['control_effort'] for r in data])
        ax2.bar(difficulty, avg_effort, color=color, alpha=0.7)
        
        # Average settling time
        avg_time = np.mean([r['settling_time'] for r in data])
        ax3.bar(difficulty, avg_time, color=color, alpha=0.7)
        
        # Initial error vs final error
        initial_errors = [r['initial_error'] for r in data]
        final_errors = [r['final_pos_error'] + r['final_vel_error'] for r in data]
        ax4.scatter(initial_errors, final_errors, color=color, alpha=0.7, 
                   label=difficulty, s=50)
    
    ax1.set_title('Success Rate by Difficulty')
    ax1.set_ylabel('Success Rate')
    ax1.set_ylim(0, 1.1)
    
    ax2.set_title('Average Control Effort')
    ax2.set_ylabel('Control Effort')
    
    ax3.set_title('Average Settling Time')
    ax3.set_ylabel('Time (s)')
    
    ax4.set_title('Initial vs Final Error')
    ax4.set_xlabel('Initial Error')
    ax4.set_ylabel('Final Error')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('PD Controller Performance Analysis', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    return results


def run_all_visualizations():
    """Run all visualization functions"""
    print("🎨 Running All Visualization Tests")
    print("=" * 50)
    
    try:
        compare_control_strategies()
        create_phase_portrait()
        results = analyze_scenario_difficulty()
        
        print("\n🎉 All visualizations completed!")
        print("\n📋 Key Insights:")
        print("1. Higher kp leads to faster response but more oscillation")
        print("2. Higher kd provides better damping but slower response")
        print("3. Phase portraits show convergence to target state")
        print("4. Scenario difficulty correlates with control effort and settling time")
        
        return results
        
    except Exception as e:
        print(f"\n❌ Visualization failed: {e}")
        raise


if __name__ == "__main__":
    results = run_all_visualizations()