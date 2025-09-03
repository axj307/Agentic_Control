"""
Minimal Double Integrator Environment
===================================

Simple double integrator system: ẍ = u
This is the most basic control system for testing agent learning.

Physics:
- State: [position, velocity]
- Dynamics: position(t+1) = position(t) + velocity(t) * dt
-           velocity(t+1) = velocity(t) + control_force * dt
- Control: force/acceleration input (u)
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Optional


class DoubleIntegrator:
    """
    Simple double integrator: ẍ = u
    
    Perfect for testing because:
    1. Linear dynamics (easy to understand)
    2. Exact analytical solutions available
    3. Well-known optimal control (LQR)
    4. Minimal state/action dimensions
    """
    
    def __init__(self, max_force: float = 1.0, dt: float = 0.1):
        self.max_force = max_force
        self.dt = dt
        
        # State tracking
        self.position = 0.0
        self.velocity = 0.0
        self.time = 0.0
        
        # History for plotting
        self.history = {
            'time': [],
            'position': [],
            'velocity': [],
            'control': []
        }
        
    def reset(self, initial_position: float = 0.0, initial_velocity: float = 0.0) -> Tuple[float, float]:
        """Reset to initial conditions"""
        self.position = initial_position
        self.velocity = initial_velocity
        self.time = 0.0
        
        # Clear history
        self.history = {'time': [], 'position': [], 'velocity': [], 'control': []}
        self._record_state()
        
        return self.position, self.velocity
    
    def step(self, control_force: float) -> Tuple[float, float]:
        """
        Apply control force and integrate dynamics
        
        Double integrator dynamics: ẍ = u
        - position(t+1) = position(t) + velocity(t) * dt
        - velocity(t+1) = velocity(t) + control_force * dt
        """
        # Clip control to physical limits
        u = np.clip(control_force, -self.max_force, self.max_force)
        
        # Exact integration for double integrator
        new_velocity = self.velocity + u * self.dt
        new_position = self.position + self.velocity * self.dt + 0.5 * u * self.dt**2
        
        # Update state
        self.position = new_position
        self.velocity = new_velocity
        self.time += self.dt
        
        # Record for plotting
        self._record_state(u)
        
        return self.position, self.velocity
    
    def _record_state(self, control: float = 0.0):
        """Record current state for plotting"""
        self.history['time'].append(self.time)
        self.history['position'].append(self.position)
        self.history['velocity'].append(self.velocity)
        self.history['control'].append(control)
    
    def get_state(self) -> Tuple[float, float]:
        """Get current state"""
        return self.position, self.velocity
    
    def get_state_string(self) -> str:
        """Get human-readable state description"""
        return f"Position: {self.position:.3f} m, Velocity: {self.velocity:.3f} m/s"
    
    def is_at_target(self, target_pos: float, target_vel: float = 0.0, 
                     pos_tol: float = 0.1, vel_tol: float = 0.1) -> bool:
        """Check if system is at target state within tolerance"""
        pos_error = abs(self.position - target_pos)
        vel_error = abs(self.velocity - target_vel)
        return pos_error < pos_tol and vel_error < vel_tol
    
    def compute_errors(self, target_pos: float, target_vel: float = 0.0) -> Tuple[float, float]:
        """Compute position and velocity errors"""
        pos_error = target_pos - self.position
        vel_error = target_vel - self.velocity
        return pos_error, vel_error
    
    def pd_controller(self, target_pos: float, target_vel: float = 0.0, 
                      kp: float = 1.0, kd: float = 2.0) -> float:
        """
        Simple PD controller for baseline comparison
        
        Control law: u = kp * (target_pos - pos) + kd * (target_vel - vel)
        """
        pos_error, vel_error = self.compute_errors(target_pos, target_vel)
        control = kp * pos_error + kd * vel_error
        return np.clip(control, -self.max_force, self.max_force)
    
    def plot_trajectory(self, target_pos: float = None, save_path: str = None):
        """Plot the trajectory history"""
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8))
        
        time = np.array(self.history['time'])
        
        # Position plot
        ax1.plot(time, self.history['position'], 'b-', linewidth=2, label='Position')
        if target_pos is not None:
            ax1.axhline(y=target_pos, color='r', linestyle='--', label=f'Target: {target_pos:.1f}')
        ax1.set_ylabel('Position (m)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Velocity plot
        ax2.plot(time, self.history['velocity'], 'g-', linewidth=2, label='Velocity')
        ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Target: 0')
        ax2.set_ylabel('Velocity (m/s)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Control plot
        ax3.plot(time, self.history['control'], 'orange', linewidth=2, label='Control Force')
        ax3.axhline(y=self.max_force, color='r', linestyle=':', alpha=0.5, label=f'Max: ±{self.max_force}')
        ax3.axhline(y=-self.max_force, color='r', linestyle=':', alpha=0.5)
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Control Force (N)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.suptitle('Double Integrator Control Trajectory')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def create_control_scenario(initial_pos: float, initial_vel: float, 
                          target_pos: float, target_vel: float = 0.0) -> Dict:
    """Create a control scenario for testing"""
    return {
        'initial_state': (initial_pos, initial_vel),
        'target_state': (target_pos, target_vel),
        'description': f"Move from ({initial_pos:.1f}, {initial_vel:.1f}) to ({target_pos:.1f}, {target_vel:.1f})"
    }


# Test scenarios for different difficulty levels
EASY_SCENARIOS = [
    create_control_scenario(0.5, 0.0, 0.0, 0.0),  # Small position error
    create_control_scenario(-0.3, 0.0, 0.0, 0.0), # Small negative error
    create_control_scenario(0.0, 0.2, 0.0, 0.0),  # Small velocity error
]

MEDIUM_SCENARIOS = [
    create_control_scenario(1.0, 0.0, 0.0, 0.0),   # Larger position error
    create_control_scenario(0.0, 0.5, 0.0, 0.0),   # Larger velocity error
    create_control_scenario(0.8, -0.3, 0.0, 0.0),  # Both errors
]

HARD_SCENARIOS = [
    create_control_scenario(1.5, 0.0, 0.0, 0.0),   # Large position error
    create_control_scenario(-1.2, 0.8, 0.0, 0.0),  # Large mixed errors
    create_control_scenario(0.5, -0.9, 0.0, 0.0),  # High velocity error
]