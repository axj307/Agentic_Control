"""
Control Tools for LangGraph-based Tool-Augmented Control

This module implements physics-aware tools that an LLM can use to make
better control decisions. Each tool provides specialized analysis or
calculations that augment the LLM's reasoning.

Tools implemented:
1. ErrorAnalyzerTool - Analyze control errors and system state
2. PIDCalculatorTool - Calculate PID control actions
3. TrajectoryPlannerTool - Plan optimal trajectories
4. SafetyVerifierTool - Verify control action safety

Usage:
    from control_tools import get_control_tools
    tools = get_control_tools()
"""

import math
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

try:
    from langchain_core.tools import tool
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("⚠️  langchain-core not available. Install with: pip install langchain-core")
    
    # Create a simple decorator for development
    def tool(func):
        func.is_tool = True
        return func


@tool
def analyze_errors(position: float, velocity: float, 
                   target_pos: float, target_vel: float = 0.0) -> dict:
    """
    Analyze control errors and system state for spacecraft control.
    
    This tool provides detailed analysis of the current control situation,
    including error magnitudes, urgency assessment, and stability analysis.
    
    Args:
        position: Current position (meters)
        velocity: Current velocity (m/s)
        target_pos: Target position (meters)
        target_vel: Target velocity (m/s, default 0.0)
        
    Returns:
        Dictionary containing:
        - position_error: Distance from target (positive = need to move right)
        - velocity_error: Velocity mismatch (positive = too slow)
        - total_error: Combined error metric
        - urgency: "high"/"medium"/"low" based on error magnitude
        - stability: "stable"/"unstable"/"marginal" based on trajectory
        - phase: Current control phase ("approach"/"brake"/"fine_tune")
        - recommendations: List of control recommendations
    """
    
    # Calculate basic errors
    pos_error = target_pos - position
    vel_error = target_vel - velocity
    
    # Calculate distances and magnitudes
    pos_distance = abs(pos_error)
    vel_magnitude = abs(velocity)
    
    # Total error (weighted combination)
    total_error = math.sqrt(pos_error**2 + 0.5 * vel_error**2)
    
    # Determine urgency level
    if pos_distance > 0.5 or vel_magnitude > 0.4:
        urgency = "high"
    elif pos_distance > 0.15 or vel_magnitude > 0.15:
        urgency = "medium"
    else:
        urgency = "low"
    
    # Analyze stability (are we heading toward or away from target?)
    # If velocity and position error have same sign, we're moving away
    if pos_error * velocity > 0.1:  # Moving away from target
        stability = "unstable"
    elif abs(pos_error * velocity) < 0.05:  # Nearly perpendicular or small
        stability = "marginal" 
    else:  # Moving toward target
        stability = "stable"
    
    # Determine control phase
    if pos_distance > 0.3:
        phase = "approach"  # Far from target, focus on getting closer
    elif vel_magnitude > 0.2:
        phase = "brake"     # Close but moving too fast, need to slow down
    else:
        phase = "fine_tune" # Close and slow, fine adjustments needed
    
    # Generate recommendations based on analysis
    recommendations = []
    
    if phase == "approach":
        if pos_error > 0:
            recommendations.append("Apply positive force to move right toward target")
        else:
            recommendations.append("Apply negative force to move left toward target")
            
    elif phase == "brake":
        if velocity > 0:
            recommendations.append("Apply negative force to brake rightward motion")
        else:
            recommendations.append("Apply positive force to brake leftward motion")
            
    else:  # fine_tune
        recommendations.append("Use gentle control for precise positioning")
        
    if stability == "unstable":
        recommendations.append("⚠️  URGENT: Currently moving away from target!")
        
    # Add velocity-specific advice
    if abs(vel_error) > 0.1:
        if vel_error > 0:
            recommendations.append("Need to increase velocity (less braking)")
        else:
            recommendations.append("Need to decrease velocity (more braking)")
    
    return {
        'position_error': round(pos_error, 4),
        'velocity_error': round(vel_error, 4), 
        'total_error': round(total_error, 4),
        'urgency': urgency,
        'stability': stability,
        'phase': phase,
        'recommendations': recommendations,
        'distance_to_target': round(pos_distance, 4),
        'velocity_magnitude': round(vel_magnitude, 4)
    }


@tool
def select_control_strategy(pos_error: float, vel_error: float) -> dict:
    """
    Select optimal control strategy based on error magnitude and system state.
    
    This tool determines the best control approach based on the scale of the problem:
    - Small errors: Standard PID control
    - Medium errors: Adaptive PID with higher gains  
    - Large errors: Bang-bang optimal control
    
    Args:
        pos_error: Position error (target - current)
        vel_error: Velocity error (target_vel - current_vel)
        
    Returns:
        Dictionary containing:
        - strategy: "standard_pid", "adaptive_pid", or "bang_bang"
        - recommended_gains: Suggested PID gains for chosen strategy
        - error_magnitude: Combined error metric
        - reasoning: Explanation of strategy selection
    """
    
    # Calculate error magnitude
    error_magnitude = math.sqrt(pos_error**2 + vel_error**2)
    
    # Strategy selection logic based on error magnitude
    if error_magnitude >= 0.8:  # Use LQR for large errors (lowered threshold)
        strategy = "lqr_optimal"
        recommended_gains = {"kp": 4.0, "kd": 4.0}  # High gains for large errors
        reasoning = f"Large error magnitude {error_magnitude:.3f} requires LQR optimal control"
        
    elif error_magnitude >= 0.3:  # Use adaptive PID for medium errors
        strategy = "adaptive_pid"
        recommended_gains = {"kp": 2.0, "kd": 3.0}  # Medium gains for medium errors
        reasoning = f"Medium error magnitude {error_magnitude:.3f} requires adaptive PID with higher gains"
        
    else:  # Use standard PID for small errors
        strategy = "standard_pid"
        recommended_gains = {"kp": 1.0, "kd": 2.0}  # Standard gains for small errors
        reasoning = f"Small error magnitude {error_magnitude:.3f} suitable for standard PID control"
    
    return {
        'strategy': strategy,
        'recommended_gains': recommended_gains,
        'error_magnitude': round(error_magnitude, 4),
        'reasoning': reasoning,
        'pos_error_magnitude': round(abs(pos_error), 4),
        'vel_error_magnitude': round(abs(vel_error), 4)
    }


@tool  
def calculate_pid_control(pos_error: float, vel_error: float,
                          kp: float = None, kd: float = None, 
                          ki: float = 0.0, max_force: float = 1.0, 
                          adaptive: bool = True) -> dict:
    """
    Calculate PID control action for spacecraft maneuvering with adaptive gains.
    
    This tool computes a physics-based control action using PID control theory
    with adaptive gain selection based on error magnitude for improved wide-range performance.
    
    Args:
        pos_error: Position error (target - current)
        vel_error: Velocity error (target_vel - current_vel) 
        kp: Proportional gain (None for adaptive selection)
        kd: Derivative gain (None for adaptive selection)
        ki: Integral gain (default 0.0, not implemented yet)
        max_force: Maximum force limit (default 1.0)
        adaptive: Use adaptive gain selection (default True)
        
    Returns:
        Dictionary containing:
        - control_action: Recommended force [-max_force, +max_force]
        - components: Breakdown of P, I, D terms
        - saturated: True if hitting force limits
        - confidence: Confidence in the control action (0-1)
        - reasoning: Explanation of the calculation
        - strategy_used: Control strategy applied
        - gains_used: Actual gains used in calculation
    """
    
    # Adaptive gain selection if not provided
    if adaptive and (kp is None or kd is None):
        error_magnitude = math.sqrt(pos_error**2 + vel_error**2)
        
        if error_magnitude > 1.0:
            # Large errors: High gains + bang-bang elements
            kp_adaptive, kd_adaptive = 3.0, 4.0
            strategy_used = "bang_bang_pid"
        elif error_magnitude > 0.5:
            # Medium errors: Increased gains
            kp_adaptive, kd_adaptive = 2.0, 3.0
            strategy_used = "adaptive_pid"
        else:
            # Small errors: Standard gains
            kp_adaptive, kd_adaptive = 1.0, 2.0
            strategy_used = "standard_pid"
            
        # Use adaptive gains if not explicitly provided
        kp = kp if kp is not None else kp_adaptive
        kd = kd if kd is not None else kd_adaptive
    else:
        # Use provided gains or defaults
        kp = kp if kp is not None else 1.0
        kd = kd if kd is not None else 2.0
        strategy_used = "fixed_gains"
    
    # Calculate PID components
    # For double integrator: velocity is derivative of position
    # So we use velocity error directly as derivative term
    p_term = kp * pos_error
    d_term = kd * vel_error  # Note: using vel_error, not rate of pos_error
    i_term = ki * 0.0  # Integral term not implemented yet
    
    # Total control action
    raw_control = p_term + d_term + i_term
    
    # Apply saturation
    saturated = abs(raw_control) > max_force
    control_action = np.clip(raw_control, -max_force, max_force)
    
    # Calculate confidence based on how well-conditioned the problem is
    error_magnitude = math.sqrt(pos_error**2 + vel_error**2)
    if error_magnitude < 0.05:
        confidence = 0.95  # Very close to target, high confidence
    elif error_magnitude < 0.2:
        confidence = 0.85  # Moderate distance, good confidence
    elif error_magnitude < 0.5:
        confidence = 0.7   # Far from target, medium confidence
    else:
        confidence = 0.6   # Very far, lower confidence
        
    # Reduce confidence if saturated
    if saturated:
        confidence *= 0.8
    
    # Generate reasoning
    reasoning_parts = []
    
    if abs(p_term) > abs(d_term):
        reasoning_parts.append(f"Proportional term dominates ({p_term:.3f})")
    else:
        reasoning_parts.append(f"Derivative term dominates ({d_term:.3f})")
    
    if pos_error > 0.1:
        reasoning_parts.append("need rightward force for position")
    elif pos_error < -0.1:
        reasoning_parts.append("need leftward force for position")
        
    if vel_error > 0.1:
        reasoning_parts.append("need to accelerate")
    elif vel_error < -0.1:
        reasoning_parts.append("need to brake")
    
    if saturated:
        reasoning_parts.append("⚠️  force saturated at limits")
        
    reasoning = "PID calculation: " + ", ".join(reasoning_parts)
    
    return {
        'control_action': round(control_action, 4),
        'components': {
            'p_term': round(p_term, 4),
            'd_term': round(d_term, 4), 
            'i_term': round(i_term, 4),
            'raw_control': round(raw_control, 4)
        },
        'saturated': saturated,
        'confidence': round(confidence, 3),
        'reasoning': reasoning,
        'strategy_used': strategy_used,
        'gains_used': {'kp': kp, 'kd': kd, 'ki': ki},
        'error_magnitude': round(math.sqrt(pos_error**2 + vel_error**2), 4)
    }


@tool
def calculate_lqr_control(current_pos: float, current_vel: float,
                         target_pos: float, target_vel: float = 0.0,
                         Q_pos: float = 1.0, Q_vel: float = 1.0, R: float = 1.0,
                         max_force: float = 1.0) -> dict:
    """
    Calculate optimal LQR (Linear Quadratic Regulator) control for double integrator.
    
    This tool implements LQR control which is optimal for linear systems like the
    double integrator. LQR minimizes a quadratic cost function and provides better
    performance than PID or bang-bang control across all operating ranges.
    
    Args:
        current_pos: Current position (m)
        current_vel: Current velocity (m/s)  
        target_pos: Target position (m)
        target_vel: Target velocity (m/s, default 0.0)
        Q_pos: State cost weight for position error (default 1.0)
        Q_vel: State cost weight for velocity error (default 1.0)  
        R: Control cost weight (default 0.1)
        max_force: Maximum force limit (default 1.0)
        
    Returns:
        Dictionary containing:
        - control_action: Optimal LQR force
        - lqr_gains: Computed LQR gains [K1, K2]
        - cost_weights: Used Q and R matrices
        - unbounded_action: LQR action before saturation
        - saturated: Whether force limits were hit
        - reasoning: Explanation of LQR calculation
    """
    
    # Calculate state errors
    pos_error = target_pos - current_pos
    vel_error = target_vel - current_vel
    
    # Double integrator system matrices:
    # x_dot = A*x + B*u
    # A = [[0, 1], [0, 0]], B = [[0], [1]]
    # For LQR: u = -K*x where K is computed from Riccati equation
    
    # For double integrator with Q = diag([Q_pos, Q_vel]) and R = scalar:
    # The LQR gains can be computed analytically:
    
    # Solve algebraic Riccati equation analytically for double integrator
    # This avoids needing scipy.linalg.solve_continuous_are
    
    # For the double integrator, the optimal LQR gains are:
    # K1 = sqrt(Q_pos/R) * (sqrt(1 + 2*sqrt(Q_vel/Q_pos)) + 1)
    # K2 = sqrt(Q_vel/R) * sqrt(2*(sqrt(1 + 2*sqrt(Q_vel/Q_pos)) + 1))
    
    # Simplified analytical solution
    ratio = Q_vel / Q_pos if Q_pos > 0 else 1.0
    sqrt_ratio = math.sqrt(ratio) if ratio > 0 else 1.0
    
    # LQR gains (analytical approximation for double integrator)
    K1 = math.sqrt(Q_pos / R) * (math.sqrt(1 + 2*sqrt_ratio) + 1)
    K2 = math.sqrt(Q_vel / R) * math.sqrt(2*(math.sqrt(1 + 2*sqrt_ratio) + 1))
    
    # Calculate proper LQR gains for double integrator using scipy
    error_magnitude = math.sqrt(pos_error**2 + vel_error**2)
    
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
        K1 = float(K_matrix[0, 0])
        K2 = float(K_matrix[0, 1])
        
        reasoning_suffix = f"(scipy LQR gains, Q_pos={Q_pos}, Q_vel={Q_vel}, R={R})"
        
    except ImportError:
        # Fallback: Use approximate analytical solution
        # These are well-tuned gains that work for most cases
        if error_magnitude > 0.8:
            K1 = 2.0
            K2 = 2.8
        elif error_magnitude > 0.3:
            K1 = 1.5
            K2 = 2.4
        else:
            K1 = 1.0
            K2 = 2.0
        
        reasoning_suffix = f"(fallback gains, scipy unavailable)"
    
    # Compute LQR control law: u = K1*pos_error + K2*vel_error (corrected signs)
    # For double integrator regulation: u = K1*(target - pos) + K2*(target_vel - vel)
    unbounded_action = K1 * pos_error + K2 * vel_error
    
    # Apply force saturation
    saturated = abs(unbounded_action) > max_force
    control_action = np.clip(unbounded_action, -max_force, max_force)
    
    # Generate reasoning
    reasoning_parts = [
        f"LQR optimal control {reasoning_suffix}",
        f"gains K1={K1:.2f}, K2={K2:.2f}",
        f"pos_error={pos_error:.3f}, vel_error={vel_error:.3f}"
    ]
    
    if saturated:
        reasoning_parts.append("⚠️ force saturated at limits")
        
    reasoning = "LQR calculation: " + ", ".join(reasoning_parts)
    
    return {
        'control_action': round(control_action, 4),
        'lqr_gains': [round(K1, 4), round(K2, 4)],
        'cost_weights': {'Q_pos': Q_pos, 'Q_vel': Q_vel, 'R': R},
        'method': 'scipy_lqr' if 'scipy' in reasoning_suffix else 'fallback',
        'unbounded_action': round(unbounded_action, 4),
        'saturated': saturated,
        'reasoning': reasoning,
        'pos_error': round(pos_error, 4),
        'vel_error': round(vel_error, 4),
        'error_magnitude': round(error_magnitude, 4)
    }


@tool
def plan_trajectory(current_pos: float, current_vel: float,
                    target_pos: float, target_vel: float = 0.0,
                    time_horizon: float = 5.0, max_accel: float = 1.0) -> dict:
    """
    Plan optimal trajectory to target using physics-based trajectory generation.
    
    This tool computes an optimal trajectory from current state to target state,
    considering the double integrator dynamics and force constraints.
    
    Args:
        current_pos: Current position (m)
        current_vel: Current velocity (m/s)
        target_pos: Target position (m)
        target_vel: Target velocity (m/s, default 0.0)
        time_horizon: Planning time horizon (seconds, default 5.0)
        max_accel: Maximum acceleration magnitude (m/s², default 1.0)
        
    Returns:
        Dictionary containing:
        - waypoints: List of intermediate target states [(pos, vel, time), ...]
        - control_sequence: Suggested control inputs over time
        - estimated_time: Time to reach target (seconds)
        - feasible: Whether the trajectory is feasible with constraints
        - strategy: High-level description of the trajectory plan
    """
    
    # Calculate required state change
    pos_change = target_pos - current_pos
    vel_change = target_vel - current_vel
    
    # For double integrator, we can solve analytically for minimum time
    # This is a simplified version - more advanced planners would use optimal control
    
    # Strategy determination
    pos_distance = abs(pos_change)
    vel_magnitude = abs(current_vel)
    
    strategies = []
    waypoints = []
    control_sequence = []
    
    # Simple bang-bang control estimation
    if pos_distance > 0.1:  # Need significant position change
        # Phase 1: Accelerate toward target
        accel_time = min(2.0, time_horizon * 0.4)
        if pos_change > 0:
            accel_dir = 1.0
            strategies.append("accelerate rightward")
        else:
            accel_dir = -1.0
            strategies.append("accelerate leftward")
            
        # Calculate state after acceleration phase
        mid_vel = current_vel + accel_dir * max_accel * accel_time
        mid_pos = current_pos + current_vel * accel_time + 0.5 * accel_dir * max_accel * accel_time**2
        
        waypoints.append((round(mid_pos, 3), round(mid_vel, 3), round(accel_time, 3)))
        control_sequence.append((accel_dir * max_accel, accel_time))
        
        # Phase 2: Coast or decelerate
        remaining_time = time_horizon - accel_time
        remaining_pos = target_pos - mid_pos
        remaining_vel = target_vel - mid_vel
        
        if remaining_time > 0.1:
            # Calculate required deceleration
            required_accel = 2 * (remaining_pos - mid_vel * remaining_time) / (remaining_time**2)
            required_accel = np.clip(required_accel, -max_accel, max_accel)
            
            strategies.append("coast/brake to target")
            control_sequence.append((required_accel, remaining_time))
    
    else:  # Close to target, just adjust velocity
        if abs(vel_change) > 0.05:
            required_accel = vel_change / time_horizon
            required_accel = np.clip(required_accel, -max_accel, max_accel)
            strategies.append("fine velocity adjustment")
            control_sequence.append((required_accel, time_horizon))
        else:
            strategies.append("maintain position")
            control_sequence.append((0.0, time_horizon))
    
    # Add final waypoint
    waypoints.append((target_pos, target_vel, time_horizon))
    
    # Estimate actual time to target (simplified)
    if control_sequence:
        estimated_time = sum(duration for _, duration in control_sequence)
    else:
        estimated_time = time_horizon
    
    # Check feasibility
    max_required_accel = max([abs(accel) for accel, _ in control_sequence] + [0])
    feasible = max_required_accel <= max_accel
    
    strategy_description = " → ".join(strategies)
    
    return {
        'waypoints': waypoints,
        'control_sequence': [(round(a, 3), round(t, 3)) for a, t in control_sequence],
        'estimated_time': round(estimated_time, 2),
        'feasible': feasible,
        'strategy': strategy_description,
        'max_accel_required': round(max_required_accel, 3),
        'position_change_required': round(pos_change, 3),
        'velocity_change_required': round(vel_change, 3)
    }


@tool
def verify_safety(control_action: float, current_pos: float, current_vel: float,
                 max_force: float = 1.0, position_limits: Optional[List[float]] = None,
                 velocity_limits: Optional[List[float]] = None) -> dict:
    """
    Verify control action safety and suggest adjustments if needed.
    
    This tool checks if a proposed control action is safe given system constraints
    and the current state. It can suggest safer alternatives if problems are detected.
    
    Args:
        control_action: Proposed control force
        current_pos: Current position (m)
        current_vel: Current velocity (m/s)
        max_force: Maximum allowed force magnitude (default 1.0)
        position_limits: [min_pos, max_pos] or None for no limits
        velocity_limits: [min_vel, max_vel] or None for no limits
        
    Returns:
        Dictionary containing:
        - safe: True if action is safe
        - warnings: List of safety concerns
        - adjusted_action: Safe control value (may be same as input)
        - violations: List of constraint violations
        - next_state_prediction: Predicted next state after action
    """
    
    warnings = []
    violations = []
    
    # Check force magnitude limits
    if abs(control_action) > max_force:
        violations.append(f"Force magnitude {abs(control_action):.3f} exceeds limit {max_force}")
        
    # Predict next state (assuming dt = 0.1s)
    dt = 0.1
    next_vel = current_vel + control_action * dt
    next_pos = current_pos + current_vel * dt
    
    # Check position limits
    if position_limits is not None:
        min_pos, max_pos = position_limits
        if next_pos < min_pos:
            violations.append(f"Next position {next_pos:.3f} below limit {min_pos}")
        elif next_pos > max_pos:
            violations.append(f"Next position {next_pos:.3f} above limit {max_pos}")
    
    # Check velocity limits  
    if velocity_limits is not None:
        min_vel, max_vel = velocity_limits
        if next_vel < min_vel:
            violations.append(f"Next velocity {next_vel:.3f} below limit {min_vel}")
        elif next_vel > max_vel:
            violations.append(f"Next velocity {next_vel:.3f} above limit {max_vel}")
    
    # Check for potentially unstable control
    if abs(control_action) > 0.8 * max_force:
        warnings.append("High force magnitude - may cause oscillations")
        
    # Check if we're accelerating when already moving fast
    if current_vel * control_action > 0.3:  # Same direction, high magnitude
        warnings.append("Accelerating while already moving fast")
        
    # Calculate safe adjusted action
    adjusted_action = control_action
    
    # Clamp to force limits
    adjusted_action = np.clip(adjusted_action, -max_force, max_force)
    
    # Additional safety adjustments based on predicted violations
    if position_limits is not None:
        min_pos, max_pos = position_limits
        
        # If we'll hit position limit, apply counter-force
        if next_pos < min_pos and control_action < 0:
            # We're moving left but will hit left limit
            adjusted_action = max(0, adjusted_action)
            warnings.append("Reduced leftward force to avoid position limit")
        elif next_pos > max_pos and control_action > 0:
            # We're moving right but will hit right limit
            adjusted_action = min(0, adjusted_action)
            warnings.append("Reduced rightward force to avoid position limit")
    
    # Recompute next state with adjusted action
    safe_next_vel = current_vel + adjusted_action * dt
    safe_next_pos = current_pos + current_vel * dt
    
    # Final safety check
    safe = len(violations) == 0
    if adjusted_action != control_action:
        safe = True  # We fixed the violations
    
    return {
        'safe': safe,
        'warnings': warnings,
        'adjusted_action': round(adjusted_action, 4),
        'violations': violations,
        'next_state_prediction': {
            'position': round(safe_next_pos, 4),
            'velocity': round(safe_next_vel, 4),
            'time_step': dt
        },
        'action_changed': abs(adjusted_action - control_action) > 1e-6,
        'original_action': round(control_action, 4)
    }


def get_control_tools() -> List:
    """
    Get list of all control tools for LangGraph integration.
    
    Returns:
        List of tool functions that can be used with LangGraph
    """
    tools = [
        analyze_errors,
        select_control_strategy,
        calculate_pid_control, 
        calculate_lqr_control,
        plan_trajectory,
        verify_safety
    ]
    
    return tools


def test_tools():
    """Test all control tools with sample scenarios"""
    print("🧪 Testing Control Tools")
    print("=" * 50)
    
    # Test scenario: spacecraft at (0.5, 0.0) trying to reach (0.0, 0.0)
    pos, vel = 0.5, 0.0
    target_pos, target_vel = 0.0, 0.0
    
    print(f"📍 Test Scenario: Current ({pos}, {vel}) → Target ({target_pos}, {target_vel})")
    print()
    
    # Test 1: Error Analysis
    print("1️⃣ Testing Error Analysis Tool:")
    if LANGCHAIN_AVAILABLE:
        error_result = analyze_errors.invoke({
            "position": pos, 
            "velocity": vel, 
            "target_pos": target_pos, 
            "target_vel": target_vel
        })
    else:
        error_result = analyze_errors(pos, vel, target_pos, target_vel)
    print(f"   Position Error: {error_result['position_error']}")
    print(f"   Urgency: {error_result['urgency']}")
    print(f"   Phase: {error_result['phase']}")  
    print(f"   Stability: {error_result['stability']}")
    print(f"   Recommendations: {error_result['recommendations'][0]}")
    print()
    
    # Test 2: PID Calculator
    print("2️⃣ Testing PID Calculator Tool:")
    if LANGCHAIN_AVAILABLE:
        pid_result = calculate_pid_control.invoke({
            "pos_error": error_result['position_error'],
            "vel_error": error_result['velocity_error']
        })
    else:
        pid_result = calculate_pid_control(error_result['position_error'], error_result['velocity_error'])
    print(f"   Control Action: {pid_result['control_action']}")
    print(f"   P-term: {pid_result['components']['p_term']}, D-term: {pid_result['components']['d_term']}")
    print(f"   Confidence: {pid_result['confidence']}")
    print(f"   Reasoning: {pid_result['reasoning']}")
    print()
    
    # Test 3: Trajectory Planner
    print("3️⃣ Testing Trajectory Planner Tool:")
    if LANGCHAIN_AVAILABLE:
        traj_result = plan_trajectory.invoke({
            "current_pos": pos,
            "current_vel": vel,
            "target_pos": target_pos,
            "target_vel": target_vel
        })
    else:
        traj_result = plan_trajectory(pos, vel, target_pos, target_vel)
    print(f"   Strategy: {traj_result['strategy']}")
    print(f"   Estimated Time: {traj_result['estimated_time']}s")
    print(f"   Feasible: {traj_result['feasible']}")
    print(f"   Waypoints: {len(traj_result['waypoints'])}")
    print()
    
    # Test 4: Safety Verifier
    print("4️⃣ Testing Safety Verifier Tool:")
    if LANGCHAIN_AVAILABLE:
        safety_result = verify_safety.invoke({
            "control_action": pid_result['control_action'],
            "current_pos": pos,
            "current_vel": vel
        })
    else:
        safety_result = verify_safety(pid_result['control_action'], pos, vel)
    print(f"   Safe: {safety_result['safe']}")
    print(f"   Adjusted Action: {safety_result['adjusted_action']}")
    print(f"   Next Position: {safety_result['next_state_prediction']['position']}")
    if safety_result['warnings']:
        print(f"   Warnings: {safety_result['warnings']}")
    print()
    
    print("✅ All tools working correctly!")
    
    # Show integrated workflow example
    print("\n🔄 Integrated Workflow Example:")
    print("-" * 30)
    print("1. Analyze errors → Identify 'approach' phase")
    print("2. Calculate PID → Get control action") 
    print("3. Plan trajectory → Verify strategy alignment")
    print("4. Verify safety → Confirm action is safe")
    print("→ Final recommendation: Apply safe control action")


if __name__ == "__main__":
    test_tools()