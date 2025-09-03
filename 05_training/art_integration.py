"""
Core ART Integration for Control Systems
======================================

Adapts ART framework for control system training using patterns from:
- agentic_control_art/agents/rollout.py (working rollout pattern)
- agentic_control_art/training/grpo_trainer.py (training configuration)
- agentic_control_langgraph/agents/langgraph/art_rollout.py (tool integration)
"""

import asyncio
import json
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

# ART framework - ALREADY INSTALLED
try:
    import art
    from art import TrainableModel, Trajectory, TrajectoryGroup
    ART_AVAILABLE = True
    print("✅ ART library imported successfully")
except ImportError:
    print("❌ ART library not found. Install with: pip install openpipe-art")
    ART_AVAILABLE = False
    # Create mock classes for development
    class TrainableModel:
        def __init__(self, **kwargs):
            self.name = kwargs.get('model_name', 'mock_model')
        def openai_client(self):
            return None
        async def train(self, groups):
            print("Mock training (ART not available)")
    
    class Trajectory:
        def __init__(self, messages, reward, metadata=None):
            self.messages = messages
            self.reward = reward
            self.metadata = metadata or {}
    
    class TrajectoryGroup:
        def __init__(self, trajectories):
            self.trajectories = trajectories

# Import your existing components
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '01_basic_physics'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '02_direct_control'))

from double_integrator import DoubleIntegrator


@dataclass
class ControlScenario:
    """Control scenario for training"""
    name: str
    initial_position: float
    initial_velocity: float
    target_position: float = 0.0
    target_velocity: float = 0.0
    max_steps: int = 50
    difficulty: str = "medium"
    
    def __str__(self):
        return f"{self.name}: ({self.initial_position:.2f}, {self.initial_velocity:.2f}) → ({self.target_position:.2f}, {self.target_velocity:.2f})"


class ARTControlTrainer:
    """Main ART training coordinator for control systems"""
    
    def __init__(self, model_name: str = "double-integrator-agent"):
        self.model_name = model_name
        self.model = None
        self.training_stats = {
            'total_trajectories': 0,
            'successful_trajectories': 0,
            'average_reward': 0.0,
            'training_episodes': 0
        }
        
    async def initialize_model(self):
        """Initialize ART model (uses existing vLLM setup)"""
        if not ART_AVAILABLE:
            print("⚠️ ART not available - using mock model")
            self.model = TrainableModel(model_name=self.model_name)
            return
            
        self.model = TrainableModel(
            model_name=self.model_name,
            base_model="Qwen/Qwen2.5-1.5B-Instruct"
        )
        print(f"✅ ART model initialized: {self.model_name}")
        
    def create_training_scenarios(self) -> List[ControlScenario]:
        """Create scenarios for training (matching your existing test scenarios)"""
        scenarios = []
        
        # Easy scenarios - close to target
        scenarios.extend([
            ControlScenario("easy_pos_small", 0.2, 0.0, difficulty="easy"),
            ControlScenario("easy_pos_neg", -0.15, 0.0, difficulty="easy"),
            ControlScenario("easy_vel_small", 0.0, 0.1, difficulty="easy"),
            ControlScenario("easy_vel_neg", 0.0, -0.12, difficulty="easy"),
        ])
        
        # Medium scenarios - moderate distance/velocity
        scenarios.extend([
            ControlScenario("med_pos_1", 0.5, 0.0, difficulty="medium"),
            ControlScenario("med_pos_2", -0.4, 0.0, difficulty="medium"),
            ControlScenario("med_vel_1", 0.0, 0.3, difficulty="medium"),
            ControlScenario("med_mixed_1", 0.3, 0.2, difficulty="medium"),
            ControlScenario("med_mixed_2", -0.2, -0.25, difficulty="medium"),
        ])
        
        # Hard scenarios - far from target or high velocity
        scenarios.extend([
            ControlScenario("hard_pos_far", 1.0, 0.0, difficulty="hard"),
            ControlScenario("hard_vel_high", 0.0, 0.6, difficulty="hard"),
            ControlScenario("hard_mixed_1", 0.8, -0.4, difficulty="hard"),
            ControlScenario("hard_mixed_2", -0.7, 0.5, difficulty="hard"),
        ])
        
        return scenarios

    def calculate_control_reward(self, env: DoubleIntegrator, scenario: ControlScenario, 
                               success: bool, step_count: int) -> float:
        """Calculate multi-objective control reward"""
        final_pos, final_vel = env.get_state()
        pos_error = abs(final_pos - scenario.target_position)
        vel_error = abs(final_vel - scenario.target_velocity)
        control_effort = sum(abs(u) for u in env.history['control'])
        
        # Multi-objective reward components
        reward = 0.0
        
        # Success bonus (most important)
        if success:
            reward += 10.0
        
        # Position accuracy (high weight)
        reward -= pos_error * 5.0
        
        # Velocity accuracy (medium weight)
        reward -= vel_error * 3.0
        
        # Control efficiency (low weight)
        reward -= control_effort * 0.1
        
        # Time penalty (encourage faster solutions)
        reward -= step_count * 0.05
        
        # Difficulty bonus
        difficulty_bonuses = {"easy": 0.0, "medium": 1.0, "hard": 2.0}
        reward += difficulty_bonuses.get(scenario.difficulty, 0.0)
        
        return reward


async def direct_control_rollout(model: TrainableModel, scenario: ControlScenario) -> Trajectory:
    """
    Direct LLM control rollout (adapted from agentic_control_art/agents/rollout.py)
    
    This creates ART trajectories from control episodes
    """
    # Initialize environment
    env = DoubleIntegrator(max_force=1.0, dt=0.1)
    env.reset(scenario.initial_position, scenario.initial_velocity)
    
    # System prompt for the agent
    system_prompt = f"""You are a spacecraft control agent. Your task is to control a double integrator system.

SYSTEM DYNAMICS: ẍ = u (acceleration equals control force)
CURRENT MISSION: {scenario.name}
GOAL: Move from ({scenario.initial_position:.3f}, {scenario.initial_velocity:.3f}) to ({scenario.target_position:.3f}, {scenario.target_velocity:.3f})

PHYSICS CONSTRAINTS:
- Control force must be between -1.0 and +1.0 N
- System has no friction or damping
- Position changes based on velocity: x(t+1) = x(t) + v(t)*dt
- Velocity changes based on control: v(t+1) = v(t) + u*dt

CONTROL OBJECTIVES:
- Reach target position with zero velocity
- Minimize control effort and time to target
- Avoid oscillation and overshoot

OUTPUT FORMAT (JSON only):
{{"action": <control_value>, "reasoning": "<your_reasoning>", "confidence": <0_to_1>}}"""

    # Initialize trajectory messages
    messages_and_choices = [
        {"role": "system", "content": system_prompt}
    ]
    
    # Control loop
    success = False
    step_count = 0
    
    for step in range(scenario.max_steps):
        # Get current state observation
        position, velocity = env.get_state()
        pos_error = scenario.target_position - position
        vel_error = scenario.target_velocity - velocity
        
        observation = f"""Step {step}: 
Current State: Position = {position:.3f} m, Velocity = {velocity:.3f} m/s
Target State: Position = {scenario.target_position:.3f} m, Velocity = {scenario.target_velocity:.3f} m/s
Errors: Position Error = {pos_error:.3f} m, Velocity Error = {vel_error:.3f} m/s
Time Remaining: {scenario.max_steps - step} steps"""
        
        messages_and_choices.append({"role": "user", "content": observation})
        
        # Get model response (ART handles this automatically)
        if ART_AVAILABLE and hasattr(model, 'openai_client'):
            try:
                client = model.openai_client()
                response = await client.chat.completions.create(
                    model=model.name,
                    messages=[
                        {"role": "system", "content": system_prompt}, 
                        {"role": "user", "content": observation}
                    ],
                    max_tokens=200,
                    temperature=0.3
                )
                agent_response = response.choices[0].message.content
            except Exception as e:
                print(f"⚠️ Model call failed: {e}, using fallback")
                # Fallback PD control response
                pd_action = max(-1.0, min(1.0, pos_error * 1.0 + vel_error * 2.0))
                agent_response = json.dumps({
                    "action": pd_action,
                    "reasoning": f"Fallback PD control: kp*pos_error + kd*vel_error = {pd_action:.3f}",
                    "confidence": 0.5
                })
        else:
            # Mock response for development
            pd_action = max(-1.0, min(1.0, pos_error * 0.8 + vel_error * 1.5))
            agent_response = json.dumps({
                "action": pd_action,
                "reasoning": f"Mock PD control with noise: error-based action = {pd_action:.3f}",
                "confidence": 0.7
            })
        
        messages_and_choices.append({"role": "assistant", "content": agent_response})
        
        # Parse action (with error handling)
        try:
            response_data = json.loads(agent_response.strip())
            action = float(response_data.get('action', 0.0))
            reasoning = response_data.get('reasoning', 'No reasoning provided')
        except (json.JSONDecodeError, ValueError):
            # Fallback parsing for malformed JSON
            action = 0.0
            reasoning = "Failed to parse response"
            
        # Apply control to environment
        env.step(action)
        step_count += 1
        
        # Check success condition
        if env.is_at_target(scenario.target_position, scenario.target_velocity, 
                           pos_tol=0.1, vel_tol=0.1):
            success = True
            break
    
    # Calculate reward using the trainer's reward function
    trainer = ARTControlTrainer()  # Temporary instance for reward calculation
    reward = trainer.calculate_control_reward(env, scenario, success, step_count)
    
    # Create trajectory metadata
    final_pos, final_vel = env.get_state()
    metadata = {
        'scenario': scenario.name,
        'approach': 'direct_control',
        'success': success,
        'steps': step_count,
        'final_position': final_pos,
        'final_velocity': final_vel,
        'final_pos_error': abs(final_pos - scenario.target_position),
        'final_vel_error': abs(final_vel - scenario.target_velocity),
        'control_effort': sum(abs(u) for u in env.history['control']),
        'difficulty': scenario.difficulty,
        'reward_components': {
            'success_bonus': 10.0 if success else 0.0,
            'position_penalty': -abs(final_pos - scenario.target_position) * 5.0,
            'velocity_penalty': -abs(final_vel - scenario.target_velocity) * 3.0,
            'control_penalty': -sum(abs(u) for u in env.history['control']) * 0.1,
            'time_penalty': -step_count * 0.05
        }
    }
    
    return Trajectory(
        messages_and_choices=messages_and_choices,
        reward=reward,
        metadata=metadata
    )


async def tool_augmented_rollout(model: TrainableModel, scenario: ControlScenario) -> Trajectory:
    """
    Tool-augmented control rollout (simulates using physics tools)
    
    In a full implementation, this would use LangGraph tools from 03_langgraph_tools/
    For now, we simulate tool usage to demonstrate the concept
    """
    # Initialize environment
    env = DoubleIntegrator(max_force=1.0, dt=0.1)
    env.reset(scenario.initial_position, scenario.initial_velocity)
    
    # System prompt with tool descriptions
    system_prompt = f"""You are a spacecraft control agent with access to advanced physics tools.

MISSION: {scenario.name}
GOAL: Move from ({scenario.initial_position:.3f}, {scenario.initial_velocity:.3f}) to ({scenario.target_position:.3f}, {scenario.target_velocity:.3f})

AVAILABLE TOOLS:
1. analyze_errors(pos, vel, target_pos, target_vel): Detailed error analysis
2. calculate_pid(pos_error, vel_error, kp, kd): PID control calculation
3. plan_trajectory(current, target, time_horizon): Optimal trajectory planning
4. verify_safety(action, current_state): Safety verification

TOOL USAGE PROTOCOL:
1. Always use analyze_errors first to understand the situation
2. Use calculate_pid or plan_trajectory to determine control action
3. Use verify_safety before executing any control action
4. Provide reasoning for tool choices and final decisions

OUTPUT FORMAT: Use tools, then provide final JSON:
{{"action": <control_value>, "reasoning": "<tool-based_reasoning>", "confidence": <0_to_1>}}"""

    messages_and_choices = [
        {"role": "system", "content": system_prompt}
    ]
    
    success = False
    step_count = 0
    tool_usage_count = 0
    
    for step in range(scenario.max_steps):
        position, velocity = env.get_state()
        pos_error = scenario.target_position - position
        vel_error = scenario.target_velocity - velocity
        
        observation = f"""Step {step}:
Current State: Position = {position:.3f} m, Velocity = {velocity:.3f} m/s
Target State: Position = {scenario.target_position:.3f} m, Velocity = {scenario.target_velocity:.3f} m/s
Available tools: analyze_errors, calculate_pid, plan_trajectory, verify_safety"""
        
        # Simulate comprehensive tool usage
        error_magnitude = np.sqrt(pos_error**2 + vel_error**2)
        urgency = "HIGH" if error_magnitude > 0.5 else "MEDIUM" if error_magnitude > 0.2 else "LOW"
        
        # Tool 1: Error Analysis
        error_analysis = f"Error Analysis → Position error: {pos_error:.3f} m, Velocity error: {vel_error:.3f} m/s, Total error: {error_magnitude:.3f}, Urgency: {urgency}, System stability: STABLE"
        
        # Tool 2: PID Calculation (optimized gains)
        kp, kd = (1.2, 2.5) if urgency == "HIGH" else (1.0, 2.0) if urgency == "MEDIUM" else (0.8, 1.8)
        pid_action = pos_error * kp + vel_error * kd
        pid_result = f"PID Control → Using gains kp={kp}, kd={kd}, Raw action: {pid_action:.3f}"
        
        # Tool 3: Safety Verification
        clipped_action = max(-1.0, min(1.0, pid_action))
        safety_status = "SAFE" if abs(pid_action) <= 1.0 else f"CLIPPED from {pid_action:.3f} to {clipped_action:.3f}"
        safety_result = f"Safety Check → Action {clipped_action:.3f} is {safety_status}"
        
        # Tool 4: Trajectory Planning (optional for complex cases)
        if error_magnitude > 0.7:
            trajectory_plan = f"Trajectory Planning → Multi-step approach recommended: Step 1: Reduce velocity, Step 2: Correct position"
            tool_usage_count += 4
        else:
            trajectory_plan = "Trajectory Planning → Direct approach sufficient"
            tool_usage_count += 3
        
        # Create detailed tool-using conversation
        messages_and_choices.extend([
            {"role": "user", "content": observation},
            {"role": "assistant", "content": f"I'll analyze this step systematically using my tools.\n\n1. analyze_errors({position:.3f}, {velocity:.3f}, {scenario.target_position:.3f}, {scenario.target_velocity:.3f})"},
            {"role": "user", "content": f"Tool Result: {error_analysis}"},
            {"role": "assistant", "content": f"Based on {urgency} urgency, I'll calculate control action.\n\n2. calculate_pid({pos_error:.3f}, {vel_error:.3f}, {kp}, {kd})"},
            {"role": "user", "content": f"Tool Result: {pid_result}"},
            {"role": "assistant", "content": f"Now I'll verify this action is safe.\n\n3. verify_safety({pid_action:.3f}, [{position:.3f}, {velocity:.3f}])"},
            {"role": "user", "content": f"Tool Result: {safety_result}"},
        ])
        
        if error_magnitude > 0.7:
            messages_and_choices.extend([
                {"role": "assistant", "content": f"Given large error, checking trajectory planning.\n\n4. plan_trajectory([{position:.3f}, {velocity:.3f}], [{scenario.target_position:.3f}, {scenario.target_velocity:.3f}], {scenario.max_steps - step})"},
                {"role": "user", "content": f"Tool Result: {trajectory_plan}"}
            ])
        
        # Final reasoning and action
        final_reasoning = f"Tool Analysis Summary: {urgency} urgency situation. Error analysis shows {error_magnitude:.3f} total error. PID calculation recommends {pid_action:.3f} force. Safety verification confirms {clipped_action:.3f} is safe. Executing physics-informed control action."
        
        final_response = json.dumps({
            "action": clipped_action,
            "reasoning": final_reasoning,
            "confidence": 0.9 if safety_status == "SAFE" else 0.7
        })
        
        messages_and_choices.append({"role": "assistant", "content": final_response})
        
        # Apply the tool-calculated control
        env.step(clipped_action)
        step_count += 1
        
        # Check success
        if env.is_at_target(scenario.target_position, scenario.target_velocity,
                           pos_tol=0.1, vel_tol=0.1):
            success = True
            break
    
    # Calculate reward with tool bonus
    trainer = ARTControlTrainer()
    base_reward = trainer.calculate_control_reward(env, scenario, success, step_count)
    
    # Add bonuses for tool usage
    tool_bonus = 2.0  # Reward for interpretable reasoning
    efficiency_bonus = 1.0 if success and step_count < scenario.max_steps * 0.8 else 0.0
    total_reward = base_reward + tool_bonus + efficiency_bonus
    
    # Enhanced metadata
    final_pos, final_vel = env.get_state()
    metadata = {
        'scenario': scenario.name,
        'approach': 'tool_augmented',
        'success': success,
        'steps': step_count,
        'final_position': final_pos,
        'final_velocity': final_vel,
        'final_pos_error': abs(final_pos - scenario.target_position),
        'final_vel_error': abs(final_vel - scenario.target_velocity),
        'control_effort': sum(abs(u) for u in env.history['control']),
        'difficulty': scenario.difficulty,
        'tool_usage_count': tool_usage_count,
        'interpretable_reasoning': True,
        'tool_bonus': tool_bonus,
        'efficiency_bonus': efficiency_bonus,
        'base_reward': base_reward,
        'total_reward': total_reward
    }
    
    return Trajectory(
        messages_and_choices=messages_and_choices,
        reward=total_reward,
        metadata=metadata
    )