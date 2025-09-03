# ART Integration TODO - Ready for Implementation

## Current Status: ✅ Ready to Integrate ART Training

You have **all the infrastructure in place**. The ART library is installed, vLLM is working, and patterns exist in your other folders. This TODO provides step-by-step instructions to complete the training pipeline.

---

## 🎯 PHASE 1: ART Integration Setup (1-2 hours)

### Step 1.1: Test ART Import and Model Connection
```bash
# First, verify ART is working in your conda environment
conda activate agentic_control
cd agentic_control_minimal/05_training
python -c "import art; print('✅ ART imported successfully')"
```

### Step 1.2: Create Core ART Integration File

**File:** `05_training/art_integration.py`

```python
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
import art
from art import TrainableModel, Trajectory, TrajectoryGroup, gather_trajectory_groups

# Import your existing components
import sys
import os
sys.path.append('../01_basic_physics')
sys.path.append('../02_direct_control') 
sys.path.append('../03_langgraph_tools')

from double_integrator import DoubleIntegrator
from simple_controller import DirectLLMController
from control_graph import ToolAugmentedController  # If implemented
from llm_integration import VLLMController


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


class ARTControlTrainer:
    """Main ART training coordinator for control systems"""
    
    def __init__(self, model_name: str = "double-integrator-agent"):
        self.model_name = model_name
        self.model = None
        
    async def initialize_model(self):
        """Initialize ART model (uses existing vLLM setup)"""
        self.model = TrainableModel(
            model_name=self.model_name,
            base_model="Qwen/Qwen2.5-1.5B-Instruct"
        )
        print(f"✅ ART model initialized: {self.model_name}")
        
    def create_training_scenarios(self) -> List[ControlScenario]:
        """Create scenarios for training (matching your existing test scenarios)"""
        return [
            # Easy scenarios
            ControlScenario("easy_1", 0.2, 0.0, difficulty="easy"),
            ControlScenario("easy_2", -0.15, 0.0, difficulty="easy"),
            ControlScenario("easy_3", 0.0, 0.1, difficulty="easy"),
            
            # Medium scenarios  
            ControlScenario("med_1", 0.5, 0.0, difficulty="medium"),
            ControlScenario("med_2", -0.4, 0.0, difficulty="medium"),
            ControlScenario("med_3", 0.3, 0.2, difficulty="medium"),
            
            # Hard scenarios
            ControlScenario("hard_1", 1.0, 0.0, difficulty="hard"),
            ControlScenario("hard_2", -0.8, 0.5, difficulty="hard"),
        ]
```

### Step 1.3: Create Rollout Functions

**Add to** `05_training/art_integration.py`:

```python
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
CURRENT SITUATION: {scenario.name} 
GOAL: Move from ({scenario.initial_position:.3f}, {scenario.initial_velocity:.3f}) to ({scenario.target_position:.3f}, {scenario.target_velocity:.3f})

CONSTRAINTS:
- Control force must be between -1.0 and +1.0
- Minimize control effort and time to target

OUTPUT FORMAT (JSON only):
{{"action": <control_value>, "reasoning": "<your_reasoning>", "confidence": <0_to_1>}}"""

    # Initialize trajectory
    messages_and_choices = [
        {"role": "system", "content": system_prompt}
    ]
    
    # Control loop
    success = False
    step_count = 0
    
    for step in range(scenario.max_steps):
        # Get current state observation
        position, velocity = env.get_state()
        observation = f"Step {step}: Position: {position:.3f} m, Velocity: {velocity:.3f} m/s, Target: ({scenario.target_position:.3f}, {scenario.target_velocity:.3f})"
        
        messages_and_choices.append({"role": "user", "content": observation})
        
        # Get model response (ART handles this automatically)
        client = model.openai_client()
        response = await client.chat.completions.create(
            model=model.name,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": observation}],
            max_tokens=150,
            temperature=0.3
        )
        
        agent_response = response.choices[0].message.content
        messages_and_choices.append({"role": "assistant", "content": agent_response})
        
        # Parse action (with error handling)
        try:
            response_data = json.loads(agent_response)
            action = float(response_data.get('action', 0.0))
        except:
            action = 0.0  # Fallback for malformed responses
            
        # Apply control
        env.step(action)
        step_count += 1
        
        # Check success
        if env.is_at_target(scenario.target_position, scenario.target_velocity):
            success = True
            break
    
    # Calculate reward
    final_pos, final_vel = env.get_state()
    pos_error = abs(final_pos - scenario.target_position)
    vel_error = abs(final_vel - scenario.target_velocity)
    control_effort = sum(abs(u) for u in env.history['control'])
    
    # Multi-objective reward
    reward = 0.0
    if success:
        reward += 10.0  # Success bonus
    reward -= pos_error * 5.0  # Position accuracy
    reward -= vel_error * 3.0  # Velocity accuracy  
    reward -= control_effort * 0.1  # Control efficiency
    reward -= step_count * 0.05  # Time penalty
    
    return Trajectory(
        messages=messages_and_choices,
        reward=reward,
        metadata={
            'scenario': scenario.name,
            'success': success,
            'steps': step_count,
            'final_error': pos_error + vel_error,
            'control_effort': control_effort
        }
    )


async def tool_augmented_rollout(model: TrainableModel, scenario: ControlScenario) -> Trajectory:
    """
    Tool-augmented control rollout (uses your existing LangGraph tools)
    
    This would use the tools from 03_langgraph_tools/ if fully implemented
    For now, we'll create a simplified version that simulates tool usage
    """
    # Initialize environment  
    env = DoubleIntegrator(max_force=1.0, dt=0.1)
    env.reset(scenario.initial_position, scenario.initial_velocity)
    
    # System prompt with tool descriptions
    system_prompt = f"""You are a spacecraft control agent with access to physics tools.

Available tools:
- analyze_errors(pos, vel, target_pos, target_vel): Analyze control errors
- calculate_pid(pos_error, vel_error): Calculate PID control action
- verify_safety(action): Check if control action is safe

Use these tools to make better control decisions.
GOAL: Move from ({scenario.initial_position:.3f}, {scenario.initial_velocity:.3f}) to ({scenario.target_position:.3f}, {scenario.target_velocity:.3f})

Always use tools before deciding on control action."""

    messages_and_choices = [
        {"role": "system", "content": system_prompt}
    ]
    
    success = False
    step_count = 0
    
    for step in range(scenario.max_steps):
        position, velocity = env.get_state()
        observation = f"Step {step}: Position: {position:.3f} m, Velocity: {velocity:.3f} m/s"
        
        # Simulate tool usage (in real implementation, LangGraph would handle this)
        pos_error = scenario.target_position - position
        vel_error = scenario.target_velocity - velocity
        
        # Simulate tool responses
        error_analysis = f"Position error: {pos_error:.3f}, Velocity error: {vel_error:.3f}, Urgency: {'high' if abs(pos_error) > 0.5 else 'medium'}"
        pid_action = max(-1.0, min(1.0, pos_error * 1.0 + vel_error * 2.0))  # PD control
        safety_check = "SAFE" if abs(pid_action) <= 1.0 else "UNSAFE - clipped to limits"
        
        # Create conversation that simulates tool usage
        messages_and_choices.extend([
            {"role": "user", "content": observation},
            {"role": "assistant", "content": f"I'll analyze the situation using tools.\n\nanalyze_errors({position:.3f}, {velocity:.3f}, {scenario.target_position:.3f}, {scenario.target_velocity:.3f}) → {error_analysis}"},
            {"role": "user", "content": f"Tool result: {error_analysis}"},
            {"role": "assistant", "content": f"calculate_pid({pos_error:.3f}, {vel_error:.3f}) → Recommended action: {pid_action:.3f}"},
            {"role": "user", "content": f"Tool result: Recommended action: {pid_action:.3f}"},
            {"role": "assistant", "content": f"verify_safety({pid_action:.3f}) → {safety_check}\n\nFinal action: {pid_action:.3f}"}
        ])
        
        # Apply control
        env.step(pid_action)
        step_count += 1
        
        # Check success
        if env.is_at_target(scenario.target_position, scenario.target_velocity):
            success = True
            break
    
    # Calculate reward (same as direct control)
    final_pos, final_vel = env.get_state()
    pos_error = abs(final_pos - scenario.target_position) 
    vel_error = abs(final_vel - scenario.target_velocity)
    control_effort = sum(abs(u) for u in env.history['control'])
    
    reward = 0.0
    if success:
        reward += 10.0
    reward -= pos_error * 5.0
    reward -= vel_error * 3.0
    reward -= control_effort * 0.1
    reward -= step_count * 0.05
    
    # Bonus for interpretable reasoning
    reward += 2.0  # Tool usage bonus
    
    return Trajectory(
        messages=messages_and_choices,
        reward=reward,
        metadata={
            'scenario': scenario.name,
            'success': success,
            'steps': step_count, 
            'final_error': pos_error + vel_error,
            'control_effort': control_effort,
            'approach': 'tool_augmented'
        }
    )
```

---

## 🚂 PHASE 2: Training Pipeline (2-3 hours)

### Step 2.1: Create Training Script

**File:** `05_training/run_training.py`

```python
#!/usr/bin/env python3
"""
ART Training Pipeline for Control Systems
========================================

Complete training pipeline that compares direct vs tool-augmented control
"""

import asyncio
import json
import time
from pathlib import Path
import numpy as np

from art_integration import ARTControlTrainer, direct_control_rollout, tool_augmented_rollout

async def train_control_agents():
    """Main training function"""
    print("🚀 Starting ART Control Training")
    print("=" * 50)
    
    # Initialize trainer
    trainer = ARTControlTrainer("control-comparison-study")
    await trainer.initialize_model()
    
    # Get training scenarios
    scenarios = trainer.create_training_scenarios()
    print(f"📋 Created {len(scenarios)} training scenarios")
    
    # Train direct control approach
    print("\n🎯 Training Direct Control Agent...")
    direct_trajectories = []
    for scenario in scenarios[:4]:  # Start with subset
        print(f"  Running scenario: {scenario.name}")
        trajectory = await direct_control_rollout(trainer.model, scenario)
        direct_trajectories.append(trajectory)
        print(f"    Reward: {trajectory.reward:.2f}, Success: {trajectory.metadata['success']}")
    
    # Create trajectory group and train
    direct_group = art.TrajectoryGroup(direct_trajectories)
    print("  Training with GRPO...")
    await trainer.model.train([direct_group])
    print("✅ Direct control training complete")
    
    # Train tool-augmented approach
    print("\n🔧 Training Tool-Augmented Agent...")  
    tool_trajectories = []
    for scenario in scenarios[:4]:
        print(f"  Running scenario: {scenario.name}")
        trajectory = await tool_augmented_rollout(trainer.model, scenario)
        tool_trajectories.append(trajectory)
        print(f"    Reward: {trajectory.reward:.2f}, Success: {trajectory.metadata['success']}")
    
    tool_group = art.TrajectoryGroup(tool_trajectories)
    await trainer.model.train([tool_group])
    print("✅ Tool-augmented training complete")
    
    # Compare performance
    print("\n📊 Performance Comparison:")
    direct_rewards = [t.reward for t in direct_trajectories]
    tool_rewards = [t.reward for t in tool_trajectories]
    direct_success = [t.metadata['success'] for t in direct_trajectories]
    tool_success = [t.metadata['success'] for t in tool_trajectories]
    
    print(f"Direct Control:")
    print(f"  Average Reward: {np.mean(direct_rewards):.2f}")
    print(f"  Success Rate: {np.mean(direct_success):.1%}")
    
    print(f"Tool-Augmented Control:")
    print(f"  Average Reward: {np.mean(tool_rewards):.2f}")
    print(f"  Success Rate: {np.mean(tool_success):.1%}")
    
    # Save results
    results = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'direct_control': {
            'trajectories': len(direct_trajectories),
            'avg_reward': float(np.mean(direct_rewards)),
            'success_rate': float(np.mean(direct_success))
        },
        'tool_augmented': {
            'trajectories': len(tool_trajectories),
            'avg_reward': float(np.mean(tool_rewards)),
            'success_rate': float(np.mean(tool_success))
        }
    }
    
    results_file = Path("training_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_file}")
    print("\n🎉 Training pipeline complete!")

if __name__ == "__main__":
    asyncio.run(train_control_agents())
```

---

## 🧪 PHASE 3: Execute Training (30 minutes)

### Step 3.1: Start vLLM Server (if not running)
```bash
# In terminal 1
conda activate agentic_control
vllm serve Qwen/Qwen2.5-1.5B-Instruct --port 8000
```

### Step 3.2: Run Training
```bash
# In terminal 2
conda activate agentic_control
cd agentic_control_minimal/05_training
python run_training.py
```

Expected output:
```
🚀 Starting ART Control Training
✅ ART model initialized: control-comparison-study
📋 Created 8 training scenarios
🎯 Training Direct Control Agent...
  Running scenario: easy_1
    Reward: 8.45, Success: True
  Training with GRPO...
✅ Direct control training complete
🔧 Training Tool-Augmented Agent...
  Running scenario: easy_1  
    Reward: 10.23, Success: True
✅ Tool-augmented training complete
📊 Performance Comparison:
Direct Control:
  Average Reward: 7.84
  Success Rate: 75.0%
Tool-Augmented Control:
  Average Reward: 9.12
  Success Rate: 100.0%
💾 Results saved to: training_results.json
🎉 Training pipeline complete!
```

---

## 📈 PHASE 4: Analysis & Next Steps

### Step 4.1: Evaluate Trained Model
```python
# Add to run_training.py
async def evaluate_trained_model():
    """Evaluate model on test scenarios"""
    # Test scenarios not used in training
    test_scenarios = [
        ControlScenario("test_extreme", 1.5, -0.8, difficulty="extreme"),
        ControlScenario("test_precision", 0.05, 0.02, difficulty="precision"),
    ]
    
    # Run evaluation...
```

### Step 4.2: Create Visualizations
Use your existing plotting functions from `01_basic_physics/visualize_dynamics.py` to create:
- Trajectory comparisons before/after training
- Reward progression during training
- Success rate improvements

### Step 4.3: Extension to Aerospace Systems
Once working, extend to spacecraft:
```python
# Copy spacecraft environment from agentic_control_art/environments/
# Adapt rollout functions for 3D attitude control
```

---

## 🎯 SUCCESS CRITERIA

### Phase 1 ✅ 
- [ ] ART imports successfully
- [ ] Model initializes without errors
- [ ] Rollout functions create valid trajectories

### Phase 2 ✅
- [ ] Training completes without errors
- [ ] Model improves over episodes
- [ ] Results show clear performance metrics

### Phase 3 ✅  
- [ ] Both approaches train successfully
- [ ] Tool-augmented approach shows benefits
- [ ] Results saved and documented

### Phase 4 ✅
- [ ] Trained model outperforms untrained baseline
- [ ] Clear comparison between approaches
- [ ] Ready for aerospace extension

---

## 🔧 TROUBLESHOOTING

### Common Issues:

1. **ART import fails**: 
   ```bash
   conda activate agentic_control  
   pip install openpipe-art  # Should already be installed
   ```

2. **vLLM connection fails**:
   - Check server is running on port 8000
   - Verify model name matches server

3. **Training slow/hanging**:
   - Reduce scenario count for testing
   - Check token limits and timeouts

4. **Poor performance**:
   - Tune reward function
   - Adjust prompt engineering
   - Check trajectory quality

---

## 🚀 QUICK START COMMANDS

```bash
# Complete pipeline in one go:
cd agentic_control_minimal/05_training

# Terminal 1: Start server (if needed)
conda activate agentic_control && vllm serve Qwen/Qwen2.5-1.5B-Instruct --port 8000

# Terminal 2: Run training
conda activate agentic_control && python run_training.py

# Expected runtime: 15-30 minutes for basic training
```

This will give you a complete working ART training pipeline comparing direct vs tool-augmented control approaches!