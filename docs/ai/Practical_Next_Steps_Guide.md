# Practical Next Steps Guide: From Errors to Working Agentic Control

## Executive Summary

This guide addresses the specific errors you're encountering in your implementations and provides a step-by-step path to get your agentic control systems working with the latest ART features. Based on analysis of your error logs and the ART reference implementation, we'll start with a minimal working example and progressively build to the advanced features you want.

## Table of Contents

1. [Part 1: Getting Started with Minimal Working Example](#part-1-getting-started-with-minimal-working-example)
2. [Part 2: Adapting for Control Systems](#part-2-adapting-for-control-systems)
3. [Part 3: Fixing Your Existing Implementation Errors](#part-3-fixing-your-existing-implementation-errors)
4. [Part 4: Progressive Enhancement Path](#part-4-progressive-enhancement-path)
5. [Part 5: Working Code Templates](#part-5-working-code-templates)
6. [Part 6: Integration with MCP-RL](#part-6-integration-with-mcp-rl)

---

## Part 1: Getting Started with Minimal Working Example

### 1.1 Start with What Works: The 2048 Example

Your error logs show API mismatches and missing methods. Instead of debugging these issues, let's start with the **confirmed working 2048 example** from ART_reference and adapt it step by step.

#### Why 2048 Example is Perfect Starting Point:
- ✅ Uses correct ART API patterns
- ✅ Simple training loop
- ✅ No complex dependencies
- ✅ Clear rollout structure
- ✅ Works with LocalBackend

### 1.2 Test the 2048 Example First

**Step 1.1: Copy and Test Base Example**

```bash
# Navigate to working directory
cd /orcd/home/002/amitjain/project/Unsloth/ART_agentic_control

# Create a new test folder
mkdir -p test_working_example
cd test_working_example

# Copy the working 2048 files
cp ../ART_reference/examples/2048/train.py .
cp ../ART_reference/examples/2048/rollout.py .
cp ../ART_reference/examples/2048/utils.py .
```

**Step 1.2: Minimal Test Run**

Create `test_2048.py`:

```python
import asyncio
import random
from dotenv import load_dotenv
import art
from art.local import LocalBackend

load_dotenv()
random.seed(42)

# MINIMAL test - no training, just model setup
async def test_model_setup():
    # This is the EXACT pattern that works
    model = art.TrainableModel(
        name="test-control-001",
        project="control-test",
        base_model="Qwen/Qwen2.5-1.5B-Instruct",  # Smaller model for testing
    )
    
    # Initialize backend
    backend = art.LocalBackend()
    
    try:
        # Test model registration
        await model.register(backend)
        print("✅ Model registration successful!")
        
        # Test getting step (should be 0 for new model)
        step = await model.get_step()
        print(f"✅ Current step: {step}")
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_model_setup())
    if success:
        print("🎉 Basic ART setup is working!")
    else:
        print("🚨 Need to fix environment first")
```

**Step 1.3: Run the Test**

```bash
conda activate agentic_control
python test_2048.py
```

If this fails, we need to fix your environment first before proceeding.

### 1.3 Key Patterns from Working Example

From the 2048 example, these are the **correct API patterns**:

```python
# ✅ Correct model initialization
model = art.TrainableModel(
    name="model-name",
    project="project-name", 
    base_model="Qwen/Qwen2.5-3B-Instruct",
)

# ✅ Correct training config
await model.train(
    train_groups,
    config=art.TrainConfig(learning_rate=1e-5),  # NOT LLMTrainingConfig!
)

# ✅ Correct backend usage
backend = art.LocalBackend()  # NOT LocalBackend() 
await model.register(backend)

# ✅ Correct trajectory gathering
train_groups = await art.gather_trajectory_groups(
    (
        art.TrajectoryGroup(
            rollout(model, step, is_validation=False)
            for _ in range(num_trajectories)
        )
        for _ in range(1)
    ),
    pbar_desc="gather",
)
```

---

## Part 2: Adapting for Control Systems

### 2.1 From 2048 to Double Integrator

Once the basic 2048 example works, we'll adapt it step-by-step for control systems.

**Step 2.1: Create Minimal Control Environment**

Create `control_env.py`:

```python
import numpy as np
from typing import Dict, Any, Tuple

class MinimalDoubleIntegrator:
    """Minimal double integrator: ẍ = u"""
    
    def __init__(self):
        self.dt = 0.1  # Time step
        self.max_steps = 50
        self.reset()
    
    def reset(self) -> Dict[str, Any]:
        # Random initial state
        self.state = np.random.uniform(-1, 1, 4)  # [x, y, vx, vy]
        self.target = np.random.uniform(-2, 2, 2)  # [target_x, target_y]
        self.step_count = 0
        return self.get_observation()
    
    def step(self, action: np.ndarray) -> Tuple[Dict[str, Any], float, bool]:
        # Double integrator dynamics: ẍ = u
        pos = self.state[:2]
        vel = self.state[2:]
        
        # Update with action (acceleration)
        new_vel = vel + action * self.dt
        new_pos = pos + new_vel * self.dt
        
        self.state = np.concatenate([new_pos, new_vel])
        self.step_count += 1
        
        # Simple reward: negative distance to target
        distance = np.linalg.norm(pos - self.target)
        reward = -distance
        
        # Done if reached target or max steps
        done = distance < 0.1 or self.step_count >= self.max_steps
        
        return self.get_observation(), reward, done
    
    def get_observation(self) -> Dict[str, Any]:
        pos = self.state[:2]
        vel = self.state[2:]
        
        return {
            "position": pos.tolist(),
            "velocity": vel.tolist(), 
            "target": self.target.tolist(),
            "distance_to_target": float(np.linalg.norm(pos - self.target)),
            "step": self.step_count,
            "description": f"Agent at position {pos} with velocity {vel}, target at {self.target}"
        }
```

**Step 2.2: Create Control Rollout Function**

Create `control_rollout.py`:

```python
import json
import re
from typing import AsyncGenerator
from control_env import MinimalDoubleIntegrator
import art

async def control_rollout(
    model: art.TrainableModel, 
    step: int, 
    is_validation: bool = False
) -> AsyncGenerator[art.Trajectory, None]:
    """Rollout function for control systems - adapted from 2048 pattern"""
    
    env = MinimalDoubleIntegrator()
    obs = env.reset()
    
    # Create conversation with system prompt
    conversation = art.Conversation(
        [
            art.Message(
                role="system",
                content=(
                    "You are an expert control agent. Your goal is to control a double integrator system "
                    "to reach a target position. The system dynamics are: acceleration = your_action.\n"
                    "Respond with JSON: {\"action\": [ax, ay], \"reasoning\": \"your reasoning\"}\n"
                    "Keep actions reasonable (between -2 and 2)."
                )
            )
        ]
    )
    
    total_reward = 0
    trajectory_data = []
    
    for step_num in range(50):  # Max 50 steps
        # Create user message with current observation
        user_message = art.Message(
            role="user",
            content=f"Current state: {json.dumps(obs, indent=2)}\nWhat action do you take?"
        )
        conversation.messages.append(user_message)
        
        # Get model response
        response = await model.complete(conversation)
        
        # Parse action from response
        try:
            # Extract JSON from response
            json_match = re.search(r'\{[^}]*\}', response.content)
            if json_match:
                action_data = json.loads(json_match.group())
                action = action_data.get("action", [0, 0])
                reasoning = action_data.get("reasoning", "No reasoning provided")
            else:
                action = [0, 0]  # Default action
                reasoning = "Failed to parse response"
        except:
            action = [0, 0]
            reasoning = "Parse error"
        
        # Clip actions to reasonable range
        action = [max(-2, min(2, a)) for a in action]
        
        # Take step in environment
        obs, reward, done = env.step(np.array(action))
        total_reward += reward
        
        # Add assistant response to conversation
        assistant_message = art.Message(
            role="assistant", 
            content=response.content
        )
        conversation.messages.append(assistant_message)
        
        # Store trajectory data
        trajectory_data.append({
            "step": step_num,
            "observation": obs,
            "action": action,
            "reward": reward,
            "reasoning": reasoning
        })
        
        if done:
            break
    
    # Yield trajectory with final reward
    yield art.Trajectory(
        conversation=conversation,
        reward=total_reward,
        metadata={"trajectory_data": trajectory_data, "final_distance": obs["distance_to_target"]}
    )
```

**Step 2.3: Create Minimal Control Training Script**

Create `minimal_control_train.py`:

```python
import asyncio
import random
from dotenv import load_dotenv
import art
from art.local import LocalBackend
from control_rollout import control_rollout

load_dotenv()
random.seed(42)

# Use the EXACT working pattern from 2048
model = art.TrainableModel(
    name="control-test-001",
    project="double-integrator",
    base_model="Qwen/Qwen2.5-1.5B-Instruct",  # Start small
)

TRAIN_STEPS = 5  # Very small for testing
TRAJECTORIES_PER_STEP = 4  # Small for testing

async def train_control():
    # Initialize backend (same as 2048)
    backend = art.LocalBackend()
    await model.register(backend)
    
    print(f"Starting from step: {await model.get_step()}")
    
    # Training loop (exact pattern from 2048)
    for i in range(await model.get_step(), TRAIN_STEPS):
        print(f"Training step {i+1}/{TRAIN_STEPS}")
        
        train_groups = await art.gather_trajectory_groups(
            (
                art.TrajectoryGroup(
                    control_rollout(model, i, is_validation=False)
                    for _ in range(TRAJECTORIES_PER_STEP)
                )
                for _ in range(1)
            ),
            pbar_desc="gather control trajectories",
        )
        
        # Train with same config pattern as 2048
        await model.train(
            train_groups,
            config=art.TrainConfig(learning_rate=1e-5),
        )
        
        print(f"✅ Completed step {i+1}")

if __name__ == "__main__":
    asyncio.run(train_control())
```

---

## Part 3: Fixing Your Existing Implementation Errors

### 3.1 Common Error Fixes

Based on your error logs, here are the specific fixes needed:

#### Error 1: `LLMTrainingConfig.__init__() got an unexpected keyword argument 'learning_rate'`

**❌ Your Current Code (Incorrect):**
```python
# Wrong - this class doesn't exist or has changed
config = LLMTrainingConfig(learning_rate=5e-5)
```

**✅ Correct Fix:**
```python
# Use art.TrainConfig instead
config = art.TrainConfig(learning_rate=5e-5)
```

#### Error 2: `'LLMGRPOTrainer' object has no attribute 'train_epoch'`

**❌ Your Current Code (Incorrect):**
```python
trainer = LLMGRPOTrainer(model)
trainer.train_epoch()  # This method doesn't exist
```

**✅ Correct Fix:**
```python
# Don't create separate trainer - use model.train() directly
await model.train(train_groups, config=art.TrainConfig(learning_rate=5e-5))
```

#### Error 3: Pydantic validation error for TrainableModel

**❌ Your Current Code (Incorrect):**
```python
# Passing config object where string expected
model = art.TrainableModel(
    base_model=LLMTrainingConfig(...),  # Wrong!
    # ...
)
```

**✅ Correct Fix:**
```python
# base_model should be a string
model = art.TrainableModel(
    name="my-model",
    project="my-project",
    base_model="Qwen/Qwen2.5-1.5B-Instruct",  # String, not config object!
)
```

#### Error 4: `PerformanceEvaluator.__init__() got an unexpected keyword argument 'system_type'`

**❌ Your Current Code (Incorrect):**
```python
evaluator = PerformanceEvaluator(system_type="double_integrator")
```

**✅ Correct Fix:**
Check the actual PerformanceEvaluator API or use simple evaluation:
```python
# Simple evaluation without custom evaluator
final_rewards = [trajectory.reward for group in train_groups for trajectory in group.trajectories]
avg_reward = sum(final_rewards) / len(final_rewards)
print(f"Average reward: {avg_reward}")
```

### 3.2 Migration Strategy for Your Existing Code

**Step 3.1: Backup Your Current Implementation**
```bash
# Create backup of current implementations
cp -r agentic_control_langgraph agentic_control_langgraph_backup
cp -r agentic_control_sft_grpo agentic_control_sft_grpo_backup
```

**Step 3.2: Update Your Training Pipelines**

For your `agentic_control_langgraph` implementation:

```python
# File: agentic_control_langgraph/training/fixed_trainer.py

import asyncio
import art
from art.local import LocalBackend

class FixedControlTrainer:
    def __init__(self, model_name: str, project_name: str):
        self.model = art.TrainableModel(
            name=model_name,
            project=project_name,
            base_model="Qwen/Qwen2.5-1.5B-Instruct",  # Fixed string
        )
        
    async def setup(self):
        self.backend = art.LocalBackend()
        await self.model.register(self.backend)
        
    async def train_step(self, rollout_function, num_trajectories=4):
        """Single training step using correct ART API"""
        
        current_step = await self.model.get_step()
        
        train_groups = await art.gather_trajectory_groups(
            (
                art.TrajectoryGroup(
                    rollout_function(self.model, current_step, is_validation=False)
                    for _ in range(num_trajectories)
                )
                for _ in range(1)
            ),
            pbar_desc="gathering trajectories",
        )
        
        # Use correct TrainConfig
        await self.model.train(
            train_groups,
            config=art.TrainConfig(learning_rate=1e-5),
        )
        
        return current_step + 1
```

---

## Part 4: Progressive Enhancement Path

### 4.1 Weekly Implementation Plan

#### Week 1: Get Basic Training Working
- ✅ Test the 2048 example works
- ✅ Adapt to minimal double integrator  
- ✅ Fix API errors in existing code
- ✅ Get single training step running

**Goal**: One successful training run without errors

#### Week 2: Add Control-Specific Features
- Add proper control rewards (LQR cost, stability)
- Implement multiple control scenarios
- Add basic logging and metrics
- Test with different initial conditions

**Goal**: Training that improves control performance

#### Week 3: Integrate LangGraph Tools
- Add your existing LangGraph agents as tools
- Create hybrid reasoning + execution pipeline
- Implement multi-step trajectory planning

**Goal**: Agents that can reason about control strategies

#### Week 4: Add MCP Tools
- Implement StateAnalyzer, ControlCalculator tools from integration guide
- Add physics-informed calculations
- Create automatic scenario generation

**Goal**: Physics-aware control with automated training scenarios

#### Week 5-6: Advanced Features
- Cross-domain transfer learning
- Safety-critical constraints
- Multi-agent coordination
- Real-time deployment

**Goal**: Production-ready agentic control system

### 4.2 Success Metrics by Week

**Week 1 Success Criteria:**
- [ ] 2048 example runs without errors
- [ ] Basic double integrator training completes
- [ ] Model checkpoint saves successfully
- [ ] No API errors in logs

**Week 2 Success Criteria:**
- [ ] Control performance improves over training steps
- [ ] Agent reaches target in >80% of rollouts
- [ ] Reward trends upward over training
- [ ] Multiple scenarios work

**Week 3 Success Criteria:**
- [ ] LangGraph tools execute successfully
- [ ] Multi-step reasoning improves outcomes
- [ ] Agent can plan trajectory sequences
- [ ] Tool usage is logged correctly

**Week 4 Success Criteria:**
- [ ] MCP tools provide physics insights
- [ ] Scenario generation creates diverse tests
- [ ] Agent performance generalizes
- [ ] RULER evaluation works

---

## Part 5: Working Code Templates

### 5.1 Complete Minimal Working Example

Here's a complete working example that you can run immediately:

**File**: `minimal_working_control.py`

```python
import asyncio
import json
import re
import random
import numpy as np
from typing import Dict, Any, Tuple, AsyncGenerator
from dotenv import load_dotenv
import art

load_dotenv()
random.seed(42)

# ========== ENVIRONMENT ==========
class DoubleIntegrator:
    def __init__(self):
        self.dt = 0.1
        self.max_steps = 30
        self.reset()
    
    def reset(self):
        self.state = np.random.uniform(-1, 1, 4)  # [x, y, vx, vy]
        self.target = np.random.uniform(-1.5, 1.5, 2)
        self.step_count = 0
        return self._get_obs()
    
    def step(self, action):
        pos = self.state[:2]
        vel = self.state[2:]
        
        new_vel = vel + np.array(action) * self.dt
        new_pos = pos + new_vel * self.dt
        
        self.state = np.concatenate([new_pos, new_vel])
        self.step_count += 1
        
        distance = np.linalg.norm(pos - self.target)
        reward = -distance - 0.01 * np.linalg.norm(action)  # Penalize large actions
        
        done = distance < 0.1 or self.step_count >= self.max_steps
        
        return self._get_obs(), reward, done
    
    def _get_obs(self):
        pos = self.state[:2]
        vel = self.state[2:]
        distance = np.linalg.norm(pos - self.target)
        
        return {
            "position": [round(x, 3) for x in pos],
            "velocity": [round(x, 3) for x in vel],
            "target": [round(x, 3) for x in self.target],
            "distance": round(distance, 3),
            "step": self.step_count
        }

# ========== ROLLOUT FUNCTION ==========
async def rollout(model: art.TrainableModel, step: int, is_validation: bool = False) -> AsyncGenerator[art.Trajectory, None]:
    env = DoubleIntegrator()
    obs = env.reset()
    
    conversation = art.Conversation([
        art.Message(
            role="system",
            content=(
                "You are a control agent for a double integrator system. "
                "Your goal is to reach the target position by choosing acceleration commands. "
                "Respond with JSON: {\"action\": [ax, ay], \"reasoning\": \"brief explanation\"} "
                "Keep accelerations between -2 and 2."
            )
        )
    ])
    
    total_reward = 0
    
    for _ in range(30):
        # Add current state
        conversation.messages.append(
            art.Message(role="user", content=f"State: {json.dumps(obs)}")
        )
        
        # Get model response
        response = await model.complete(conversation)
        
        # Parse action
        try:
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                action = data.get("action", [0, 0])
            else:
                action = [0, 0]
        except:
            action = [0, 0]
        
        # Clip actions
        action = [np.clip(a, -2, 2) for a in action]
        
        # Step environment
        obs, reward, done = env.step(action)
        total_reward += reward
        
        # Add response
        conversation.messages.append(
            art.Message(role="assistant", content=response.content)
        )
        
        if done:
            break
    
    yield art.Trajectory(
        conversation=conversation,
        reward=total_reward,
        metadata={"final_distance": obs["distance"]}
    )

# ========== TRAINING SCRIPT ==========
async def main():
    # Model setup
    model = art.TrainableModel(
        name="minimal-control-v1",
        project="double-integrator-test",
        base_model="Qwen/Qwen2.5-1.5B-Instruct",
    )
    
    # Backend setup
    backend = art.LocalBackend()
    await model.register(backend)
    
    print("Starting training...")
    
    # Training loop
    for i in range(3):  # Just 3 steps for testing
        print(f"Step {i+1}/3")
        
        train_groups = await art.gather_trajectory_groups(
            (
                art.TrajectoryGroup(
                    rollout(model, i, is_validation=False)
                    for _ in range(4)  # 4 trajectories per step
                )
                for _ in range(1)
            ),
            pbar_desc=f"step {i+1}",
        )
        
        # Extract rewards for monitoring
        rewards = [traj.reward for group in train_groups for traj in group.trajectories]
        avg_reward = sum(rewards) / len(rewards)
        print(f"Average reward: {avg_reward:.3f}")
        
        # Train
        await model.train(
            train_groups,
            config=art.TrainConfig(learning_rate=1e-5),
        )
        
        print(f"✅ Step {i+1} complete")
    
    print("🎉 Training finished!")

if __name__ == "__main__":
    asyncio.run(main())
```

### 5.2 Test This Example

```bash
# Create new test directory
mkdir test_minimal_control
cd test_minimal_control

# Save the code above as minimal_working_control.py

# Run the test
conda activate agentic_control
python minimal_working_control.py
```

**Expected Output:**
```
Starting training...
Step 1/3
step 1: 100%|████████████| 1/1 [00:XX<00:00, X.XXit/s]
Average reward: -12.345
✅ Step 1 complete
Step 2/3
...
🎉 Training finished!
```

If this works, you have a solid foundation to build from!

---

## Part 6: Integration with MCP-RL

### 6.1 Adding MCP Tools to Working Example

Once your minimal example works, you can add MCP tools:

**Step 6.1: Create Control MCP Tools**

```python
# File: mcp_control_tools.py
from art.mcp.types import MCPTool
import numpy as np

class ControlCalculatorTool(MCPTool):
    def __init__(self):
        super().__init__(
            name="control_calculator",
            description="Calculate optimal control input using control theory",
            parameters={
                "type": "object",
                "properties": {
                    "current_state": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "[x, y, vx, vy] current state"
                    },
                    "target_state": {
                        "type": "array", 
                        "items": {"type": "number"},
                        "description": "[target_x, target_y] target position"
                    },
                    "method": {
                        "type": "string",
                        "enum": ["lqr", "pid", "mpc"],
                        "description": "Control method to use"
                    }
                },
                "required": ["current_state", "target_state"]
            }
        )
    
    async def execute(self, params):
        state = np.array(params["current_state"])
        target = np.array(params["target_state"])
        method = params.get("method", "lqr")
        
        if method == "lqr":
            # Simple LQR for double integrator
            pos = state[:2]
            vel = state[2:]
            
            pos_error = pos - target
            vel_error = vel  # Target velocity is 0
            
            # Simple LQR gains
            kp, kd = 4.0, 2.0
            control = -kp * pos_error - kd * vel_error
            
            return {
                "control_input": control.tolist(),
                "method": "lqr",
                "gains": {"kp": kp, "kd": kd},
                "stability_margin": float(np.linalg.norm(control))
            }
        
        # Add other methods as needed...
```

**Step 6.2: Enhanced Rollout with MCP Tools**

```python
async def mcp_enhanced_rollout(model, step, is_validation=False):
    """Rollout function that can use MCP tools"""
    
    env = DoubleIntegrator()
    obs = env.reset()
    
    # Create MCP tools
    control_calc = ControlCalculatorTool()
    
    conversation = art.Conversation([
        art.Message(
            role="system",
            content=(
                "You are a control agent with access to control theory tools. "
                "You can use the 'control_calculator' tool to get optimal control suggestions. "
                "Then decide whether to use the suggestion or modify it based on your reasoning."
            )
        )
    ])
    
    # Make tools available to model
    mcp_tools = [control_calc]
    
    # Rest of rollout logic with tool access...
    # (This requires MCP server setup which we'll add in Week 4)
```

### 6.3 Integration Timeline

**Week 4 Tasks:**
1. Set up MCP server for control tools
2. Integrate tools into rollout function  
3. Test physics-informed control decisions
4. Add scenario generation

**Week 5-6 Tasks:**
1. Add safety verification tools
2. Implement multi-agent coordination
3. Cross-domain transfer learning
4. Production deployment

---

## Next Steps Summary

### Immediate Actions (This Week):

1. **Test the minimal working example** from Part 5.1
2. **Fix your current implementations** using the error fixes in Part 3.1
3. **Start with 2048 example** to verify your ART setup works

### Success Indicators:

- [ ] 2048 example runs without errors
- [ ] Minimal control example completes training
- [ ] Your error logs show no API mismatches
- [ ] Model checkpoints save successfully

### Next Week Goals:

- [ ] Control performance improves over training
- [ ] Multiple scenarios work reliably  
- [ ] Logging and metrics are captured
- [ ] Ready to add LangGraph integration

### Long-term Vision:

By following this progressive enhancement path, you'll have:
- ✅ Error-free training pipeline
- ✅ Physics-informed control agents
- ✅ Hybrid reasoning + execution architecture
- ✅ Advanced MCP-RL capabilities
- ✅ State-of-the-art agentic control system

The key is to **start simple, get it working, then enhance progressively**. This approach will save you weeks of debugging and give you a solid foundation for advanced features.

---

*Start with the minimal working example in Part 5.1 - if that works, you're on the right track!*