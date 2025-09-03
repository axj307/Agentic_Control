# Detailed TODO: Minimal Agentic Control Implementation

## Project Goal
Build a step-by-step understanding of agentic control for aerospace systems by implementing and comparing two approaches:
1. **Direct LLM Control**: LLM directly outputs control actions
2. **Tool-Augmented Control**: LLM uses tools (via LangGraph) for physics calculations

---

## ✅ COMPLETED WORK

### Phase 1: Basic Setup ✅
- [x] Created project structure with 5 step-by-step folders
- [x] Created README.md with high-level overview
- [x] Created requirements.txt with minimal dependencies
- [x] Implemented `01_basic_physics/double_integrator.py`
  - Simple double integrator class (ẍ = u)
  - PD controller baseline for comparison
  - Trajectory plotting capabilities
- [x] Created `01_basic_physics/test_environment.py`
  - Physics verification tests
  - PD controller performance baseline
- [x] Created `01_basic_physics/visualize_dynamics.py`
  - Phase portraits
  - Control strategy comparisons
  - Difficulty analysis
- [x] Created `02_direct_control/simple_controller.py`
  - Mock LLM controller implementation
  - Natural language observation interface
  - JSON action response format
  - Comparison with PD baseline

---

## 🔄 IMMEDIATE NEXT STEPS (Today/Tomorrow)

### 1. Verify Current Implementations
```bash
# Test physics implementation
cd 01_basic_physics
python test_environment.py  # Should show 100% success rate for PD controller

# Test visualization
python visualize_dynamics.py  # Should generate comparison plots

# Test direct controller
cd ../02_direct_control
python simple_controller.py  # Should show Mock LLM performance
```

Expected outputs:
- Physics tests: All pass
- PD Controller: ~34 steps average, 100% success
- Mock LLM: ~85-90% success (intentionally imperfect)

### 2. Complete Direct Control Implementation

#### 2a. Create Real LLM Integration (`02_direct_control/llm_integration.py`)
```python
# TODO: Implement these classes
class VLLMController:
    """Connect to local vLLM server"""
    def __init__(self, model_name="Qwen/Qwen2.5-1.5B-Instruct", port=8000):
        # Connect to vLLM server
        # Handle inference requests
        pass

class OpenAIController:
    """Connect to OpenAI API (backup option)"""
    def __init__(self, api_key=None, model="gpt-3.5-turbo"):
        # Initialize OpenAI client
        # Handle API requests
        pass
```

#### 2b. Create Rollout Function (`02_direct_control/rollout.py`)
```python
# TODO: Implement rollout for ART training
async def control_rollout(model, scenario):
    """Execute one rollout for training"""
    # Initialize environment
    # Run control loop
    # Track trajectory
    # Calculate reward
    # Return ART Trajectory object
    pass
```

#### 2c. Test Real LLM Control (`02_direct_control/test_llm.py`)
```python
# TODO: Test script that:
# 1. Starts vLLM server (or connects to it)
# 2. Runs control episodes
# 3. Compares with PD baseline
# 4. Saves results to JSON
```

---

## 📋 PHASE 3: LANGGRAPH TOOLS (Week 2)

### 3. Implement Control Tools (`03_langgraph_tools/control_tools.py`)

#### Tool 1: ErrorAnalyzerTool
```python
@tool
def analyze_errors(position: float, velocity: float, 
                   target_pos: float, target_vel: float) -> dict:
    """
    Analyze control errors and system state.
    
    Returns:
    - position_error: Distance from target
    - velocity_error: Velocity mismatch
    - total_error: Combined error metric
    - urgency: high/medium/low
    - stability: stable/unstable/marginal
    """
    # TODO: Implement physics-based error analysis
    pass
```

#### Tool 2: PIDCalculatorTool
```python
@tool
def calculate_pid_control(pos_error: float, vel_error: float,
                          kp: float = 1.0, kd: float = 2.0) -> dict:
    """
    Calculate PID control action.
    
    Returns:
    - control_action: Recommended force
    - components: {p_term, d_term}
    - saturated: bool (if hitting limits)
    """
    # TODO: Implement PID calculation with anti-windup
    pass
```

#### Tool 3: TrajectoryPlannerTool
```python
@tool
def plan_trajectory(current_state: list, target_state: list,
                   time_horizon: float = 5.0) -> dict:
    """
    Plan optimal trajectory to target.
    
    Returns:
    - waypoints: List of intermediate targets
    - control_sequence: Suggested control inputs
    - estimated_time: Time to reach target
    """
    # TODO: Implement simple trajectory planning
    pass
```

#### Tool 4: SafetyVerifierTool
```python
@tool
def verify_safety(control_action: float, current_state: list,
                 max_force: float = 1.0) -> dict:
    """
    Verify control action safety.
    
    Returns:
    - safe: bool
    - warnings: List of safety concerns
    - adjusted_action: Safe control value
    """
    # TODO: Check constraints and stability
    pass
```

### 4. Build Control Graph (`03_langgraph_tools/control_graph.py`)

```python
from langgraph.graph import StateGraph, END

# TODO: Implement graph structure
def build_control_graph():
    """
    Build LangGraph control workflow.
    
    Nodes:
    1. observe_state - Get current observation
    2. analyze_errors - Use ErrorAnalyzerTool
    3. plan_action - Use PIDCalculatorTool or TrajectoryPlannerTool
    4. verify_safety - Use SafetyVerifierTool
    5. execute_action - Output final control
    
    Edges:
    - observe -> analyze
    - analyze -> plan
    - plan -> verify
    - verify -> execute or back to plan if unsafe
    """
    graph = StateGraph()
    # TODO: Add nodes and edges
    return graph.compile()
```

### 5. Test Tool-Augmented Control (`03_langgraph_tools/test_graph.py`)

```python
# TODO: Test each tool independently
def test_individual_tools():
    """Test each tool in isolation"""
    pass

# TODO: Test full graph execution
def test_graph_control():
    """Test complete control loop with graph"""
    pass

# TODO: Compare with direct control
def compare_approaches():
    """Compare direct vs tool-augmented control"""
    pass
```

---

## 📊 PHASE 4: COMPARISON & ANALYSIS (Week 3)

### 6. Create Comparison Framework (`04_comparison/run_comparison.py`)

```python
# TODO: Implement comprehensive comparison
class ControlComparison:
    def __init__(self):
        self.methods = {
            'pd_baseline': PDController(),
            'direct_llm': DirectLLMController(),
            'tool_augmented': ToolAugmentedController(),
        }
    
    def run_comparison(self, scenarios):
        """
        Compare all methods on same scenarios.
        
        Metrics to collect:
        - Success rate
        - Steps to target
        - Control effort
        - Computation time
        - Interpretability score
        """
        pass
    
    def generate_report(self):
        """Generate comparison report with plots"""
        pass
```

### 7. Visualization (`04_comparison/plot_results.py`)

```python
# TODO: Create visualization functions
def plot_trajectories_comparison():
    """Side-by-side trajectory plots"""
    pass

def plot_performance_metrics():
    """Bar charts of key metrics"""
    pass

def plot_reasoning_trace():
    """Visualize tool-augmented reasoning steps"""
    pass
```

---

## 🚂 PHASE 5: TRAINING WITH ART (Week 4)

### 8. Implement ART Training (`05_training/train_agent.py`)

```python
import art
from art import TrainableModel, Trajectory

# TODO: Implement training pipeline
async def train_control_agent(method='direct'):
    """
    Train agent using GRPO.
    
    Steps:
    1. Initialize model
    2. Generate scenarios
    3. Collect trajectories
    4. Calculate rewards
    5. Train with GRPO
    6. Evaluate improvement
    """
    pass
```

### 9. Reward Design (`05_training/rewards.py`)

```python
# TODO: Implement reward functions
def calculate_control_reward(trajectory, scenario):
    """
    Multi-objective reward:
    - Position accuracy: -|pos_error|
    - Velocity accuracy: -|vel_error|  
    - Control efficiency: -sum(|u|)
    - Time penalty: -steps
    - Success bonus: +10 if reached target
    """
    pass
```

### 10. Evaluation (`05_training/evaluate.py`)

```python
# TODO: Implement evaluation
def evaluate_trained_agent(model, test_scenarios):
    """
    Evaluate trained vs untrained performance.
    
    Metrics:
    - Success rate improvement
    - Sample efficiency
    - Generalization to harder scenarios
    - Robustness to perturbations
    """
    pass
```

---

## 🔬 ADVANCED EXTENSIONS (Optional)

### A. Aerospace-Specific Enhancements

1. **Add Spacecraft Dynamics**
   - Copy spacecraft environment from other folders
   - Add attitude control scenarios
   - Test with 3D control problems

2. **Multi-Phase Missions**
   - Orbital transfer scenarios
   - Rendezvous and docking
   - Formation flying

3. **Add Constraints**
   - Fuel limits
   - Actuator saturation
   - Keep-out zones

### B. Advanced LangGraph Features

1. **Adaptive Routing**
   - Simple scenarios → direct control
   - Complex scenarios → full tool suite
   - Emergency situations → safety-first mode

2. **Multi-Agent Coordination**
   - Multiple spacecraft control
   - Distributed decision making
   - Communication delays

3. **Hierarchical Control**
   - High-level mission planning
   - Mid-level trajectory generation
   - Low-level control execution

### C. Research Extensions

1. **Transfer Learning**
   - Train on double integrator
   - Transfer to spacecraft
   - Measure performance degradation

2. **Explainability Study**
   - Track reasoning chains
   - Identify failure modes
   - Generate natural language explanations

3. **Robustness Analysis**
   - Add noise to observations
   - Test with model mismatch
   - Evaluate under failures

---

## 📝 DOCUMENTATION NEEDS

### Required Documentation

1. **API Documentation** (`docs/api.md`)
   - All tool interfaces
   - Graph state format
   - Controller APIs

2. **Tutorial Notebook** (`notebooks/tutorial.ipynb`)
   - Step-by-step walkthrough
   - Interactive examples
   - Visualization of results

3. **Configuration Guide** (`docs/config.md`)
   - vLLM setup instructions
   - Model selection guide
   - Hyperparameter tuning

4. **Results Report** (`docs/results.md`)
   - Performance comparisons
   - Key findings
   - Recommendations for aerospace applications

---

## 🎯 SUCCESS CRITERIA

### Minimum Success (Must Have)
- [ ] Double integrator works with both control approaches
- [ ] Clear performance comparison between methods
- [ ] At least one method achieves >90% success rate
- [ ] Tool-augmented approach shows interpretable reasoning

### Target Success (Should Have)
- [ ] Both methods trainable with ART/GRPO
- [ ] Training improves performance by >20%
- [ ] Tool-augmented outperforms direct by >15%
- [ ] Complete documentation and tutorials

### Stretch Goals (Nice to Have)
- [ ] Spacecraft environment working
- [ ] Multi-agent coordination demo
- [ ] Transfer learning demonstrated
- [ ] Published as open-source package

---

## 🚀 QUICK START COMMANDS

```bash
# Set up environment
conda activate agentic_control
cd agentic_control_minimal

# Test what's working
python 01_basic_physics/test_environment.py
python 02_direct_control/simple_controller.py

# Start vLLM server (for real LLM testing)
vllm serve Qwen/Qwen2.5-1.5B-Instruct --port 8000

# Install additional dependencies as needed
pip install langgraph langchain-core  # For Phase 3
pip install openpipe-art  # For Phase 5
```

---

## 📅 SUGGESTED TIMELINE

**Week 1**: Complete Phases 1-2 (Physics + Direct Control)
**Week 2**: Complete Phase 3 (LangGraph Tools)
**Week 3**: Complete Phase 4 (Comparison)
**Week 4**: Complete Phase 5 (Training)
**Week 5+**: Advanced extensions based on results

---

## 💡 KEY INSIGHTS TO EXPLORE

1. **Does tool-augmented control really help?**
   - Hypothesis: Yes, for complex scenarios
   - Test: Compare on increasingly difficult tasks

2. **Can LLMs learn control without physics knowledge?**
   - Hypothesis: Partially, but inefficiently
   - Test: Train from scratch vs with tools

3. **What's the sample efficiency difference?**
   - Hypothesis: Tools reduce training time 10x
   - Test: Measure episodes to convergence

4. **How does this scale to real aerospace problems?**
   - Hypothesis: Tool approach scales better
   - Test: Transfer from simple to complex systems

---

## 🐛 COMMON ISSUES & SOLUTIONS

1. **vLLM server not starting**
   - Check GPU memory
   - Try smaller model (Qwen2.5-0.5B)
   - Use CPU inference as fallback

2. **LangGraph import errors**
   - Install specific version: `pip install langgraph==0.2.0`
   - Check compatibility with langchain-core

3. **ART training fails**
   - Verify trajectory format
   - Check reward calculation
   - Start with fewer scenarios

4. **Poor control performance**
   - Tune PD gains for baseline
   - Adjust prompt for LLM
   - Check observation/action scaling

---

**Remember**: Start simple, test often, document everything. The goal is understanding, not just implementation!