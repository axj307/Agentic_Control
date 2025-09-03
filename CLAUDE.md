# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Environment

Always activate the conda environment before running Python commands:
```bash
conda activate agentic_control && python [script.py]
```

## Core Commands

### Running Tests
- Basic physics verification: `conda activate agentic_control && python 01_basic_physics/test_environment.py`
- Direct LLM control test: `conda activate agentic_control && python 02_direct_control/test_llm.py`
- Tool-augmented control test: `conda activate agentic_control && python 03_langgraph_tools/test_graph.py`
- ART integration test: `conda activate agentic_control && python 05_training/test_art.py`

### Running Experiments
- Full comparison experiment: `conda activate agentic_control && python run_experiments.py --difficulty all`
- Quick test (easy scenarios): `conda activate agentic_control && python run_experiments.py --difficulty easy`
- Generate analysis plots: `conda activate agentic_control && python slurm/create_unified_plots.py`

### Development Testing
- Individual component testing pattern: `conda activate agentic_control && python [directory]/test_*.py`
- All tests use pytest format but are run directly as scripts

## Architecture Overview

This is a **step-by-step agentic control implementation** with progressive complexity:

### 1. Double Integrator Physics (`01_basic_physics/`)
- `DoubleIntegrator` class: Core physics simulation (ẍ = u)
- Provides PD controller baseline and visualization capabilities
- Defines test scenarios (EASY_SCENARIOS, MEDIUM_SCENARIOS, HARD_SCENARIOS)

### 2. Direct LLM Control (`02_direct_control/`)
- `BaseLLMController` abstract class with `VLLMController` and `OpenAIController` implementations
- LLMs receive natural language state descriptions and return JSON control actions
- No physics tools - LLM must learn control from scratch

### 3. Tool-Augmented Control (`03_langgraph_tools/`)
- **Key Architecture**: `ToolAugmentedController` uses LangGraph workflow
- **Control Graph Workflow**: observe_state → analyze_errors → select_strategy → plan_action → verify_safety → execute_action
- **Physics Tools** (`control_tools.py`): `analyze_errors`, `select_control_strategy`, `calculate_pid_control`, `calculate_lqr_control`, `plan_trajectory`, `verify_safety`
- Embeds physics knowledge in tools rather than requiring LLM to learn it

### 4. Training Integration (`05_training/`)
- ART (Actor-Residual Training) framework integration
- GRPO (Group Relative Policy Optimization) for RL training
- Generates training data from control experiments

### Configuration System
- `experiment_config.py`: Centralized configuration for scenarios, plotting, and paths
- `DEFAULT_CONFIG` contains all experimental parameters
- Dynamic path resolution with `setup_python_path()` for module imports

### Results Management
- `results/` directory structure: `/data`, `/plots`, `/reports`
- Automatic timestamping and archiving of experimental results
- `maintain_results.sh` for result management

## Key Design Patterns

### Control Interface
All controllers implement the same interface:
```python
result = controller.get_action(position, velocity, target_pos, target_vel)
# Returns: {"action": float, "confidence": float, "reasoning": string}
```

### Physics Tools Pattern
Tools return structured dictionaries with consistent formats:
- Error analysis provides phase/urgency assessment
- Strategy selection recommends control approach
- Control calculation provides action with reasoning
- Safety verification ensures action bounds

### Experiment Runner Pattern
1. Create controller instances
2. Run scenarios from `experiment_config.py`
3. Generate comparison visualizations
4. Save timestamped results with reports

### Mock Implementation Handling
Several modules provide mock implementations when dependencies (LangGraph, OpenAI) are unavailable, allowing development to continue with reduced functionality.

## Dependencies

Core: numpy, scipy, matplotlib, pytest
Optional: langgraph, langchain-core, requests, openai, openpipe-art

Missing dependencies trigger warning messages with installation instructions.