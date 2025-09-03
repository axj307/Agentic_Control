# ART Training Integration

This directory contains the ART (Actor-Residual Training) integration for the agentic control pipeline.

## Core Files

### Training Scripts
- `train_control_agent.py` - **Main training script** (comprehensive ART pipeline)
- `art_integration.py` - ART framework integration utilities
- `ruler_rewards.py` - Reward calculation for control tasks

### Testing & Development
- `test_art.py` - Integration tests for ART system
- `test_art_setup.py` - Setup verification tests
- `test_basic_rollout.py` - Basic rollout functionality tests

### Legacy Files (remove during cleanup)
- `train_agent.py` - Old training script
- `run_training.py` - Deprecated training runner

## Data & Models

### Models Directory (`art_models/`)
- `linear_baseline.json` - Baseline model configuration
- Trained models saved here during experiments

### Data Directory (`art_data/`)
- Training data in ART format
- Generated from control experiments

## Usage

### Main Training Pipeline
```bash
# Full training with evaluation
conda activate agentic_control && python 05_training/train_control_agent.py --episodes 100

# Evaluation only
conda activate agentic_control && python 05_training/train_control_agent.py --eval_only --model_path ./models/control-agent-v1

# Quick test
conda activate agentic_control && python 05_training/test_art.py
```

### Integration Testing
```bash
# Test ART setup
conda activate agentic_control && python 05_training/test_art_setup.py

# Test basic rollouts
conda activate agentic_control && python 05_training/test_basic_rollout.py
```

## Configuration

### ART Config (`art_config.json`)
```json
{
  "model": {
    "type": "linear_baseline",
    "parameters": {...}
  },
  "training": {
    "episodes": 100,
    "difficulty": "medium"
  }
}
```

## Integration with Main Pipeline

1. **Data Generation**: Use `run_experiments.py` to generate trajectory data
2. **ART Training**: Use `train_control_agent.py` to train on generated data  
3. **Evaluation**: Compare trained agent against baselines
4. **Results**: Saved to `results/` directory

## Checkpoint Management

**Important**: Checkpoints are automatically saved to `/scratch` to avoid filling the 200GB home directory quota.

### Checkpoint Configuration
- **Location**: `/scratch/{username}/agentic_control_checkpoints/`
- **Auto-configured**: HuggingFace transformers automatically use scratch directory
- **Archived models**: Existing models moved to `archived_models_YYYYMMDD/`

### Checkpoint Commands
```bash
# Check checkpoint status
./manage_checkpoints.sh status

# Clean up old checkpoints (keep latest 3)
./manage_checkpoints.sh cleanup 3

# Setup checkpoint directory
./manage_checkpoints.sh setup

# Or use Python directly
conda activate agentic_control && python 05_training/checkpoint_config.py --status
```

### Manual Management
```bash
# View checkpoint directory
ls -la /scratch/$USER/agentic_control_checkpoints/

# Check disk usage
du -sh /scratch/$USER/agentic_control_checkpoints/
```

## Dependencies

- Core: `numpy`, `scipy`, `matplotlib`
- ART: `openpipe-art` (optional - mock implementation available)
- LLM: `openai` or `vllm` (for controller backends)

## Architecture

```
experiment_config.py -> run_experiments.py -> 05_training/train_control_agent.py
                                          |
                                          v
                                    results/data/*.json
```

The training pipeline consumes trajectory data from the main experiment runner and produces trained agents that can be evaluated against the baseline controllers.