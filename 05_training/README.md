# ART (Automated Red Teaming) Training Pipeline

This directory contains the ART training pipeline for developing robust agentic control systems through adversarial training and data-driven improvements.

## 🎯 Overview

The ART pipeline implements an automated red teaming framework that:

1. **Generates Adversarial Scenarios**: Creates challenging control scenarios automatically
2. **Collects Training Data**: Gathers trajectories from multiple controllers 
3. **Analyzes Failure Patterns**: Identifies systematic weaknesses in control approaches
4. **Trains Improved Models**: Learns from successful behaviors to create better controllers
5. **Evaluates Performance**: Tests trained models on unseen challenging scenarios

## 📁 Files

- `train_agent.py` - Main ART pipeline implementation
- `art_config.json` - Configuration file with training parameters
- `test_art.py` - Test suite for validating ART components
- `README.md` - This documentation file

## 🚀 Quick Start

### Basic Usage

```bash
# Test the ART pipeline
python test_art.py

# Run full ART pipeline with default settings
python train_agent.py --mode full_pipeline --num_episodes 100

# Generate data only
python train_agent.py --mode generate_data --num_episodes 500

# Train model on existing data
python train_agent.py --mode train --data_path art_data/art_data_123456.json

# Evaluate trained model
python train_agent.py --mode evaluate --model_path art_models/linear_baseline.json
```

### Configuration

Edit `art_config.json` to customize training parameters:

```json
{
  "num_episodes": 1000,
  "difficulty_levels": ["easy", "medium", "hard", "extreme", "adversarial"],
  "position_range": [-2.0, 2.0],
  "velocity_range": [-1.5, 1.5],
  "model_type": "supervised",
  "test_scenarios": 500,
  "success_threshold": 0.95
}
```

## 🏗️ Architecture

### Core Components

1. **AdversarialScenarioGenerator**
   - Creates challenging control scenarios
   - Analyzes failure patterns from previous runs
   - Generates scenarios focused on discovered weaknesses

2. **ARTDataCollector** 
   - Collects trajectory data from multiple controllers
   - Manages training/test dataset splits
   - Provides data preprocessing utilities

3. **ARTTrainer**
   - Trains supervised learning models from successful trajectories
   - Implements baseline linear regression approach
   - Extensible for neural network models

4. **ARTManager**
   - Orchestrates the complete ART pipeline
   - Manages data flow between components
   - Handles experimental configuration

### Data Flow

```
Scenario Generation → Multi-Controller Evaluation → Data Collection
        ↑                                               ↓
Adversarial Analysis ← Model Evaluation ← Model Training
```

## 📊 Generated Data Structure

### Trajectory Data
```json
{
  "positions": [1.0, 0.9, 0.7, ...],
  "velocities": [0.0, -0.1, -0.2, ...], 
  "actions": [-1.0, -0.8, -0.6, ...],
  "success": true,
  "num_steps": 25,
  "controller_name": "tool_augmented",
  "difficulty": "hard",
  "reasoning": ["Phase: approach | PID control...", ...],
  "confidence": [0.85, 0.92, ...]
}
```

### Analysis Results
```json
{
  "total_episodes": 1000,
  "failure_rate": 0.15,
  "failure_patterns": {
    "position_range": [-1.8, 1.5],
    "velocity_range": [-1.2, 0.9],
    "common_positions": [-1.2, 0.8, 1.5]
  }
}
```

## 🧠 Model Training

### Current Implementation
- **Linear Regression Baseline**: Simple linear policy learning from successful trajectories
- **State-Action Pairs**: Extracts (position, velocity) → action mappings
- **Evaluation Metrics**: MSE, R² score, success rate on test scenarios

### Extensibility
The framework is designed for easy extension to:
- **Neural Networks**: Deep learning models for complex policies
- **Reinforcement Learning**: RL agents trained on ART scenarios
- **Hybrid Approaches**: Combining supervised and RL methods

## 📈 Performance Metrics

### Training Metrics
- **Data Collection**: Number of episodes, success/failure rates
- **Model Training**: MSE, R² score, convergence metrics  
- **Pattern Analysis**: Failure clustering, scenario difficulty distribution

### Evaluation Metrics
- **Success Rate**: Percentage of test scenarios completed successfully
- **Robustness**: Performance across different difficulty levels
- **Efficiency**: Average steps to completion
- **Confidence**: Model certainty in control decisions

## 🔬 Research Applications

### Aerospace Control
- **Spacecraft Maneuvering**: Train controllers for complex orbital maneuvers
- **Landing Systems**: Develop robust controllers for planetary landing
- **Formation Flying**: Multi-agent control coordination

### Robustness Testing  
- **Adversarial Scenarios**: Systematic discovery of failure modes
- **Safety Verification**: Validation of control system boundaries
- **Performance Boundaries**: Characterization of operating envelopes

### Data-Driven Control
- **Transfer Learning**: Apply learned policies to new scenarios
- **Domain Adaptation**: Adapt controllers to different system parameters
- **Continuous Learning**: Online improvement through operational data

## 🛠️ Advanced Usage

### Custom Controllers

Add your own controllers to the ART pipeline:

```python
class MyController:
    def get_action(self, position, velocity, target_pos=0.0, target_vel=0.0):
        # Your control logic here
        action = my_control_law(position, velocity)
        return {
            'action': action,
            'confidence': 0.9,
            'reasoning': 'My control approach'
        }

# Add to ART pipeline
controllers = {
    'my_controller': MyController(),
    # ... other controllers
}
```

### Custom Scenario Generation

Implement domain-specific scenario generators:

```python
class MyScenarioGenerator(AdversarialScenarioGenerator):
    def generate_custom_scenarios(self, num_scenarios):
        # Your scenario generation logic
        scenarios = []
        for i in range(num_scenarios):
            scenario = ControlScenario(
                name=f"custom_{i}",
                # ... custom parameters
            )
            scenarios.append(scenario)
        return scenarios
```

## 🔧 Dependencies

Required packages:
```bash
pip install numpy scipy scikit-learn
```

Optional for advanced features:
```bash
pip install torch tensorflow  # For neural network models
pip install gym stable-baselines3  # For RL integration
```

## 📝 Logging and Debugging

The ART pipeline provides comprehensive logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Key log events:
- Data collection progress
- Training metrics and convergence
- Failure pattern analysis
- Model evaluation results

## 🤝 Contributing

To extend the ART pipeline:

1. **Add New Model Types**: Implement in `ARTTrainer.train_*_model()`
2. **Improve Scenario Generation**: Enhance `AdversarialScenarioGenerator`
3. **Add Evaluation Metrics**: Extend evaluation functions
4. **Integration Tests**: Add tests in `test_art.py`

## 📚 References

- **Automated Red Teaming**: Adversarial testing methodologies
- **Control Theory**: Optimal and robust control approaches  
- **Machine Learning**: Supervised and reinforcement learning for control
- **Agentic AI**: Tool-augmented reasoning systems

---

The ART pipeline enables systematic development of robust, high-performance agentic control systems through data-driven adversarial training.