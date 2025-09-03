# Agentic Control

A comprehensive implementation of ART-based agentic control for aerospace systems, featuring clean experiment pipelines and production-ready infrastructure.

## 🎯 Overview

This repository implements agentic control using ART (Agent Reward Training) for double integrator systems, with a focus on:

- **Clean, portable codebase** - No hard-coded paths, standardized naming
- **Professional visualization** - Publication-ready plots and analysis
- **SLURM integration** - Scalable cluster computing support  
- **ART training pipeline** - Fine-tune tool-augmented controllers

## 🏗️ Architecture

### Controllers Implemented:
1. **PD Baseline** - Classical proportional-derivative control
2. **Tool-Augmented** - LLM with physics-aware tools (6 specialized tools)
3. **ART-Trained** - Fine-tuned tool-augmented controller (coming soon)

### Tool-Augmented Control Flow:
```
State → LangGraph → [Analyze → Select Strategy → Calculate → Verify] → Action
```

## 📁 Project Structure

```
├── 01_basic_physics/       # Double integrator environment
├── 03_langgraph_tools/     # Tool-augmented controller
├── 05_training/            # ART integration modules
├── slurm/                  # SLURM cluster scripts
├── results/                # Organized experimental data
├── run_experiments.py      # Main experiment pipeline
├── experiment_config.py    # Centralized configuration
└── clean_results.py       # Maintenance utilities
```

## 🚀 Quick Start

### Local Experiments
```bash
# Run comparison experiments
python run_experiments.py --difficulty all

# Check results
./maintain_results.sh status
```

### SLURM Cluster
```bash
# Submit to cluster
./submit_clean_experiments.sh all true

# Monitor progress
squeue -u $USER
```

## 🔧 Installation

```bash
conda activate agentic_control
pip install -r requirements.txt
```

## 📊 Current Results

- **PD Baseline**: 100% success across 9 test scenarios
- **Tool-Augmented**: 100% success with ~13% improved efficiency
- **Professional plots**: 4-panel comparison visualizations
- **ART-ready data**: Trajectory format compatible with training pipeline

## 🎯 Key Features

✅ **Production Ready**: Clean, maintainable, portable codebase  
✅ **SLURM Integration**: Scalable cluster computing support  
✅ **Professional Visualization**: Publication-quality plots and analysis  
✅ **ART Compatible**: Training pipeline ready for fine-tuning  
✅ **Automated Maintenance**: Self-organizing results and cleanup tools  

## 📈 Next Steps

Ready for ART implementation to fine-tune the tool-augmented controller using the comprehensive baseline data and robust experimental infrastructure.

---

**Status**: Production-ready research codebase with clean pipeline for agentic control experiments.