# SLURM Scripts Directory

Contains scripts for running experiments on SLURM cluster systems.

## Files

### Experiment Scripts
- `run_clean_experiments.sh` - Main experiment runner for PD vs Tool comparison
- `run_art_analysis.sh` - ART training analysis experiments  
- `experiment_runner_clean.py` - Python backend for clean experiments

### Submission Scripts
- `submit_clean_experiments.sh` - Submit PD vs Tool experiments
  ```bash
  ./slurm/submit_clean_experiments.sh [difficulty] [save_plots]
  ```

- `submit_art_analysis.sh` - Submit ART analysis
  ```bash  
  ./slurm/submit_art_analysis.sh
  ```

## Usage

### Quick Start
```bash
# Submit all experiments
./slurm/submit_clean_experiments.sh all true

# Submit just easy experiments  
./slurm/submit_clean_experiments.sh easy true

# Monitor job
squeue -u $USER
tail -f logs/clean_experiments_*.out
```

### Job Configuration
- **Time Limit**: 1 hour (adjust in scripts if needed)
- **Memory**: 4GB per job
- **Output**: Results saved to `results/` directory
- **Logs**: Saved to `logs/` directory

## Monitoring

```bash
# Check job status
squeue -u $USER

# View job output  
tail -f logs/clean_experiments_JOBID.out

# Cancel job
scancel JOBID
```

## Results Location
- **Data**: `results/data/`
- **Plots**: `results/plots/`
- **Reports**: `results/reports/`
- **Logs**: `logs/`