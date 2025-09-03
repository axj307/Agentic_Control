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

## Log Management

SLURM job logs are organized in `slurm/logs/` to keep the codebase clean.

### Log Commands
```bash
# Show recent jobs and their logs
python slurm/manage_logs.py

# Check logs status  
python slurm/manage_logs.py --status

# Show specific job summary
python slurm/manage_logs.py --job-summary 12345

# Clean old logs (keep latest 7 days)
python slurm/manage_logs.py --clean --keep-days 7

# Preview what would be cleaned
python slurm/manage_logs.py --clean --dry-run
```

### Log Files
- **Location**: `slurm/logs/`
- **Format**: `{job_name}_{job_id}.out` and `{job_name}_{job_id}.err`
- **Auto-cleanup**: Use `manage_logs.py` to prevent bloat

## Results Location
- **Data**: `results/data/`
- **Plots**: `results/plots/`
- **Reports**: `results/reports/`
- **SLURM Logs**: `slurm/logs/`