# Results Directory

This directory contains experimental results from the agentic control pipeline.

## Structure

```
results/
├── data/           # Trajectory data and analysis results (JSON)
├── plots/          # Generated visualizations (PNG)
├── reports/        # Analysis reports (Markdown)
└── archive_*/      # Archived old results (auto-generated)
```

## File Naming Conventions

### Current Standard Format
- **Trajectories**: `pd_vs_tool_trajectories_YYYYMMDD_HHMMSS.json`
- **Plots**: `pd_vs_tool_comparison_YYYYMMDD_HHMMSS.png`
- **Reports**: `pd_vs_tool_analysis_YYYYMMDD_HHMMSS.md`
- **ART Analysis**: `simple_art_analysis_YYYYMMDD_HHMMSS.json`

### Legacy Formats (auto-archived)
- `comparison_results_*.json` → archived
- `enhanced_comparison_*.png` → archived  
- `experiment_summary_*.md` → archived

## Automatic Management

Results are automatically managed using `clean_results.py`:

```bash
# Show current status
python clean_results.py --status

# Keep latest 3 files per category (recommended)
python clean_results.py --keep-latest 3

# Preview what would be cleaned
python clean_results.py --dry-run

# Auto cleanup (non-interactive)
python clean_results.py --auto-clean
```

## Storage Guidelines

- **Keep Latest**: 3-5 results per experiment type
- **Archive Old**: Results are moved to `archive_*/` directories
- **Cleanup Frequency**: Run cleanup after major experiments
- **Backup**: Archives are kept for recovery if needed

## Result Types

### PD vs Tool Comparison
- Compares PD controller baseline with tool-augmented LLM
- Contains trajectory data, performance metrics, and visualizations

### ART Training Analysis  
- Analysis of Actor-Residual Training experiments
- Includes reward calculations and model evaluations

### Legacy Results
- Historical experiment formats (automatically archived)
- Kept for reference but not actively maintained