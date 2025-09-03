#!/bin/bash
# Submit clean agentic control experiments to SLURM
# Usage: ./submit_clean_experiments.sh [difficulty] [save_plots]

DIFFICULTY=${1:-"all"}    # easy, medium, hard, all
SAVE_PLOTS=${2:-"true"}   # true, false

echo "🚀 Submitting Clean Agentic Control Experiments"
echo "📋 Difficulty: $DIFFICULTY"
echo "📊 Save Plots: $SAVE_PLOTS"
echo ""

# Submit job
JOB_ID=$(sbatch --parsable slurm/run_clean_experiments.sh $DIFFICULTY $SAVE_PLOTS)

if [ $? -eq 0 ]; then
    echo "✅ Job submitted successfully!"
    echo "🆔 Job ID: $JOB_ID"
    echo ""
    echo "📋 Monitor progress with:"
    echo "   squeue -u $USER"
    echo "   tail -f slurm/logs/clean_experiments_${JOB_ID}.out"
    echo ""
    echo "📁 Results will be saved to:"
    echo "   results/plots/pd_vs_tool_comparison_*.png"
    echo "   results/data/pd_vs_tool_trajectories_*.json"
    echo "   results/reports/pd_vs_tool_analysis_*.md"
else
    echo "❌ Job submission failed!"
    exit 1
fi