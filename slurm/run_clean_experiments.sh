#!/bin/bash
#SBATCH --job-name=agentic_control_clean
#SBATCH --output=slurm/logs/clean_experiments_%j.out
#SBATCH --error=slurm/logs/clean_experiments_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=pi_linaresr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

echo "🚀 Starting Clean Agentic Control Experiments..."
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Node: $(hostname)"

# Get the project root directory dynamically
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "Project root: $PROJECT_ROOT"
cd "$PROJECT_ROOT"

# Load environment
source ~/.bashrc
conda activate agentic_control

echo "Environment activated"
echo "Python path: $(which python)"
echo "Conda environment: $CONDA_DEFAULT_ENV"

echo ""
echo "=== CLEAN AGENTIC CONTROL PIPELINE ==="
echo "🎯 System: Double Integrator Control"
echo "🤖 Controllers: PD Baseline vs Tool-Augmented"
echo "📊 Enhanced visualization with professional plots"
echo ""

# Parse command line arguments with defaults
DIFFICULTY=${1:-"all"}     # easy, medium, hard, all
SAVE_PLOTS=${2:-"true"}    # true, false

echo "📋 Experiment Configuration:"
echo "   Difficulty: $DIFFICULTY"
echo "   Save Plots: $SAVE_PLOTS"
echo ""

# Create output directories
mkdir -p slurm/logs
mkdir -p results/plots
mkdir -p results/data
mkdir -p results/reports

# Test configuration first
echo "🧪 Testing clean pipeline configuration..."
python -c "
from experiment_config import get_scenarios, get_project_paths
print(f'✅ Configuration loaded: {len(get_scenarios())} total scenarios')
print(f'✅ Project paths: {get_project_paths()[\"root\"]}')
"

config_test_exit_code=$?
if [ $config_test_exit_code -ne 0 ]; then
    echo "❌ Configuration test failed, aborting..."
    exit 1
fi

# Run the clean experiments
echo ""
echo "🚀 Running Clean Experiments..."

python run_experiments.py \
    --difficulty $DIFFICULTY \
    $(if [ "$SAVE_PLOTS" = "true" ]; then echo "--save-plots"; fi)

experiment_exit_code=$?

# Check results and provide summary
if [ $experiment_exit_code -eq 0 ]; then
    echo ""
    echo "✅ CLEAN EXPERIMENTS COMPLETED SUCCESSFULLY!"
    echo ""
    echo "📊 Results Summary:"
    echo "🔧 Total execution time: $(($SECONDS / 60)) minutes"
    
    # Show generated files
    echo ""
    echo "📁 Generated Files:"
    if [ -d "results" ]; then
        echo "   📈 PD vs Tool Comparison Plots:"
        find results/plots -name "pd_vs_tool_comparison_*.png" -exec echo "      {}" \; 2>/dev/null
        
        echo "   💾 Trajectory Data:"
        find results/data -name "pd_vs_tool_trajectories_*.json" -exec echo "      {}" \; 2>/dev/null
        
        echo "   📝 Analysis Reports:"
        find results/reports -name "pd_vs_tool_analysis_*.md" -exec echo "      {}" \; 2>/dev/null
        
        # Show latest report summary
        latest_report=$(find results/reports -name "pd_vs_tool_analysis_*.md" | sort | tail -1)
        if [ -f "$latest_report" ]; then
            echo ""
            echo "📋 Latest Results Summary:"
            echo "$(tail -10 "$latest_report")"
        fi
    fi
    
    echo ""
    echo "🎯 Key Features of Clean Pipeline:"
    echo "  🔧 No hard-coded paths - fully portable"
    echo "  📊 Enhanced visualizations with professional styling"
    echo "  💾 Automated data saving and reporting"
    echo "  📝 Comprehensive markdown reports"
    echo "  🚀 Ready for scaling to larger experiments"
    
else
    echo ""
    echo "❌ CLEAN EXPERIMENTS FAILED!"
    echo "📋 Check error logs above and in slurm/logs/clean_experiments_$SLURM_JOB_ID.err"
fi

echo ""
echo "End time: $(date)"
echo "=" | tr -d '\n' | head -c 80; echo

exit $experiment_exit_code