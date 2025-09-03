#!/bin/bash
#SBATCH --job-name=art_analysis
#SBATCH --output=logs/art_analysis_%j.out
#SBATCH --error=logs/art_analysis_%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=pi_linaresr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

echo "🚀 Starting ART Training Analysis..."
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
echo "=== ART TRAINING ANALYSIS ==="
echo "🎯 System: Double Integrator Control"
echo "🤖 Analysis: Performance evaluation and training recommendations"
echo "📊 Output: ART training projections and configuration"
echo ""

# Create output directories
mkdir -p logs
mkdir -p results/data
mkdir -p results/reports

# Test configuration first
echo "🧪 Testing ART analysis configuration..."
python -c "
from experiment_config import get_latest_results_file
latest = get_latest_results_file('pd_vs_tool_trajectories_*.json')
if latest:
    print(f'✅ Found trajectory data: {latest.name}')
else:
    print('❌ No trajectory data found')
    exit(1)
"

config_test_exit_code=$?
if [ $config_test_exit_code -ne 0 ]; then
    echo "❌ Configuration test failed, aborting..."
    exit 1
fi

# Run ART analysis
echo ""
echo "🎯 Running ART Training Analysis..."

python simple_art_trainer.py

analysis_exit_code=$?

# Check results and provide summary
if [ $analysis_exit_code -eq 0 ]; then
    echo ""
    echo "✅ ART ANALYSIS COMPLETED SUCCESSFULLY!"
    echo ""
    echo "📊 Results Summary:"
    echo "🔧 Total execution time: $(($SECONDS / 60)) minutes"
    
    # Show generated files
    echo ""
    echo "📁 Generated Files:"
    if [ -d "results" ]; then
        echo "   📝 Analysis Reports:"
        find results/reports -name "simple_art_analysis_*.md" -exec echo "      {}" \; 2>/dev/null
        
        echo "   💾 Analysis Data:"
        find results/data -name "simple_art_analysis_*.json" -exec echo "      {}" \; 2>/dev/null
        
        # Show latest report summary
        latest_report=$(find results/reports -name "simple_art_analysis_*.md" | sort | tail -1)
        if [ -f "$latest_report" ]; then
            echo ""
            echo "📋 Latest Analysis Summary:"
            echo "$(grep -A 5 'Key Findings:' "$latest_report" 2>/dev/null || grep -A 5 'Expected Improvements:' "$latest_report")"
        fi
    fi
    
    echo ""
    echo "🎯 ART Training Insights:"
    echo "  📈 Performance analysis complete with training recommendations"
    echo "  🏆 Reward optimization strategy identified"
    echo "  💾 Training configuration ready for implementation"
    echo "  📝 Comprehensive report available for review"
    
else
    echo ""
    echo "❌ ART ANALYSIS FAILED!"
    echo "📋 Check error logs above and in logs/art_analysis_$SLURM_JOB_ID.err"
fi

echo ""
echo "End time: $(date)"
echo "=" | tr -d '\n' | head -c 80; echo

exit $analysis_exit_code