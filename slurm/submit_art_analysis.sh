#!/bin/bash
# Submit ART training analysis to SLURM
# Usage: ./submit_art_analysis.sh

echo "🚀 Submitting ART Training Analysis"
echo ""

# Submit job
JOB_ID=$(sbatch --parsable slurm/run_art_analysis.sh)

if [ $? -eq 0 ]; then
    echo "✅ Job submitted successfully!"
    echo "🆔 Job ID: $JOB_ID"
    echo ""
    echo "📋 Monitor progress with:"
    echo "   squeue -u $USER"
    echo "   tail -f slurm/logs/art_analysis_${JOB_ID}.out"
    echo ""
    echo "📁 Results will be saved to:"
    echo "   results/reports/simple_art_analysis_*.md"
    echo "   results/data/simple_art_analysis_*.json"
else
    echo "❌ Job submission failed!"
    exit 1
fi