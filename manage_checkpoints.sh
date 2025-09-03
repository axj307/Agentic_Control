#!/bin/bash
# Checkpoint Management Script
# ===========================
#
# Convenient wrapper for checkpoint operations.
# Usage:
#     ./manage_checkpoints.sh status      # Show current status
#     ./manage_checkpoints.sh cleanup 3   # Keep only latest 3 checkpoints  
#     ./manage_checkpoints.sh setup       # Setup checkpoint directory

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKPOINT_SCRIPT="$SCRIPT_DIR/05_training/checkpoint_config.py"

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate agentic_control

case "$1" in
    "status")
        echo "📊 Checking checkpoint status..."
        python "$CHECKPOINT_SCRIPT" --status
        ;;
    "cleanup")
        KEEP=${2:-3}
        echo "🧹 Cleaning up checkpoints (keeping latest $KEEP)..."
        python "$CHECKPOINT_SCRIPT" --cleanup "$KEEP"
        ;;
    "setup")
        echo "⚙️  Setting up checkpoint directory..."
        python "$CHECKPOINT_SCRIPT" --setup
        ;;
    *)
        echo "Usage: $0 {status|cleanup [N]|setup}"
        echo ""
        echo "Examples:"
        echo "  $0 status           # Show checkpoint status"
        echo "  $0 cleanup 3        # Keep latest 3 checkpoints"
        echo "  $0 setup            # Setup checkpoint directory"
        exit 1
        ;;
esac