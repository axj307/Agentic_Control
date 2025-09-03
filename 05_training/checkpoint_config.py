#!/usr/bin/env python3
"""
Checkpoint Configuration for ART Training
=========================================

Configures HuggingFace transformers to save checkpoints to /scratch directory
instead of home directory to avoid filling up the 200GB home quota.

This module should be imported at the beginning of any training script that
uses HuggingFace transformers checkpointing.
"""

import os
import getpass
from pathlib import Path

def setup_checkpoint_directory():
    """Setup checkpoint directory on /scratch and configure transformers."""
    
    # Get current user
    username = getpass.getuser()
    
    # Create checkpoint directory on scratch
    scratch_checkpoints = Path(f"/scratch/{username}/agentic_control_checkpoints")
    scratch_checkpoints.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Checkpoint directory: {scratch_checkpoints}")
    
    # Configure transformers to use scratch directory
    try:
        import transformers
        from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
        
        # Store original prefix for reference
        original_prefix = PREFIX_CHECKPOINT_DIR
        
        # Set custom checkpoint prefix to point to scratch
        transformers.trainer_utils.PREFIX_CHECKPOINT_DIR = str(scratch_checkpoints) + "/" + PREFIX_CHECKPOINT_DIR
        
        print(f"✅ Transformers checkpoint prefix configured:")
        print(f"   Original: {original_prefix}")
        print(f"   New: {transformers.trainer_utils.PREFIX_CHECKPOINT_DIR}")
        
        return str(scratch_checkpoints)
        
    except ImportError:
        print("⚠️  Transformers not available - checkpoint configuration skipped")
        return str(scratch_checkpoints)

def get_checkpoint_dir():
    """Get the configured checkpoint directory path."""
    username = getpass.getuser()
    return f"/scratch/{username}/agentic_control_checkpoints"

def cleanup_old_checkpoints(keep_latest=3):
    """Clean up old checkpoints, keeping only the latest N."""
    checkpoint_dir = Path(get_checkpoint_dir())
    
    if not checkpoint_dir.exists():
        print("📁 No checkpoint directory found")
        return
    
    # Find all checkpoint directories
    checkpoints = []
    for item in checkpoint_dir.iterdir():
        if item.is_dir() and item.name.startswith("checkpoint"):
            checkpoints.append(item)
    
    if len(checkpoints) <= keep_latest:
        print(f"✅ Only {len(checkpoints)} checkpoints found - no cleanup needed")
        return
    
    # Sort by modification time (newest first)
    checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    # Remove old checkpoints
    old_checkpoints = checkpoints[keep_latest:]
    total_size_mb = 0
    
    for checkpoint in old_checkpoints:
        size_mb = sum(f.stat().st_size for f in checkpoint.rglob('*') if f.is_file()) / (1024 * 1024)
        total_size_mb += size_mb
        
        print(f"🗑️  Removing old checkpoint: {checkpoint.name} ({size_mb:.1f} MB)")
        
        # Remove the checkpoint directory
        import shutil
        shutil.rmtree(checkpoint)
    
    print(f"💾 Freed up {total_size_mb:.1f} MB of storage")

def get_checkpoint_status():
    """Get status of current checkpoints."""
    checkpoint_dir = Path(get_checkpoint_dir())
    
    if not checkpoint_dir.exists():
        return {
            "directory": str(checkpoint_dir),
            "exists": False,
            "checkpoints": 0,
            "total_size_mb": 0
        }
    
    # Count checkpoints and calculate size
    checkpoints = [d for d in checkpoint_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint")]
    total_size = sum(f.stat().st_size for d in checkpoints for f in d.rglob('*') if f.is_file())
    
    return {
        "directory": str(checkpoint_dir),
        "exists": True,
        "checkpoints": len(checkpoints),
        "total_size_mb": total_size / (1024 * 1024),
        "checkpoint_names": [d.name for d in checkpoints]
    }

def print_checkpoint_status():
    """Print formatted checkpoint status."""
    status = get_checkpoint_status()
    
    print("📊 CHECKPOINT STATUS")
    print("=" * 50)
    print(f"Directory: {status['directory']}")
    print(f"Exists: {'✅ Yes' if status['exists'] else '❌ No'}")
    
    if status['exists']:
        print(f"Checkpoints: {status['checkpoints']}")
        print(f"Total Size: {status['total_size_mb']:.1f} MB")
        
        if status['checkpoints'] > 0:
            print(f"Checkpoint Names:")
            for name in status['checkpoint_names']:
                print(f"  - {name}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Checkpoint management utilities")
    parser.add_argument("--setup", action="store_true", help="Setup checkpoint directory")
    parser.add_argument("--status", action="store_true", help="Show checkpoint status") 
    parser.add_argument("--cleanup", type=int, metavar="N", help="Keep only latest N checkpoints")
    
    args = parser.parse_args()
    
    if args.setup:
        setup_checkpoint_directory()
    elif args.status:
        print_checkpoint_status()
    elif args.cleanup:
        cleanup_old_checkpoints(keep_latest=args.cleanup)
    else:
        print("Use --setup, --status, or --cleanup N")
        print("Example: python checkpoint_config.py --setup")