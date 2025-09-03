# Checkpoint Management for Multiple Worktrees

This guide shows how to implement checkpoint management that prevents filling up your home directory quota while supporting multiple worktrees/projects.

## Problem

By default, HuggingFace transformers save checkpoints to paths within your home directory, which can quickly fill up limited quotas (e.g., 200GB). When working with multiple worktrees or projects, you need separate checkpoint directories to avoid conflicts.

## Solution Overview

1. **Auto-detect worktree/project**: Create unique checkpoint directories per project
2. **Use /scratch directory**: Store checkpoints on scratch storage with larger quota
3. **Configure before imports**: Set HuggingFace checkpoint prefix before any transformers imports
4. **Provide management tools**: Scripts for status, cleanup, and maintenance

## Implementation

### Step 1: Create `checkpoint_config.py`

```python
#!/usr/bin/env python3
"""
Checkpoint Configuration for Multiple Worktrees
==============================================

Configures HuggingFace transformers to save checkpoints to /scratch directory
with unique names per worktree/project to avoid conflicts.
"""

import os
import getpass
from pathlib import Path
import subprocess

def get_worktree_identifier():
    """Get a unique identifier for the current worktree/project."""
    try:
        # Try to get git worktree info
        result = subprocess.run(['git', 'rev-parse', '--show-toplevel'], 
                              capture_output=True, text=True, cwd=os.getcwd())
        if result.returncode == 0:
            git_root = Path(result.stdout.strip())
            # Use the parent directory name + git root name for uniqueness
            parent_name = git_root.parent.name
            repo_name = git_root.name
            return f"{parent_name}_{repo_name}"
    except:
        pass
    
    # Fallback: use current directory name + parent
    current_path = Path.cwd()
    parent_name = current_path.parent.name
    current_name = current_path.name
    return f"{parent_name}_{current_name}"

def setup_checkpoint_directory(project_name=None):
    """Setup checkpoint directory on /scratch and configure transformers.
    
    Args:
        project_name: Custom project name. If None, auto-detected from worktree.
    """
    
    # Get current user
    username = getpass.getuser()
    
    # Get unique identifier for this worktree/project
    if project_name is None:
        project_name = get_worktree_identifier()
    
    # Create checkpoint directory on scratch with worktree-specific name
    scratch_checkpoints = Path(f"/scratch/{username}/checkpoints_{project_name}")
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

def get_checkpoint_dir(project_name=None):
    """Get the configured checkpoint directory path."""
    username = getpass.getuser()
    if project_name is None:
        project_name = get_worktree_identifier()
    return f"/scratch/{username}/checkpoints_{project_name}"

# Additional utility functions...
```

### Step 2: Update Training Scripts

**CRITICAL**: Import checkpoint configuration BEFORE any transformers imports:

```python
#!/usr/bin/env python3
"""
Your Training Script
"""

# Configure checkpoint directory BEFORE any transformers imports
from checkpoint_config import setup_checkpoint_directory
checkpoint_dir = setup_checkpoint_directory()

# Now safe to import transformers, ART, etc.
import transformers
import art
# ... other imports

# Your training code here...
```

### Step 3: Create Management Script

Create `manage_checkpoints.sh`:

```bash
#!/bin/bash
# Checkpoint Management Script

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKPOINT_SCRIPT="$SCRIPT_DIR/checkpoint_config.py"

# Activate your conda environment
eval "$(conda shell.bash hook)"
conda activate your_environment

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
```

### Step 4: Update .gitignore

Prevent accidental commits of checkpoint directories:

```gitignore
# Checkpoint directories (now on /scratch)
*/.art/
*/checkpoints_*/
.art/

# Large model caches
*/unsloth_compiled_cache/
**/unsloth_compiled_cache/
```

## Directory Structure

With this implementation, multiple worktrees will have separate checkpoint directories:

```
/scratch/username/
├── checkpoints_project1_main/           # Main worktree
├── checkpoints_project1_experiment_a/   # Experiment A worktree  
├── checkpoints_project1_experiment_b/   # Experiment B worktree
├── checkpoints_project2_main/           # Different project
└── archived_models_20250903/            # Manual archives
```

## Usage Examples

### Basic Usage
```bash
# Setup (run once per worktree)
python checkpoint_config.py --setup

# Check status
./manage_checkpoints.sh status

# Clean old checkpoints
./manage_checkpoints.sh cleanup 3
```

### For Multiple Worktrees
```bash
# In worktree A
cd /path/to/project/worktree-experiment-a
python checkpoint_config.py --setup
# Creates: /scratch/username/checkpoints_project_worktree-experiment-a/

# In worktree B  
cd /path/to/project/worktree-experiment-b
python checkpoint_config.py --setup
# Creates: /scratch/username/checkpoints_project_worktree-experiment-b/
```

### Custom Project Names
```python
# In your training script, specify custom name
from checkpoint_config import setup_checkpoint_directory
checkpoint_dir = setup_checkpoint_directory(project_name="my_custom_experiment")
```

### Monitoring Disk Usage
```bash
# Check all checkpoint directories
ls -la /scratch/$USER/checkpoints_*/

# Check sizes
du -sh /scratch/$USER/checkpoints_*/

# Find largest checkpoint directories
du -sh /scratch/$USER/checkpoints_*/ | sort -hr
```

## Benefits

1. **No Home Directory Bloat**: All checkpoints go to /scratch with larger quota
2. **Worktree Isolation**: Each worktree gets its own checkpoint directory
3. **Auto-Detection**: Automatically creates unique directory names
4. **Easy Management**: Simple scripts for status and cleanup
5. **Flexible**: Support for custom project names
6. **Safe**: Prevents accidental overwrites between experiments

## Migration from Existing Checkpoints

If you have existing checkpoints in home directory:

```bash
# Move existing checkpoints to scratch
cp -r ~/.cache/huggingface/transformers/ /scratch/$USER/archived_models_$(date +%Y%m%d)/

# Or move .art directories
cp -r your_project/.art/ /scratch/$USER/archived_models_$(date +%Y%m%d)/

# Remove from home (after verifying copy worked)
rm -rf ~/.cache/huggingface/transformers/
rm -rf your_project/.art/
```

## Troubleshooting

### Issue: Checkpoints Still Going to Home
**Solution**: Ensure checkpoint configuration is imported BEFORE any transformers imports.

### Issue: Permission Denied on /scratch
**Solution**: Check if /scratch/$USER directory exists and is writable.

### Issue: Different Worktrees Using Same Directory
**Solution**: Verify `get_worktree_identifier()` returns different names for different worktrees.

### Issue: Out of Space on /scratch
**Solution**: Run cleanup more frequently or decrease keep_latest parameter.

## Advanced Configuration

### Custom Checkpoint Locations
```python
# For specific clusters or storage systems
def get_cluster_storage_path():
    if os.path.exists("/lustre"):
        return "/lustre/username"
    elif os.path.exists("/gpfs"):
        return "/gpfs/username"
    else:
        return f"/scratch/{getpass.getuser()}"
```

### Automatic Cleanup Hooks
```python
# Add to training script
import atexit
from checkpoint_config import cleanup_old_checkpoints

def cleanup_on_exit():
    print("🧹 Cleaning up old checkpoints...")
    cleanup_old_checkpoints(keep_latest=2)

atexit.register(cleanup_on_exit)
```

This implementation ensures that each of your worktrees will have completely separate checkpoint directories, preventing any conflicts or overwrites between different experiments or branches.