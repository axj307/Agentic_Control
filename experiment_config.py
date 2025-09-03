"""
Experiment Configuration
Centralized configuration for all experiments
"""

import os
from pathlib import Path

# Project paths (dynamic)
PROJECT_ROOT = Path(__file__).parent.absolute()
RESULTS_DIR = PROJECT_ROOT / "results"
SLURM_DIR = PROJECT_ROOT / "slurm"

# Module paths
PHYSICS_DIR = PROJECT_ROOT / "01_basic_physics"
CONTROL_DIR = PROJECT_ROOT / "02_direct_control"
TOOLS_DIR = PROJECT_ROOT / "03_langgraph_tools"

# Experiment settings
DEFAULT_CONFIG = {
    # Environment settings
    "dt": 0.1,
    "max_force": 1.0,
    "max_steps": 100,
    "position_tolerance": 0.1,
    "velocity_tolerance": 0.1,
    
    # Controller settings
    "pd_gains": {"kp": 1.0, "kd": 2.0},
    
    # Scenario settings
    "test_scenarios": {
        "easy": [
            {"name": "small_pos_right", "init_pos": 0.5, "init_vel": 0.0, "target_pos": 0.0, "target_vel": 0.0},
            {"name": "small_pos_left", "init_pos": -0.3, "init_vel": 0.0, "target_pos": 0.0, "target_vel": 0.0},
            {"name": "small_vel", "init_pos": 0.0, "init_vel": 0.2, "target_pos": 0.0, "target_vel": 0.0},
        ],
        "medium": [
            {"name": "med_pos_right", "init_pos": 1.0, "init_vel": 0.0, "target_pos": 0.0, "target_vel": 0.0},
            {"name": "med_vel", "init_pos": 0.0, "init_vel": 0.5, "target_pos": 0.0, "target_vel": 0.0},
            {"name": "mixed_errors", "init_pos": 0.8, "init_vel": -0.3, "target_pos": 0.0, "target_vel": 0.0},
        ],
        "hard": [
            {"name": "large_pos_right", "init_pos": 1.5, "init_vel": 0.0, "target_pos": 0.0, "target_vel": 0.0},
            {"name": "large_mixed", "init_pos": -1.2, "init_vel": 0.8, "target_pos": 0.0, "target_vel": 0.0},
            {"name": "high_vel", "init_pos": 0.5, "init_vel": -0.9, "target_pos": 0.0, "target_vel": 0.0},
        ]
    },
    
    # Plotting settings
    "plot_config": {
        "figsize": (16, 12),
        "fontsize_title": 20,
        "fontsize_label": 16,
        "fontsize_tick": 14,
        "linewidth": 3.0,
        "dpi": 300,
        "colors": {
            "pd": "#1f77b4",      # blue
            "tool": "#ff7f0e",    # orange
            "target": "#2ca02c",  # green
            "limits": "#d62728"   # red
        }
    }
}

def get_project_paths():
    """Get all project paths as a dictionary"""
    return {
        "root": PROJECT_ROOT,
        "results": RESULTS_DIR,
        "slurm": SLURM_DIR,
        "physics": PHYSICS_DIR,
        "control": CONTROL_DIR,
        "tools": TOOLS_DIR,
    }

def setup_python_path():
    """Add all module directories to Python path"""
    import sys
    paths = get_project_paths()
    for key in ["root", "physics", "control", "tools"]:
        path_str = str(paths[key])
        if path_str not in sys.path:
            sys.path.append(path_str)

def create_results_directories():
    """Create all necessary results directories"""
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "plots").mkdir(exist_ok=True)
    (RESULTS_DIR / "data").mkdir(exist_ok=True)
    (RESULTS_DIR / "reports").mkdir(exist_ok=True)

def get_scenarios(difficulty="all"):
    """Get scenarios based on difficulty level"""
    scenarios = DEFAULT_CONFIG["test_scenarios"]
    
    if difficulty == "all":
        all_scenarios = []
        for level_scenarios in scenarios.values():
            all_scenarios.extend(level_scenarios)
        return all_scenarios
    elif difficulty in scenarios:
        return scenarios[difficulty]
    else:
        raise ValueError(f"Unknown difficulty level: {difficulty}")

def get_latest_results_file(pattern="pd_vs_tool_trajectories_*.json"):
    """Get the latest results file matching pattern"""
    data_dir = RESULTS_DIR / "data"
    files = list(data_dir.glob(pattern))
    if not files:
        return None
    return max(files, key=lambda x: x.stat().st_mtime)

def cleanup_old_results(keep_latest=3):
    """Clean up old result files, keeping only the most recent ones"""
    import shutil
    from datetime import datetime
    
    # Create archive directory
    archive_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir = RESULTS_DIR / f"archive_{archive_timestamp}"
    
    patterns = {
        "data": "pd_vs_tool_trajectories_*.json",
        "plots": "pd_vs_tool_comparison_*.png", 
        "reports": "pd_vs_tool_analysis_*.md"
    }
    
    archived_count = 0
    
    for subdir, pattern in patterns.items():
        dir_path = RESULTS_DIR / subdir
        if not dir_path.exists():
            continue
            
        files = list(dir_path.glob(pattern))
        if len(files) <= keep_latest:
            continue
            
        # Sort by modification time, keep most recent
        files.sort(key=lambda x: x.stat().st_mtime)
        old_files = files[:-keep_latest]
        
        if old_files:
            # Create archive subdirectory
            (archive_dir / subdir).mkdir(parents=True, exist_ok=True)
            
            for old_file in old_files:
                dest_path = archive_dir / subdir / old_file.name
                shutil.move(str(old_file), str(dest_path))
                archived_count += 1
    
    if archived_count > 0:
        print(f"🧹 Archived {archived_count} old files to {archive_dir}")
    
    return archived_count

if __name__ == "__main__":
    print("🔧 Experiment Configuration")
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Results Dir: {RESULTS_DIR}")
    print(f"Available scenarios: {len(get_scenarios())} total")
    for difficulty in ["easy", "medium", "hard"]:
        count = len(get_scenarios(difficulty))
        print(f"  {difficulty}: {count} scenarios")