#!/usr/bin/env python3
"""
Clean Results Directory
Organizes and archives old results, keeps only latest standardized files
"""

import os
import shutil
from pathlib import Path
from datetime import datetime

def clean_results_directory():
    """Clean and organize the results directory"""
    results_dir = Path("results")
    
    if not results_dir.exists():
        print("No results directory found.")
        return
    
    print("🧹 Cleaning results directory...")
    
    # Create archive directory with timestamp
    archive_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir = results_dir / f"archive_{archive_timestamp}"
    archive_dir.mkdir(exist_ok=True)
    
    # Create clean structure
    clean_dirs = ["data", "plots", "reports"]
    for dir_name in clean_dirs:
        (results_dir / dir_name).mkdir(exist_ok=True)
        (archive_dir / dir_name).mkdir(exist_ok=True)
    
    print(f"📁 Created archive: {archive_dir}")
    
    # File patterns to keep (latest standardized format)
    keep_patterns = {
        "data": "pd_vs_tool_trajectories_*.json",
        "plots": "pd_vs_tool_comparison_*.png", 
        "reports": "pd_vs_tool_analysis_*.md"
    }
    
    # File patterns to archive (old formats)
    archive_patterns = {
        "data": [
            "comparison_results_*.json",
            "trajectories_*.json", 
            "performance_*.json"
        ],
        "plots": [
            "enhanced_comparison_*.png",
            "control_analysis_*.png",
            "pd_vs_tool_augmented_comparison.png",
            "performance_summary_*.png",
            "trajectory_*.png"
        ],
        "reports": [
            "comparison_report_*.md",
            "experiment_summary_*.md",
            "experiment_report_*.md"
        ]
    }
    
    files_kept = 0
    files_archived = 0
    
    # Process each directory
    for dir_name in clean_dirs:
        source_dir = results_dir / dir_name
        archive_subdir = archive_dir / dir_name
        
        if not source_dir.exists():
            continue
            
        print(f"\n📂 Processing {dir_name}/")
        
        # Get all files in directory
        all_files = list(source_dir.glob("*"))
        
        # Find files to keep (latest standardized format)
        keep_files = list(source_dir.glob(keep_patterns[dir_name]))
        keep_latest = None
        if keep_files:
            # Keep only the most recent file
            keep_latest = max(keep_files, key=lambda x: x.stat().st_mtime)
            print(f"   ✅ Keeping latest: {keep_latest.name}")
            files_kept += 1
        
        # Archive everything else
        for file_path in all_files:
            if file_path.is_file() and (not keep_latest or file_path != keep_latest):
                dest_path = archive_subdir / file_path.name
                shutil.move(str(file_path), str(dest_path))
                print(f"   📦 Archived: {file_path.name}")
                files_archived += 1
    
    print(f"\n✅ Cleanup completed!")
    print(f"   📊 Files kept: {files_kept}")
    print(f"   📦 Files archived: {files_archived}")
    print(f"   📁 Archive location: {archive_dir}")
    
    # Show current clean structure
    print(f"\n📋 Clean results structure:")
    for dir_name in clean_dirs:
        dir_path = results_dir / dir_name
        files = list(dir_path.glob("*"))
        if files:
            print(f"   {dir_name}/")
            for file_path in files:
                print(f"     {file_path.name}")
        else:
            print(f"   {dir_name}/ (empty - ready for new results)")

def clean_logs_directory():
    """Clean old log files"""
    logs_dir = Path("logs")
    
    if not logs_dir.exists():
        print("No logs directory found.")
        return
        
    print("\n🧹 Cleaning logs directory...")
    
    log_files = list(logs_dir.glob("*.out")) + list(logs_dir.glob("*.err"))
    
    if len(log_files) > 10:  # Keep only latest 10 log files
        log_files.sort(key=lambda x: x.stat().st_mtime)
        old_logs = log_files[:-10]
        
        # Create logs archive
        archive_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        logs_archive = logs_dir / f"archive_{archive_timestamp}"
        logs_archive.mkdir(exist_ok=True)
        
        for log_file in old_logs:
            dest_path = logs_archive / log_file.name
            shutil.move(str(log_file), str(dest_path))
            
        print(f"   📦 Archived {len(old_logs)} old log files to {logs_archive}")
        print(f"   ✅ Kept {len(log_files) - len(old_logs)} recent log files")
    else:
        print("   ✅ Log directory already clean")

if __name__ == "__main__":
    print("🧹 Results Directory Cleanup Tool")
    print("=" * 50)
    
    # Show current state
    results_dir = Path("results")
    if results_dir.exists():
        total_files = sum(len(list(subdir.glob("*"))) for subdir in results_dir.iterdir() if subdir.is_dir())
        print(f"📊 Current state: {total_files} files in results/")
    
    # Confirm cleanup
    response = input("\n🤔 Proceed with cleanup? This will archive old files (y/N): ")
    
    if response.lower() in ['y', 'yes']:
        clean_results_directory()
        clean_logs_directory()
        print(f"\n🎉 Cleanup completed! Your results directory is now organized.")
        print(f"💡 Old files are safely archived and can be restored if needed.")
    else:
        print("❌ Cleanup cancelled.")