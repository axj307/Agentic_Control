#!/usr/bin/env python3
"""
Advanced Results Management System
==================================

Automatically manages experimental results to prevent storage bloat:
- Keeps only the latest N results per type
- Archives older results in compressed format  
- Maintains clean directory structure
- Provides comprehensive status reporting
- Smart file categorization and cleanup

Usage:
    python clean_results.py                    # Interactive cleanup
    python clean_results.py --status           # Show current status
    python clean_results.py --keep-latest 3    # Keep 3 latest per group
    python clean_results.py --auto-clean       # Non-interactive cleanup
"""

import os
import shutil
import zipfile
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import json

def get_file_size_mb(file_path: Path) -> float:
    """Get file size in MB."""
    return file_path.stat().st_size / (1024 * 1024)

def extract_timestamp_from_filename(filename: str) -> datetime:
    """Extract timestamp from filename pattern."""
    import re
    pattern = r'(\d{8}_\d{6})'
    match = re.search(pattern, filename)
    
    if match:
        try:
            return datetime.strptime(match.group(1), '%Y%m%d_%H%M%S')
        except ValueError:
            pass
    
    # Fallback to current time for files without timestamp
    return datetime.now()

def categorize_results_files(results_dir: Path) -> Dict[str, List[Tuple[Path, datetime]]]:
    """Categorize and timestamp all results files."""
    categories = {
        'pd_vs_tool_trajectories': [],
        'pd_vs_tool_comparison': [], 
        'pd_vs_tool_analysis': [],
        'comparison_results': [],
        'art_analysis': [],
        'enhanced_plots': [],
        'experiment_reports': [],
        'other': []
    }
    
    # Scan all subdirectories
    for subdir in ['data', 'plots', 'reports']:
        subdir_path = results_dir / subdir
        if not subdir_path.exists():
            continue
            
        for file_path in subdir_path.glob('*'):
            if file_path.is_file():
                filename = file_path.name
                timestamp = extract_timestamp_from_filename(filename)
                
                # Categorize based on filename patterns
                if 'pd_vs_tool_trajectories' in filename:
                    categories['pd_vs_tool_trajectories'].append((file_path, timestamp))
                elif 'pd_vs_tool_comparison' in filename:
                    categories['pd_vs_tool_comparison'].append((file_path, timestamp))
                elif 'pd_vs_tool_analysis' in filename:
                    categories['pd_vs_tool_analysis'].append((file_path, timestamp))
                elif 'comparison_results' in filename:
                    categories['comparison_results'].append((file_path, timestamp))
                elif 'art_analysis' in filename or 'simple_art' in filename:
                    categories['art_analysis'].append((file_path, timestamp))
                elif 'enhanced_comparison' in filename:
                    categories['enhanced_plots'].append((file_path, timestamp))
                elif 'experiment_summary' in filename or 'comparison_report' in filename:
                    categories['experiment_reports'].append((file_path, timestamp))
                else:
                    categories['other'].append((file_path, timestamp))
    
    # Sort each category by timestamp (newest first)
    for category in categories.values():
        category.sort(key=lambda x: x[1], reverse=True)
    
    return categories

def print_results_status(results_dir: Path):
    """Print detailed status of results directory."""
    if not results_dir.exists():
        print("❌ Results directory not found!")
        return
    
    categories = categorize_results_files(results_dir)
    
    print("📊 RESULTS DIRECTORY STATUS")
    print("=" * 60)
    
    total_files = 0
    total_size_mb = 0
    
    for category_name, files in categories.items():
        if not files:
            continue
            
        count = len(files)
        size_mb = sum(get_file_size_mb(f[0]) for f in files)
        latest = files[0][1].strftime('%Y-%m-%d %H:%M:%S') if files else 'None'
        
        print(f"📁 {category_name:25}: {count:3} files, {size_mb:6.1f} MB, latest: {latest}")
        
        total_files += count
        total_size_mb += size_mb
    
    print("-" * 60)
    print(f"📈 TOTAL: {total_files:3} files, {total_size_mb:6.1f} MB")
    
    # Show directory breakdown
    print(f"\n📂 DIRECTORY BREAKDOWN:")
    for subdir in ['data', 'plots', 'reports']:
        subdir_path = results_dir / subdir
        if subdir_path.exists():
            files = list(subdir_path.glob('*'))
            size_mb = sum(get_file_size_mb(f) for f in files if f.is_file())
            print(f"   {subdir:10}: {len(files):3} files, {size_mb:6.1f} MB")
    
    # Check for archives
    archives = list(results_dir.glob('archive_*'))
    if archives:
        print(f"\n📦 ARCHIVES: {len(archives)} archive directories found")
        for archive in archives[-3:]:  # Show latest 3 archives
            files = list(archive.rglob('*'))
            file_count = len([f for f in files if f.is_file()])
            print(f"   {archive.name}: {file_count} files")

def smart_cleanup(results_dir: Path, keep_latest: int = 3, dry_run: bool = False):
    """Intelligent cleanup keeping latest N files per category."""
    categories = categorize_results_files(results_dir)
    
    print(f"🧹 SMART CLEANUP (keeping latest {keep_latest} per category)")
    print("=" * 60)
    
    if dry_run:
        print("🔍 DRY RUN MODE - showing what would be cleaned")
        print()
    
    # Create archive directory
    archive_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir = results_dir / f"archive_{archive_timestamp}"
    
    files_to_remove = []
    total_saved_mb = 0
    
    for category_name, files in categories.items():
        if len(files) <= keep_latest:
            print(f"✅ {category_name:25}: {len(files)} files (no cleanup needed)")
            continue
        
        old_files = files[keep_latest:]
        category_size_mb = sum(get_file_size_mb(f[0]) for f in old_files)
        
        print(f"🗑️  {category_name:25}: removing {len(old_files)} old files ({category_size_mb:.1f} MB)")
        
        for file_path, timestamp in old_files:
            files_to_remove.append(file_path)
            total_saved_mb += get_file_size_mb(file_path)
            print(f"     - {file_path.name}")
    
    print(f"\n💾 TOTAL SPACE TO SAVE: {total_saved_mb:.1f} MB")
    
    if not files_to_remove:
        print("✨ No cleanup needed - directory is already optimal!")
        return
    
    if dry_run:
        print("\n🔍 This was a dry run. Remove --dry-run to execute cleanup.")
        return
    
    # Actually remove files
    if not archive_dir.exists():
        archive_dir.mkdir()
        for subdir in ['data', 'plots', 'reports']:
            (archive_dir / subdir).mkdir()
    
    for file_path in files_to_remove:
        # Move to archive instead of deleting
        relative_path = file_path.relative_to(results_dir)
        archive_path = archive_dir / relative_path
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(file_path), str(archive_path))
    
    print(f"\n✅ Cleanup completed!")
    print(f"📦 Old files archived to: {archive_dir.name}")

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
    parser = argparse.ArgumentParser(
        description="Advanced Results Management System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python clean_results.py --status              # Show current status  
    python clean_results.py --keep-latest 3       # Keep 3 latest per type
    python clean_results.py --dry-run             # Preview what would be cleaned
    python clean_results.py --auto-clean          # Non-interactive cleanup
        """
    )
    
    parser.add_argument("--status", action="store_true", 
                       help="Show detailed status of results directory")
    parser.add_argument("--keep-latest", type=int, default=3,
                       help="Number of latest files to keep per category (default: 3)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be cleaned without making changes") 
    parser.add_argument("--auto-clean", action="store_true",
                       help="Run automatic cleanup without interactive prompts")
    
    args = parser.parse_args()
    
    results_dir = Path("results")
    
    if args.status:
        print_results_status(results_dir)
    elif args.auto_clean:
        print("🤖 AUTOMATIC CLEANUP MODE")
        print("=" * 50)
        smart_cleanup(results_dir, keep_latest=args.keep_latest, dry_run=args.dry_run)
        if not args.dry_run:
            clean_logs_directory()
    else:
        # Interactive mode
        print("🧹 ADVANCED RESULTS MANAGEMENT SYSTEM")
        print("=" * 50)
        
        print_results_status(results_dir)
        
        if not results_dir.exists():
            print("❌ No results directory found. Nothing to clean.")
            exit(0)
            
        print(f"\n🤔 Proceed with smart cleanup (keep latest {args.keep_latest} per category)?")
        
        if args.dry_run:
            print("🔍 Running in DRY RUN mode - no files will be modified")
            smart_cleanup(results_dir, keep_latest=args.keep_latest, dry_run=True)
        else:
            response = input("   Continue? (y/N): ")
            if response.lower() in ['y', 'yes']:
                smart_cleanup(results_dir, keep_latest=args.keep_latest)
                clean_logs_directory()
                print(f"\n🎉 Cleanup completed! Directory is optimized.")
            else:
                print("❌ Cleanup cancelled.")