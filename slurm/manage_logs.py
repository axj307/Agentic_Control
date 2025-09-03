#!/usr/bin/env python3
"""
SLURM Log Management Utility
===========================

Manages SLURM job logs to prevent directory bloat.
- Lists recent jobs and their logs
- Cleans up old log files
- Provides job status summaries

Usage:
    python slurm/manage_logs.py --status
    python slurm/manage_logs.py --clean --keep-days 7
    python slurm/manage_logs.py --list-jobs 10
"""

import os
import argparse
from pathlib import Path
from datetime import datetime, timedelta
import subprocess
import re

class SlurmLogManager:
    """Manages SLURM job logs in slurm/logs/ directory."""
    
    def __init__(self, logs_dir="slurm/logs"):
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
    
    def get_log_files(self):
        """Get all SLURM log files with metadata."""
        log_files = []
        
        for log_file in self.logs_dir.glob("*.out"):
            # Extract job ID from filename (e.g., clean_experiments_12345.out)
            match = re.search(r'_(\d+)\.out$', log_file.name)
            if match:
                job_id = match.group(1)
                err_file = self.logs_dir / f"{log_file.stem.replace('.out', '')}.err"
                
                log_info = {
                    'job_id': job_id,
                    'name': log_file.stem.replace(f'_{job_id}', ''),
                    'out_file': log_file,
                    'err_file': err_file if err_file.exists() else None,
                    'size_mb': log_file.stat().st_size / (1024 * 1024),
                    'modified': datetime.fromtimestamp(log_file.stat().st_mtime)
                }
                log_files.append(log_info)
        
        return sorted(log_files, key=lambda x: x['modified'], reverse=True)
    
    def get_job_status(self, job_id):
        """Get SLURM job status if available."""
        try:
            result = subprocess.run(['sacct', '-j', str(job_id), '--format=State,ExitCode,Elapsed', '-n'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0 and result.stdout.strip():
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    if not line.strip().endswith('.batch') and not line.strip().endswith('.extern'):
                        parts = line.split()
                        if len(parts) >= 3:
                            return {
                                'state': parts[0],
                                'exit_code': parts[1],
                                'elapsed': parts[2]
                            }
        except Exception:
            pass
        return None
    
    def list_recent_jobs(self, limit=10):
        """List recent SLURM jobs and their logs."""
        log_files = self.get_log_files()
        
        print("📋 RECENT SLURM JOBS")
        print("=" * 80)
        print(f"{'Job ID':<10} {'Name':<25} {'Size (MB)':<10} {'Modified':<20} {'Status':<15}")
        print("-" * 80)
        
        for log_info in log_files[:limit]:
            status_info = self.get_job_status(log_info['job_id'])
            status = status_info['state'] if status_info else 'Unknown'
            
            print(f"{log_info['job_id']:<10} {log_info['name']:<25} {log_info['size_mb']:>9.1f} "
                  f"{log_info['modified'].strftime('%Y-%m-%d %H:%M'):<20} {status:<15}")
        
        total_size = sum(log['size_mb'] for log in log_files)
        print("-" * 80)
        print(f"Total: {len(log_files)} jobs, {total_size:.1f} MB")
    
    def show_log_summary(self, job_id):
        """Show summary of a specific job's logs."""
        log_files = self.get_log_files()
        job_logs = [log for log in log_files if log['job_id'] == str(job_id)]
        
        if not job_logs:
            print(f"❌ No logs found for job {job_id}")
            return
        
        log_info = job_logs[0]
        status_info = self.get_job_status(job_id)
        
        print(f"📊 JOB {job_id} SUMMARY")
        print("=" * 50)
        print(f"Name: {log_info['name']}")
        print(f"Modified: {log_info['modified']}")
        print(f"Log Size: {log_info['size_mb']:.1f} MB")
        
        if status_info:
            print(f"Status: {status_info['state']}")
            print(f"Exit Code: {status_info['exit_code']}")
            print(f"Elapsed: {status_info['elapsed']}")
        
        print(f"\n📁 Files:")
        print(f"  Output: {log_info['out_file']}")
        if log_info['err_file']:
            print(f"  Error:  {log_info['err_file']}")
        
        # Show last few lines of output
        if log_info['out_file'].exists():
            print(f"\n📋 Last 10 lines of output:")
            try:
                with open(log_info['out_file'], 'r') as f:
                    lines = f.readlines()
                    for line in lines[-10:]:
                        print(f"  {line.rstrip()}")
            except Exception as e:
                print(f"  Error reading log: {e}")
    
    def clean_old_logs(self, keep_days=7, dry_run=False):
        """Remove log files older than specified days."""
        cutoff_date = datetime.now() - timedelta(days=keep_days)
        log_files = self.get_log_files()
        
        old_logs = [log for log in log_files if log['modified'] < cutoff_date]
        
        if not old_logs:
            print(f"✅ No logs older than {keep_days} days found")
            return
        
        print(f"🧹 CLEANING LOGS OLDER THAN {keep_days} DAYS")
        print("=" * 60)
        
        if dry_run:
            print("🔍 DRY RUN - showing what would be removed:")
        
        total_size = 0
        for log_info in old_logs:
            total_size += log_info['size_mb']
            
            if dry_run:
                print(f"  Would remove: {log_info['out_file'].name} "
                      f"({log_info['size_mb']:.1f} MB, {log_info['modified'].strftime('%Y-%m-%d')})")
            else:
                print(f"  Removing: {log_info['out_file'].name}")
                log_info['out_file'].unlink()
                if log_info['err_file'] and log_info['err_file'].exists():
                    log_info['err_file'].unlink()
        
        print(f"\n💾 Total space {'would be' if dry_run else ''} freed: {total_size:.1f} MB")
        
        if dry_run:
            print("🔍 This was a dry run. Remove --dry-run to actually delete files.")
    
    def get_status(self):
        """Get overall status of SLURM logs."""
        log_files = self.get_log_files()
        
        if not log_files:
            print("📊 SLURM LOGS STATUS: No log files found")
            return
        
        total_size = sum(log['size_mb'] for log in log_files)
        oldest = min(log['modified'] for log in log_files)
        newest = max(log['modified'] for log in log_files)
        
        print("📊 SLURM LOGS STATUS")
        print("=" * 50)
        print(f"Directory: {self.logs_dir}")
        print(f"Total Jobs: {len(log_files)}")
        print(f"Total Size: {total_size:.1f} MB")
        print(f"Date Range: {oldest.strftime('%Y-%m-%d')} to {newest.strftime('%Y-%m-%d')}")
        
        # Count by job type
        job_types = {}
        for log in log_files:
            job_type = log['name']
            job_types[job_type] = job_types.get(job_type, 0) + 1
        
        print(f"\n📋 Job Types:")
        for job_type, count in sorted(job_types.items()):
            print(f"  {job_type}: {count} jobs")

def main():
    parser = argparse.ArgumentParser(description="Manage SLURM job logs")
    parser.add_argument("--status", action="store_true", help="Show logs status")
    parser.add_argument("--list-jobs", type=int, default=10, 
                       help="List recent N jobs (default: 10)")
    parser.add_argument("--job-summary", type=str, help="Show summary for specific job ID")
    parser.add_argument("--clean", action="store_true", help="Clean old log files")
    parser.add_argument("--keep-days", type=int, default=7, 
                       help="Keep logs from last N days (default: 7)")
    parser.add_argument("--dry-run", action="store_true", 
                       help="Show what would be cleaned without deleting")
    
    args = parser.parse_args()
    
    manager = SlurmLogManager()
    
    if args.status:
        manager.get_status()
    elif args.job_summary:
        manager.show_log_summary(args.job_summary)
    elif args.clean:
        manager.clean_old_logs(keep_days=args.keep_days, dry_run=args.dry_run)
    else:
        manager.list_recent_jobs(limit=args.list_jobs)

if __name__ == "__main__":
    main()