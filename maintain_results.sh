#!/bin/bash
# Maintenance utility for results directory
# Usage: ./maintain_results.sh [action]
# Actions: clean, status, archive

ACTION=${1:-"status"}

echo "🛠️  Results Directory Maintenance"
echo "================================"

case $ACTION in
    "status")
        echo "📊 Current Results Status:"
        echo ""
        
        if [ -d "results" ]; then
            for dir in data plots reports; do
                if [ -d "results/$dir" ]; then
                    count=$(find "results/$dir" -name "pd_vs_tool_*" | wc -l)
                    total=$(find "results/$dir" -maxdepth 1 -type f | wc -l)
                    echo "   $dir/: $count standardized files, $total total"
                fi
            done
            
            # Count archives
            archives=$(find "results" -maxdepth 1 -name "archive_*" -type d | wc -l)
            echo "   archives: $archives directories"
        else
            echo "   No results directory found"
        fi
        ;;
        
    "clean")
        echo "🧹 Running cleanup..."
        python clean_results.py
        ;;
        
    "archive")
        echo "📦 Creating manual archive..."
        timestamp=$(date +"%Y%m%d_%H%M%S")
        archive_dir="results/manual_archive_$timestamp"
        mkdir -p "$archive_dir"
        
        # Move all files except most recent of each type
        for dir in data plots reports; do
            if [ -d "results/$dir" ]; then
                mkdir -p "$archive_dir/$dir"
                # Find all files except the most recent
                find "results/$dir" -name "pd_vs_tool_*" -printf '%T@ %p\n' | sort -n | head -n -1 | cut -d' ' -f2- | while read file; do
                    if [ -f "$file" ]; then
                        mv "$file" "$archive_dir/$dir/"
                        echo "   📦 Archived: $(basename "$file")"
                    fi
                done
            fi
        done
        echo "✅ Manual archive created: $archive_dir"
        ;;
        
    "help")
        echo "Available actions:"
        echo "   status  - Show current directory status (default)"
        echo "   clean   - Run interactive cleanup"
        echo "   archive - Create manual archive keeping latest files"
        echo "   help    - Show this help"
        ;;
        
    *)
        echo "❌ Unknown action: $ACTION"
        echo "Use: ./maintain_results.sh help"
        exit 1
        ;;
esac