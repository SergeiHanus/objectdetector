#!/bin/bash
# Clean up processed dataset while preserving raw data
# Usage: ./src/cleanup_dataset.sh [--models] [--android] [--all]

set -e

# Verify we're running from the correct directory
if [[ ! -d "venv" ]] || [[ ! -d "data" ]] || [[ ! -d "src" ]]; then
    echo "❌ Error: This script must be run from the project root directory"
    echo "   Expected structure: venv/, data/, src/, config/ at same level"
    echo "   Current directory: $(pwd)"
    echo "   Please run from: /data/code/image-detector/"
    echo "   Usage: ./src/cleanup_dataset.sh [--models] [--android] [--all]"
    exit 1
fi

echo "🧹 === DATASET CLEANUP UTILITY ==="
echo ""

# Parse command line arguments
CLEAN_MODELS=false
CLEAN_ANDROID=false
CLEAN_ALL=false

for arg in "$@"; do
    case $arg in
        --models)
            CLEAN_MODELS=true
            shift
            ;;
        --android)
            CLEAN_ANDROID=true
            shift
            ;;
        --all)
            CLEAN_ALL=true
            shift
            ;;
        *)
            echo "Unknown option: $arg"
            echo "Usage: ./src/cleanup_dataset.sh [--models] [--android] [--all]"
            exit 1
            ;;
    esac
done

# Check what will be preserved
echo "📋 **What will be PRESERVED:**"
if [ -d "data/raw" ]; then
    raw_images=$(ls data/raw/*.{jpg,jpeg,png,JPG,JPEG,PNG} 2>/dev/null | wc -l)
    raw_annotations=$(ls data/raw/*.txt 2>/dev/null | wc -l)
    echo "   ✅ data/raw/ - Original images and annotations"
    echo "      📸 Images: $raw_images"
    echo "      🏷️  Annotations: $raw_annotations"
else
    echo "   ⚠️  data/raw/ directory not found!"
fi

echo "   ✅ All scripts in src/"
echo "   ✅ Configuration files (config/)"
echo "   ✅ Virtual environment (venv/)"
echo "   ✅ Project files (README.md, requirements.txt, etc.)"

echo ""
echo "🗑️  **What will be DELETED:**"

# Always cleaned directories
cleanup_dirs=()
cleanup_descriptions=()

if [ -d "data/train" ]; then
    train_images=$(find data/train -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" 2>/dev/null | wc -l)
    cleanup_dirs+=("data/train")
    cleanup_descriptions+=("Training dataset ($train_images images)")
fi

if [ -d "data/val" ]; then
    val_images=$(find data/val -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" 2>/dev/null | wc -l)
    cleanup_dirs+=("data/val")
    cleanup_descriptions+=("Validation dataset ($val_images images)")
fi

if [ -d "data/test" ]; then
    test_images=$(find data/test -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" 2>/dev/null | wc -l)
    cleanup_dirs+=("data/test")
    cleanup_descriptions+=("Test dataset ($test_images images)")
fi

# Optional cleanup directories
optional_dirs=("data/verification" "data/edge_cases" "data/test_new")
for dir in "${optional_dirs[@]}"; do
    if [ -d "$dir" ]; then
        dir_images=$(find "$dir" -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" 2>/dev/null | wc -l)
        cleanup_dirs+=("$dir")
        cleanup_descriptions+=("$(basename "$dir") images ($dir_images images)")
    fi
done

# Models (if requested)
if [ "$CLEAN_MODELS" = true ] || [ "$CLEAN_ALL" = true ]; then
    if [ -d "models" ]; then
        model_files=$(find models -name "*.pt" -o -name "*.onnx" -o -name "*.tflite" 2>/dev/null | wc -l)
        cleanup_dirs+=("models")
        cleanup_descriptions+=("Trained models ($model_files files)")
    fi
fi

# Android assets (if requested)
if [ "$CLEAN_ANDROID" = true ] || [ "$CLEAN_ALL" = true ]; then
    if [ -d "android/assets" ]; then
        asset_files=$(find android/assets -type f 2>/dev/null | wc -l)
        cleanup_dirs+=("android/assets")
        cleanup_descriptions+=("Android model assets ($asset_files files)")
    fi
fi

# Show what will be deleted
if [ ${#cleanup_dirs[@]} -eq 0 ]; then
    echo "   📭 Nothing to clean up!"
    echo ""
    echo "🎯 **Available cleanup options:**"
    echo "   --models   : Also clean trained models"
    echo "   --android  : Also clean Android assets"
    echo "   --all      : Clean everything except raw data"
    exit 0
fi

for i in "${!cleanup_dirs[@]}"; do
    echo "   ❌ ${cleanup_dirs[$i]} - ${cleanup_descriptions[$i]}"
done

# Show disk space that will be freed
echo ""
echo "💾 **Disk Space:**"
total_size=0
for dir in "${cleanup_dirs[@]}"; do
    if [ -d "$dir" ]; then
        dir_size=$(du -sh "$dir" 2>/dev/null | cut -f1)
        echo "   📁 $dir: $dir_size"
    fi
done

# Confirmation prompt
echo ""
echo "⚠️  **CONFIRMATION REQUIRED**"
echo "This action will permanently delete the directories listed above."
echo "Your raw images and annotations in data/raw/ will be preserved."
echo ""
read -p "Are you sure you want to continue? (y/N): " -n 1 -r
echo

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Cleanup cancelled."
    exit 0
fi

# Perform cleanup
echo ""
echo "🧹 **Starting cleanup...**"

cleanup_count=0
for dir in "${cleanup_dirs[@]}"; do
    if [ -d "$dir" ]; then
        echo "   🗑️  Removing $dir..."
        rm -rf "$dir"
        echo "   ✅ Deleted: $dir"
        ((cleanup_count++))
    fi
done

# Clean up any empty parent directories (but not data/ itself)
for parent_dir in "data/train" "data/val" "data/test"; do
    parent=$(dirname "$parent_dir")
    if [ -d "$parent" ] && [ "$parent" != "data" ]; then
        if [ -z "$(ls -A "$parent" 2>/dev/null)" ]; then
            echo "   🗑️  Removing empty directory: $parent"
            rmdir "$parent" 2>/dev/null || true
        fi
    fi
done

echo ""
echo "✅ **Cleanup completed!**"
echo "   📊 Directories removed: $cleanup_count"
echo "   ✅ Raw data preserved in data/raw/"

# Show what to do next
echo ""
echo "🚀 **Next Steps:**"
echo "   1. Check your annotations: ./src/check_annotations.sh"
echo "   2. Re-organize dataset: python src/data_preparation.py"
echo "   3. Train new model: python src/train_model.py"
echo ""
echo "💡 **Tip:** Use different cleanup options:"
echo "   ./src/cleanup_dataset.sh --models    # Also remove trained models"
echo "   ./src/cleanup_dataset.sh --all       # Clean everything except raw data" 