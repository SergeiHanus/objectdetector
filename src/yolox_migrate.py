#!/usr/bin/env python3
"""
YOLOX Migration Helper Script
Assist in transitioning from YOLOv8 to YOLOX training pipeline
"""

import os
import sys
import shutil
from pathlib import Path

def check_current_setup():
    """Check current project setup and identify migration needs"""
    
    print("🔍 Checking current project setup...")
    
    # Check if we're in the project root
    if not (os.path.exists('yolox') and os.path.exists('data') and os.path.exists('src')):
        print("❌ Error: This script must be run from the project root directory")
        print("   Expected structure: yolox/, data/, src/, config/ at same level")
        print("   Current directory:", os.getcwd())
        print("   Please run from: /data/code/image-detector/")
        sys.exit(1)
    
    # Check existing YOLOv8 files
    yolov8_files = [
        'src/train_model.py',
        'src/test_model.py',
        'src/export_model.py',
        'requirements.txt'
    ]
    
    existing_yolov8 = []
    for file_path in yolov8_files:
        if os.path.exists(file_path):
            existing_yolov8.append(file_path)
    
    print(f"Found {len(existing_yolov8)} existing YOLOv8 files")
    
    # Check YOLOX files
    yolox_files = [
        'src/yolox_setup.py',
        'src/yolox_data_prep.py',
        'src/yolox_train.py',
        'src/yolox_test.py',
        'requirements_yolox.txt',
        'README_YOLOX.md'
    ]
    
    existing_yolox = []
    for file_path in yolox_files:
        if os.path.exists(file_path):
            existing_yolox.append(file_path)
    
    print(f"Found {len(existing_yolox)} YOLOX files")
    
    # Check dataset status
    dataset_status = check_dataset_status()
    
    return {
        'yolov8_files': existing_yolov8,
        'yolox_files': existing_yolox,
        'dataset_status': dataset_status
    }

def check_dataset_status():
    """Check the current dataset status"""
    
    print("\n📊 Checking dataset status...")
    
    status = {
        'raw_data': False,
        'organized': False,
        'coco_ready': False
    }
    
    # Check raw data
    if os.path.exists('data/raw'):
        raw_images = len([f for f in os.listdir('data/raw') if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        raw_labels = len([f for f in os.listdir('data/raw') if f.endswith('.txt')])
        
        if raw_images > 0 and raw_labels > 0:
            status['raw_data'] = True
            print(f"✅ Raw data: {raw_images} images, {raw_labels} labels")
        else:
            print("⚠️  Raw data incomplete")
    else:
        print("❌ Raw data directory not found")
    
    # Check organized dataset
    if os.path.exists('data/train/images') and os.path.exists('data/val/images'):
        train_count = len(os.listdir('data/train/images'))
        val_count = len(os.listdir('data/val/images'))
        
        if train_count > 0 and val_count > 0:
            status['organized'] = True
            print(f"✅ Organized dataset: {train_count} train, {val_count} val")
        else:
            print("⚠️  Organized dataset incomplete")
    else:
        print("❌ Organized dataset not found")
    
    # Check COCO annotations
    coco_files = [
        'data/train/labels.json',
        'data/val/labels.json'
    ]
    
    coco_exists = all(os.path.exists(f) for f in coco_files)
    if coco_exists:
        status['coco_ready'] = True
        print("✅ COCO annotations ready")
    else:
        print("❌ COCO annotations not found")
    
    return status

def backup_yolov8_files():
    """Create backup of existing YOLOv8 files"""
    
    print("\n💾 Creating backup of YOLOv8 files...")
    
    backup_dir = 'backup_yolov8'
    os.makedirs(backup_dir, exist_ok=True)
    
    yolov8_files = [
        'src/train_model.py',
        'src/test_model.py', 
        'src/export_model.py',
        'requirements.txt'
    ]
    
    backed_up = []
    for file_path in yolov8_files:
        if os.path.exists(file_path):
            backup_path = os.path.join(backup_dir, os.path.basename(file_path))
            shutil.copy2(file_path, backup_path)
            backed_up.append(file_path)
            print(f"✅ Backed up: {file_path}")
    
    print(f"✅ Backed up {len(backed_up)} files to {backup_dir}/")
    return backup_dir

def create_migration_guide():
    """Create a migration guide for the user"""
    
    print("\n📝 Creating migration guide...")
    
    guide_content = """# YOLOv8 to YOLOX Migration Guide

## What Changed

Your project has been successfully migrated from YOLOv8 to YOLOX training pipeline.

## New Scripts

- `src/yolox_setup.py` - Install YOLOX and dependencies
- `src/yolox_data_prep.py` - Prepare dataset for YOLOX training  
- `src/yolox_train.py` - Train YOLOX model for circle detection
- `src/yolox_test.py` - Test trained YOLOX model

## Migration Steps

### 1. Setup YOLOX Environment
```bash
python src/yolox_setup.py
```

### 2. Prepare Dataset (if needed)
```bash
python src/yolox_data_prep.py
```

### 3. Train Model
```bash
python src/yolox_train.py
```

### 4. Test Model
```bash
python src/yolox_test.py
```

## Key Differences

| Aspect | YOLOv8 | YOLOX |
|--------|--------|-------|
| **Model** | YOLOv8n | YOLOX-Small |
| **Input Size** | Configurable | Fixed 640x640 |
| **Export** | TFLite, ONNX | ONNX |
| **Configuration** | YAML files | Python configs |

## Benefits of YOLOX

- More consistent training results
- Better ONNX export quality
- Proven production architecture
- Active community support

## Rollback (if needed)

If you need to rollback to YOLOv8:
1. Restore files from `backup_yolov8/` directory
2. Use original training scripts
3. Models remain compatible

## Support

- See `README_YOLOX.md` for detailed documentation
- Check troubleshooting section for common issues
- YOLOX repository: https://github.com/Megvii-BaseDetection/YOLOX

---
Migration completed successfully! 🎉
"""
    
    guide_path = 'MIGRATION_GUIDE.md'
    with open(guide_path, 'w') as f:
        f.write(guide_content)
    
    print(f"✅ Migration guide created: {guide_path}")

def show_next_steps():
    """Show the next steps for the user"""
    
    print("\n🎯 === Migration Complete! ===")
    print("\n📋 Next Steps:")
    print("   1. Setup YOLOX environment:")
    print("      python src/yolox_setup.py")
    print("\n   2. Prepare dataset (if needed):")
    print("      python src/yolox_data_prep.py")
    print("\n   3. Train YOLOX model:")
    print("      python src/yolox_train.py")
    print("\n   4. Test the model:")
    print("      python src/yolox_test.py")
    print("\n📚 Documentation:")
    print("   - README_YOLOX.md - Complete YOLOX guide")
    print("   - MIGRATION_GUIDE.md - Migration details")
    print("   - backup_yolov8/ - Your original files")
    print("\n🔄 Rollback (if needed):")
    print("   - Restore files from backup_yolov8/ directory")
    print("   - Use original YOLOv8 training scripts")
    print("\n🎉 Welcome to YOLOX training!")

def main():
    """Main migration function"""
    
    print("🔄 === YOLOv8 to YOLOX Migration Helper ===")
    
    # Check current setup
    setup_info = check_current_setup()
    
    # Show current status
    print(f"\n📊 Current Status:")
    print(f"   YOLOv8 files: {len(setup_info['yolov8_files'])}")
    print(f"   YOLOX files: {len(setup_info['yolox_files'])}")
    print(f"   Dataset ready: {setup_info['dataset_status']['coco_ready']}")
    
    # Check if migration is needed
    if len(setup_info['yolox_files']) == len([
        'src/yolox_setup.py',
        'src/yolox_data_prep.py', 
        'src/yolox_train.py',
        'src/yolox_test.py',
        'requirements_yolox.txt',
        'README_YOLOX.md'
    ]):
        print("\n✅ YOLOX migration already completed!")
        show_next_steps()
        return
    
    # Backup existing files
    backup_dir = backup_yolov8_files()
    
    # Create migration guide
    create_migration_guide()
    
    # Show next steps
    show_next_steps()
    
    print(f"\n💾 Your original YOLOv8 files are safely backed up in: {backup_dir}/")

if __name__ == "__main__":
    main()
