#!/usr/bin/env python3
"""
YOLOX Setup Script
Install YOLOX and prepare the environment for circle detection training.
Platform-aware: CUDA on Linux, default PyTorch (MPS/CPU) on macOS.
"""

import os
import sys
import subprocess
import shutil
import platform

# Pin YOLOX revision for reproducible remote setup (0.3.0 release)
YOLOX_REVISION = "0.3.0"

def is_darwin():
    return platform.system() == "Darwin"

def check_requirements():
    """Check if basic requirements are met"""
    
    print("🔍 Checking system requirements...")
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 7):
        print("❌ Error: Python 3.7+ is required")
        print(f"   Current version: {python_version.major}.{python_version.minor}")
        sys.exit(1)
    
    print(f"✅ Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Check if we're in the project root
    if not (os.path.exists('yolox') and os.path.exists('data') and os.path.exists('src')):
        print("❌ Error: This script must be run from the project root directory")
        print("   Expected structure: yolox/, data/, src/, config/ at same level")
        print("   Current directory:", os.getcwd())
        print("   Please run from: /data/code/image-detector/")
        sys.exit(1)
    
    print("✅ Project structure verified")
    
    # Check if virtual environment is activated
    if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("❌ Error: Virtual environment not activated")
        print("   Please activate your virtual environment first:")
        print("   source yolox/bin/activate")
        sys.exit(1)
    
    print("✅ Virtual environment activated")
    
    # Check for Python development headers
    print("🔍 Checking Python development headers...")
    python_include = os.path.join(sys.prefix, 'include', f'python{sys.version_info.major}.{sys.version_info.minor}')
    if not os.path.exists(python_include):
        print("⚠️  Warning: Python development headers not found")
        print("   This may cause PyTorch compilation issues")
        print("   Consider installing: sudo dnf install python3-devel (Fedora/RHEL)")
        print("   or: sudo apt-get install python3-dev (Ubuntu/Debian)")
    else:
        print("✅ Python development headers found")
    
    # Check device availability (CUDA / MPS / CPU)
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        elif is_darwin() and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            print("✅ MPS (Apple Silicon) available")
        else:
            print("⚠️  Warning: No GPU backend (CUDA/MPS); YOLOX will use CPU")
    except ImportError:
        print("⚠️  Warning: PyTorch not installed")
        print("   Will install during setup")

def install_dependencies():
    """Install YOLOX dependencies. On macOS use default PyTorch (MPS/CPU); on Linux try CUDA first."""
    
    print("\n📦 Installing YOLOX dependencies...")
    
    if is_darwin():
        # macOS: install default PyTorch (includes MPS on Apple Silicon)
        print("🔧 Installing PyTorch (default; MPS on Apple Silicon)...")
        try:
            subprocess.run([
                sys.executable, "-m", "pip", "install",
                "torch", "torchvision", "torchaudio",
            ], check=True, capture_output=True, text=True)
            print("✅ PyTorch installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Error installing PyTorch: {e}")
            print_manual_pytorch_steps()
            sys.exit(1)
    else:
        # Linux: try CUDA first, then CPU fallback
        print("🔧 Installing PyTorch with GPU support (CUDA)...")
        try:
            subprocess.run([
                sys.executable, "-m", "pip", "install",
                "torch", "torchvision", "torchaudio",
                "--index-url", "https://download.pytorch.org/whl/cu118"
            ], check=True, capture_output=True, text=True)
            print("✅ PyTorch GPU version installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"⚠️  CUDA install failed, trying CPU fallback: {e}")
            try:
                subprocess.run([
                    sys.executable, "-m", "pip", "install",
                    "torch", "torchvision", "torchaudio",
                    "--index-url", "https://download.pytorch.org/whl/cpu"
                ], check=True, capture_output=True, text=True)
                print("✅ PyTorch CPU version installed")
            except subprocess.CalledProcessError as e2:
                print(f"❌ PyTorch installation failed: {e2}")
                print_manual_pytorch_steps()
                sys.exit(1)
    
    # Install other dependencies
    print("📦 Installing other dependencies...")
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements_yolox.txt"
        ], check=True, capture_output=True, text=True)
        print("✅ Dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        print(f"   Error output: {e.stderr}")
        sys.exit(1)

def setup_yolox():
    """Setup YOLOX repository and install"""
    
    print("\n🚀 Setting up YOLOX...")
    
    # Check if YOLOX already exists
    if os.path.exists('YOLOX'):
        print("📁 YOLOX directory already exists")
        
        # Check if it's a git repository
        if os.path.exists('YOLOX/.git'):
            print("🔄 Checking YOLOX revision (pinned)...")
            os.chdir('YOLOX')
            try:
                subprocess.run(["git", "fetch", "origin", "tag", YOLOX_REVISION, "--no-tags"], check=True, capture_output=True, text=True, timeout=60)
                subprocess.run(["git", "checkout", YOLOX_REVISION], check=True, capture_output=True, text=True)
                print(f"✅ YOLOX at {YOLOX_REVISION}")
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                print(f"⚠️  Could not pin to {YOLOX_REVISION}: {e}")
            os.chdir('..')
        else:
            print("⚠️  Warning: YOLOX directory exists but is not a git repository")
            print("   Removing and re-cloning...")
            shutil.rmtree('YOLOX')
            clone_yolox()
    else:
        clone_yolox()
    
    # Install YOLOX in development mode
    print("🔧 Installing YOLOX in development mode...")
    os.chdir('YOLOX')
    
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-v", "-e", "."
        ], check=True, capture_output=True, text=True)
        print("✅ YOLOX installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing YOLOX: {e}")
        print(f"   Error output: {e.stderr}")
        os.chdir('..')
        sys.exit(1)
    
    os.chdir('..')
    print("✅ YOLOX setup completed")

def print_manual_pytorch_steps():
    """Print manual PyTorch installation steps"""
    
    print("\n📋 MANUAL PYTORCH INSTALLATION STEPS:")
    print("=" * 50)
    print("1. First, uninstall any existing PyTorch installation:")
    print("   pip uninstall torch torchvision torchaudio -y")
    print("   pip cache purge")
    print()
    print("2. Install Python development headers (Fedora/RHEL):")
    print("   sudo dnf install python3-devel gcc gcc-c++ make")
    print("   OR for Ubuntu/Debian:")
    print("   sudo apt-get install python3-dev build-essential")
    print()
    print("3. For GPU support, install PyTorch with CUDA:")
    print("   # CUDA 11.8 (latest stable):")
    print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    print("   # CUDA 11.7 (alternative):")
    print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117")
    print("   # CUDA 11.6 (older but stable):")
    print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu116")
    print()
    print("4. If GPU installation fails, fallback to CPU version:")
    print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu")
    print()
    print("5. Alternative: Use conda for GPU support:")
    print("   conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia")
    print()
    print("6. Check CUDA availability after installation:")
    print("   python -c \"import torch; print(f'CUDA available: {torch.cuda.is_available()}')\"")
    print()
    print("7. After successful PyTorch installation, run this script again:")
    print("   python src/yolox_setup.py")
    print("=" * 50)

def clone_yolox():
    """Clone YOLOX repository and check out pinned revision."""
    
    print("📥 Cloning YOLOX repository...")
    
    try:
        subprocess.run([
            "git", "clone", "--depth", "1", "--branch", YOLOX_REVISION,
            "https://github.com/Megvii-BaseDetection/YOLOX.git"
        ], check=True, capture_output=True, text=True, timeout=120)
        print(f"✅ YOLOX repository cloned at {YOLOX_REVISION}")
    except subprocess.CalledProcessError as e:
        # Fallback: clone without branch (e.g. tag may need full clone)
        try:
            subprocess.run(["git", "clone", "https://github.com/Megvii-BaseDetection/YOLOX.git"], check=True, capture_output=True, text=True, timeout=120)
            os.chdir("YOLOX")
            subprocess.run(["git", "checkout", YOLOX_REVISION], check=True, capture_output=True, text=True)
            os.chdir("..")
            print(f"✅ YOLOX repository cloned and checked out {YOLOX_REVISION}")
        except subprocess.CalledProcessError as e2:
            print(f"❌ Error cloning YOLOX: {e2}")
            print(f"   Error output: {e2.stderr if hasattr(e2, 'stderr') else ''}")
            sys.exit(1)

def create_yolox_configs():
    """Create necessary YOLOX configuration files"""
    
    print("\n⚙️  Creating YOLOX configurations...")
    
    # Create exps directory
    os.makedirs('exps', exist_ok=True)
    
    # Create base YOLOX-S configuration
    base_config = '''# YOLOX-S base configuration
_base_ = [
    '../yolox_s_300e_coco.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_300e.py',
    '../_base_/default_runtime.py'
]

# Model settings
model = dict(
    bbox_head=dict(
        num_classes=1,  # Only circle class
    )
)

# Training settings
total_epochs = 100
evaluation = dict(interval=10, metric='bbox')
save_checkpoint_interval = 10
log_interval = 50

# Export settings
export = dict(
    type='onnx',
    input_shape=(1, 3, 640, 640),
    opset_version=11,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)
'''
    
    with open('exps/yolox_s_base.py', 'w') as f:
        f.write(base_config)
    
    print("✅ Created base YOLOX configuration: exps/yolox_s_base.py")
    
    # Create circle detection specific config
    circle_config = '''# YOLOX-S Circle Detection Configuration
_base_ = ['./yolox_s_base.py']

# Dataset settings
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        dataset=dict(
            ann_file='data/train/labels.json',
            img_prefix='data/train/images/',
            classes=['circle']
        )
    ),
    val=dict(
        dataset=dict(
            ann_file='data/val/labels.json',
            img_prefix='data/val/images/',
            classes=['circle']
        )
    ),
    test=dict(
        dataset=dict(
            ann_file='data/val/labels.json',
            img_prefix='data/val/images/',
            classes=['circle']
        )
    )
)

# Optimizer settings
optimizer = dict(
    type='SGD',
    lr=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    nesterov=True,
)

# Learning rate scheduler
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.001,
    step=[80, 90]
)
'''
    
    with open('exps/yolox_s_circle.py', 'w') as f:
        f.write(circle_config)
    
    print("✅ Created circle detection configuration: exps/yolox_s_circle.py")

def verify_setup():
    """Verify YOLOX setup"""
    
    print("\n🔍 Verifying YOLOX setup...")
    
    # Check PyTorch installation and device
    try:
        import torch
        print(f"✅ PyTorch imported successfully: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"   Device: CUDA ({torch.cuda.get_device_name(0)})")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            print("   Device: MPS (Apple Silicon)")
        else:
            print("   Device: CPU")
    except ImportError as e:
        print(f"❌ Error importing PyTorch: {e}")
        print("   PyTorch installation failed. Please follow manual steps above.")
        return False
    
    # Check YOLOX installation
    try:
        import yolox
        print(f"✅ YOLOX imported successfully: {yolox.__version__}")
    except ImportError as e:
        print(f"❌ Error importing YOLOX: {e}")
        print("   YOLOX installation failed. Please check error messages above.")
        return False
    
    # Check YOLOX directory
    if not os.path.exists('YOLOX'):
        print("❌ YOLOX directory not found")
        return False
    
    # Check configuration files
    config_files = [
        'exps/yolox_s_base.py',
        'exps/yolox_s_circle.py'
    ]
    
    for config_file in config_files:
        if os.path.exists(config_file):
            print(f"✅ {config_file} exists")
        else:
            print(f"❌ Missing: {config_file}")
            return False
    
    print("✅ YOLOX setup verification completed!")
    return True

def main():
    """Main setup function"""
    
    print("🎯 === YOLOX Setup for Circle Detection ===")
    
    # Check requirements
    check_requirements()
    
    # Install dependencies
    install_dependencies()
    
    # Setup YOLOX
    setup_yolox()
    
    # Create configurations
    create_yolox_configs()
    
    # Verify setup
    if verify_setup():
        print("\n🎉 YOLOX setup completed successfully!")
        print("\n📋 Next steps:")
        print("   1. Prepare your dataset: python src/yolox_data_prep.py")
        print("   2. Train the model: python src/yolox_train.py")
        print("   3. Test the model: python src/yolox_test.py")
        print("\n📚 For more information, see the README_YOLOX.md file")
    else:
        print("\n❌ YOLOX setup failed. Please check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
