#!/usr/bin/env python3
"""
QuantumSentiment Quick Start Script

Automated setup and validation for new users to get the system running quickly.

Usage:
    python quick_start.py
    python quick_start.py --skip-training  # Skip model training
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
TRAINING_DATA_MASSIVE = PROJECT_ROOT / "data" / "massive" / "massive_training_data.parquet"
TRAINING_DATA_PRODUCTION = PROJECT_ROOT / "data" / "training" / "production"

def print_header():
    """Print welcome header"""
    print("🚀" + "="*60 + "🚀")
    print("    QUANTUMSENTIMENT TRADING BOT - QUICK START")
    print("🚀" + "="*60 + "🚀")
    print()

def check_python_version():
    """Check Python version is adequate"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required. Current version:", sys.version)
        return False
    
    print("✅ Python version:", sys.version_info.major, ".", sys.version_info.minor)
    return True

def check_venv():
    """Check if running in virtual environment"""
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ Running in virtual environment")
        return True
    else:
        print("⚠️  Not running in virtual environment")
        print("   Recommended: python -m venv .venv && source .venv/bin/activate")
        response = input("   Continue anyway? (y/N): ")
        return response.lower() == 'y'

def install_dependencies():
    """Install required dependencies"""
    print("\n📦 Installing dependencies...")
    
    try:
        print("Installing core dependencies...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True, capture_output=True)
        print("✅ Core dependencies installed")
        
        print("Installing ML dependencies...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements-ml.txt"], 
                      check=True, capture_output=True)
        print("✅ ML dependencies installed")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def create_directories():
    """Create required directories"""
    print("\n📁 Creating directories...")
    
    dirs = ["models", "data", "logs", "cache", "backups"]
    
    for dir_name in dirs:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"✅ Created {dir_name}/")
    
    return True

def check_env_file():
    """Check if .env file exists and help create it"""
    print("\n🔐 Checking environment configuration...")
    
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if env_file.exists():
        print("✅ .env file found")
        return True
    
    print("⚠️  .env file not found")
    
    if env_example.exists():
        print("📝 Found .env.example file")
        response = input("   Copy .env.example to .env? (Y/n): ")
        
        if response.lower() != 'n':
            env_file.write_text(env_example.read_text())
            print("✅ Created .env file from template")
            print("📝 IMPORTANT: Edit .env file with your API keys!")
            print("   Required: ALPACA_API_KEY, ALPACA_API_SECRET")
            print("   Optional: REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET")
            return True
    
    print("❌ Please create .env file with your API keys")
    print("   See README.md for details")
    return False

def run_setup_test():
    """Run setup validation"""
    print("\n🧪 Running setup validation...")
    
    script = PROJECT_ROOT / "scripts" / "test_setup.py"
    try:
        result = subprocess.run(
            [sys.executable, str(script)],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )
        
        if result.stdout:
            print(result.stdout.rstrip())
        if result.returncode == 0:
            print("✅ Setup validation passed!")
            return True
        print("❌ Setup validation failed:")
        if result.stderr:
            print(result.stderr.rstrip())
        return False
            
    except Exception as e:
        print(f"⚠️  Could not run setup test: {e}")
        return False

def _training_data_available() -> bool:
    if TRAINING_DATA_MASSIVE.exists():
        return True
    if TRAINING_DATA_PRODUCTION.exists() and any(TRAINING_DATA_PRODUCTION.iterdir()):
        return True
    return False


def train_models(skip_training=False):
    """Train models or check if they exist"""
    print("\n🧠 Checking models...")
    
    models_dir = PROJECT_ROOT / "models"
    model_files = list(models_dir.glob("**/*.pkl")) + list(models_dir.glob("**/*.pt"))
    
    if model_files:
        print(f"✅ Found {len(model_files)} existing model files")
        return True
    
    train_hint = (
        "   python training/train_simple_massive.py --symbols 10\n"
        "   # or after downloading production data:\n"
        "   python train_production.py"
    )

    if skip_training:
        print("⚠️  No models found, but training skipped")
        print(train_hint)
        return True

    if not _training_data_available():
        print("⚠️  No models found and no training data detected")
        print("   Download data first, then train:")
        print("   python scripts/download_quality_data.py --symbols 30")
        print("   python training/train_simple_massive.py --symbols 10")
        return True

    print("📚 No models found. Training data is available.")
    if TRAINING_DATA_MASSIVE.exists():
        print("   Quick path: training/train_simple_massive.py (~30-60 minutes)")
    else:
        print("   Path: training/train_production_model.py")
    
    response = input("   Start training now? (Y/n): ")
    if response.lower() == 'n':
        print("⏭️  Skipping model training")
        print(train_hint)
        return True
    
    try:
        print("🎯 Starting quick training...")
        if TRAINING_DATA_MASSIVE.exists():
            cmd = [
                sys.executable,
                str(PROJECT_ROOT / "training" / "train_simple_massive.py"),
                "--symbols", "10",
            ]
        else:
            cmd = [sys.executable, str(PROJECT_ROOT / "train_production.py")]
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
        print("✅ Models trained successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Model training failed: {e}")
        print(train_hint)
        return False

def show_next_steps():
    """Show what to do next"""
    print("\n" + "🎉" + "="*60 + "🎉")
    print("    SETUP COMPLETE! READY TO TRADE!")
    print("🎉" + "="*60 + "🎉")
    print()
    print("📋 Next Steps:")
    print()
    print("1. 📝 Edit your .env file with API keys:")
    print("   - ALPACA_API_KEY and ALPACA_API_SECRET (required)")
    print("   - REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET (optional)")
    print()
    print("2. 🧪 Test your setup:")
    print("   python scripts/test_setup.py")
    print()
    print("3. 📊 Run a backtest:")
    print("   python backtest.py --start-date 2024-01-01 --end-date 2024-06-30")
    print()
    print("4. 📈 Start paper trading:")
    print("   python src/main.py --mode paper")
    print()
    print("5. 📚 Train better models (optional):")
    print("   python training/train_simple_massive.py --symbols 30")
    print("   python train_production.py  # requires data/training/production")
    print()
    print("📖 For detailed documentation, see README.md")
    print("💡 For troubleshooting, see the README.md troubleshooting section")
    print()

def main():
    """Main setup process"""
    parser = argparse.ArgumentParser(description="QuantumSentiment Quick Start")
    parser.add_argument("--skip-training", action="store_true", 
                       help="Skip model training for faster setup")
    args = parser.parse_args()
    
    print_header()
    
    # Step 1: Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Step 2: Check virtual environment
    if not check_venv():
        sys.exit(1)
    
    # Step 3: Install dependencies
    if not install_dependencies():
        print("❌ Failed to install dependencies. Please install manually:")
        print("   pip install -r requirements.txt")
        print("   pip install -r requirements-ml.txt")
        sys.exit(1)
    
    # Step 4: Create directories
    create_directories()
    
    # Step 5: Check/create .env file
    if not check_env_file():
        print("❌ Please create .env file before continuing")
        sys.exit(1)
    
    # Step 6: Run setup validation
    setup_ok = run_setup_test()
    
    # Step 7: Train models (optional)
    models_ok = train_models(skip_training=args.skip_training)
    
    # Show results and next steps
    if setup_ok and models_ok:
        show_next_steps()
    else:
        print("\n⚠️  Setup completed with warnings.")
        print("   Check the issues above and run setup test:")
        print("   python scripts/test_setup.py")

if __name__ == "__main__":
    main()