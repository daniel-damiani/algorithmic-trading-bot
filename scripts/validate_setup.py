#!/usr/bin/env python3
"""
Setup Validation Script

Validates that the QuantumSentiment trading bot is properly set up
and all components can be imported and initialized correctly.
"""

import os
import sys
from pathlib import Path
import structlog

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure simple logging
import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Also configure structlog as backup
structlog.configure(
    processors=[
        structlog.dev.ConsoleRenderer()
    ],
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)


def check_environment():
    """Check environment variables and dependencies"""
    print("🔍 Checking environment setup")
    logger.info("🔍 Checking environment setup")
    
    issues = []
    
    # Check API credentials
    if not os.getenv('ALPACA_API_KEY'):
        issues.append("❌ Missing ALPACA_API_KEY in .env file")
    else:
        print("✅ ALPACA_API_KEY found")
        logger.info("✅ ALPACA_API_KEY found")
    
    if not os.getenv('ALPACA_API_SECRET'):
        issues.append("❌ Missing ALPACA_API_SECRET in .env file")
    else:
        print("✅ ALPACA_API_SECRET found")
        logger.info("✅ ALPACA_API_SECRET found")
    
    # Check optional credentials
    if os.getenv('REDDIT_CLIENT_ID'):
        logger.info("✅ Reddit API credentials found")
    else:
        logger.info("⚠️  Reddit API credentials not found (optional)")
    
    if os.getenv('NEWSAPI_KEY'):
        logger.info("✅ NewsAPI credentials found")
    else:
        logger.info("⚠️  NewsAPI credentials not found (optional)")
    
    return issues


def check_imports():
    """Check that all required modules can be imported"""
    logger.info("📦 Checking module imports")
    
    issues = []
    
    # Core modules
    modules_to_test = [
        ('src.data', 'AlpacaClient'),
        ('src.configuration', 'load_config'),
        ('src.training', 'ModelTrainingPipeline'),
        ('src.models.lstm', 'PriceLSTM'),
        ('src.models.cnn', 'ChartPatternCNN'),
        ('src.models.xgboost', 'MarketRegimeXGBoost'),
        ('src.models.transformers', 'FinBERT'),
        ('src.models.ensemble', 'StackedEnsemble'),
    ]
    
    for module_name, class_name in modules_to_test:
        try:
            module = __import__(module_name, fromlist=[class_name])
            getattr(module, class_name)
            logger.info(f"✅ {module_name}.{class_name}")
        except Exception as e:
            issues.append(f"❌ {module_name}.{class_name}: {e}")
    
    return issues


def check_model_configs():
    """Check that model configurations are valid"""
    logger.info("⚙️  Checking model configurations")
    
    issues = []
    
    try:
        from src.models.lstm import PriceLSTMConfig
        from src.models.cnn import ChartPatternConfig
        from src.models.xgboost import MarketRegimeConfig
        from src.models.transformers import FinBERTConfig
        from src.models.ensemble import StackedEnsembleConfig
        
        # Test config creation
        configs = [
            PriceLSTMConfig(),
            ChartPatternConfig(),
            MarketRegimeConfig(),
            FinBERTConfig(),
            StackedEnsembleConfig()
        ]
        
        for config in configs:
            logger.info(f"✅ {config.__class__.__name__} created successfully")
            
    except Exception as e:
        issues.append(f"❌ Model config error: {e}")
    
    return issues


def check_data_client():
    """Check that data client can be initialized"""
    logger.info("📊 Checking data client initialization")
    
    issues = []
    
    try:
        from src.data import AlpacaClient
        
        if os.getenv('ALPACA_API_KEY') and os.getenv('ALPACA_API_SECRET'):
            client = AlpacaClient()
            logger.info("✅ AlpacaClient initialized successfully")
        else:
            logger.info("⚠️  Skipping AlpacaClient test (no credentials)")
            
    except Exception as e:
        issues.append(f"❌ AlpacaClient initialization error: {e}")
    
    return issues


def check_file_structure():
    """Check that required files and directories exist"""
    logger.info("📁 Checking file structure")
    
    issues = []
    
    required_files = [
        'requirements.txt',
        'src/__init__.py',
        'src/train_models.py',
        'config/config.yaml',
        'scripts/download_historical_data.py',
        'scripts/test_training.py',
    ]
    
    required_dirs = [
        'src/models',
        'src/data',
        'src/training',
        'config',
        'scripts'
    ]
    
    for file_path in required_files:
        if (project_root / file_path).exists():
            logger.info(f"✅ {file_path}")
        else:
            issues.append(f"❌ Missing file: {file_path}")
    
    for dir_path in required_dirs:
        if (project_root / dir_path).exists():
            logger.info(f"✅ {dir_path}/")
        else:
            issues.append(f"❌ Missing directory: {dir_path}/")
    
    return issues


def main():
    """Run all validation checks"""
    print("🚀 QuantumSentiment Trading Bot - Setup Validation")
    print("=" * 60)
    logger.info("🚀 QuantumSentiment Trading Bot - Setup Validation")
    logger.info("=" * 60)
    
    all_issues = []
    
    # Run all checks
    all_issues.extend(check_environment())
    all_issues.extend(check_file_structure())
    all_issues.extend(check_imports())
    all_issues.extend(check_model_configs())
    all_issues.extend(check_data_client())
    
    print("=" * 60)
    
    if all_issues:
        print("❌ Setup validation failed!")
        print(f"Found {len(all_issues)} issues:")
        for issue in all_issues:
            print(f"  {issue}")
        print("")
        print("📖 Please check SETUP_GUIDE.md for setup instructions")
        logger.error("❌ Setup validation failed!")
        return False
    else:
        print("🎉 Setup validation passed!")
        print("✅ All components are properly configured")
        print("")
        print("🚀 Ready to:")
        print("  1. Download data: python scripts/download_historical_data.py")
        print("  2. Test training: python scripts/test_training.py")
        print("  3. Train models: python src/train_models.py")
        logger.info("🎉 Setup validation passed!")
        return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)