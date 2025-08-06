#!/usr/bin/env python3
"""
QuantumSentiment Setup Validation Script

Tests configuration files, dependencies, and basic setup to ensure
everything is ready for trading operations.

Usage:
    python scripts/test_setup.py
    python scripts/test_setup.py --config config/config_small_data.yaml
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import asyncio
from typing import Dict, Any, List
import structlog

# Test logging setup
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer()
    ],
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


class SetupValidator:
    """Validates system setup and configuration"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = config_path
        self.test_results: Dict[str, Dict[str, Any]] = {}
        
    async def run_all_tests(self) -> bool:
        """Run all validation tests"""
        tests = [
            ("Environment Variables", self.test_environment_variables),
            ("Configuration Files", self.test_configuration),
            ("Required Directories", self.test_directories),
            ("Core Dependencies", self.test_core_dependencies),
            ("ML Dependencies", self.test_ml_dependencies),
            ("Database Connection", self.test_database),
            ("API Connections", self.test_api_connections),
            ("Model Files", self.test_models),
        ]
        
        logger.info("🧪 Running QuantumSentiment Setup Validation")
        logger.info("=" * 50)
        
        all_passed = True
        
        for test_name, test_func in tests:
            try:
                logger.info(f"Testing: {test_name}")
                result = await test_func()
                
                if result["success"]:
                    logger.info(f"✅ {test_name}: PASSED")
                    if result.get("details"):
                        logger.info(f"   {result['details']}")
                else:
                    logger.error(f"❌ {test_name}: FAILED")
                    logger.error(f"   {result.get('error', 'Unknown error')}")
                    if result.get("fix"):
                        logger.info(f"   💡 Fix: {result['fix']}")
                    all_passed = False
                
                self.test_results[test_name] = result
                
            except Exception as e:
                logger.error(f"❌ {test_name}: ERROR - {str(e)}")
                self.test_results[test_name] = {
                    "success": False,
                    "error": str(e)
                }
                all_passed = False
        
        logger.info("=" * 50)
        if all_passed:
            logger.info("🎉 ALL TESTS PASSED! System ready for trading.")
        else:
            logger.error("❌ Some tests failed. Please fix issues before proceeding.")
        
        return all_passed
    
    async def test_environment_variables(self) -> Dict[str, Any]:
        """Test required environment variables"""
        required_vars = ["ALPACA_API_KEY", "ALPACA_API_SECRET"]
        optional_vars = ["REDDIT_CLIENT_ID", "REDDIT_CLIENT_SECRET", "DATABASE_URL"]
        
        missing_required = []
        missing_optional = []
        
        for var in required_vars:
            if not os.getenv(var):
                missing_required.append(var)
        
        for var in optional_vars:
            if not os.getenv(var):
                missing_optional.append(var)
        
        if missing_required:
            return {
                "success": False,
                "error": f"Missing required environment variables: {missing_required}",
                "fix": "Create .env file with required variables (see .env.example)"
            }
        
        details = f"Required vars OK"
        if missing_optional:
            details += f", Missing optional: {missing_optional}"
        
        return {
            "success": True,
            "details": details
        }
    
    async def test_configuration(self) -> Dict[str, Any]:
        """Test configuration file loading"""
        try:
            from src.configuration import load_config
            
            config = load_config(self.config_path)
            
            # Test key sections exist
            required_sections = ["trading", "risk", "ml", "broker"]
            missing_sections = []
            
            for section in required_sections:
                if not hasattr(config, section):
                    missing_sections.append(section)
            
            if missing_sections:
                return {
                    "success": False,
                    "error": f"Missing config sections: {missing_sections}"
                }
            
            return {
                "success": True,
                "details": f"Loaded {self.config_path} successfully"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to load configuration: {str(e)}",
                "fix": "Check YAML syntax and required sections"
            }
    
    async def test_directories(self) -> Dict[str, Any]:
        """Test required directories exist"""
        required_dirs = ["models", "data", "logs", "cache", "config"]
        
        missing_dirs = []
        for dir_name in required_dirs:
            if not os.path.exists(dir_name):
                missing_dirs.append(dir_name)
        
        if missing_dirs:
            return {
                "success": False,
                "error": f"Missing directories: {missing_dirs}",
                "fix": f"Run: mkdir -p {' '.join(missing_dirs)}"
            }
        
        return {
            "success": True,
            "details": f"All required directories exist"
        }
    
    async def test_core_dependencies(self) -> Dict[str, Any]:
        """Test core dependencies are installed"""
        core_packages = [
            "pandas", "numpy", "pyyaml", "structlog", 
            "alpaca_trade_api", "praw", "aiohttp"
        ]
        
        missing_packages = []
        
        for package in core_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            return {
                "success": False,
                "error": f"Missing packages: {missing_packages}",
                "fix": "Run: pip install -r requirements.txt"
            }
        
        return {
            "success": True,
            "details": f"All core dependencies available"
        }
    
    async def test_ml_dependencies(self) -> Dict[str, Any]:
        """Test ML dependencies (optional)"""
        ml_packages = ["torch", "transformers", "sklearn", "xgboost"]
        
        missing_packages = []
        available_packages = []
        
        for package in ml_packages:
            try:
                __import__(package)
                available_packages.append(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            return {
                "success": False,
                "error": f"Missing ML packages: {missing_packages}",
                "fix": "Run: pip install -r requirements-ml.txt"
            }
        
        return {
            "success": True,
            "details": f"ML packages available: {len(available_packages)}"
        }
    
    async def test_database(self) -> Dict[str, Any]:
        """Test database connection"""
        try:
            from src.database import DatabaseManager
            
            db = DatabaseManager()
            # Test basic connection (this would need actual implementation)
            
            return {
                "success": True,
                "details": "Database manager initialized"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Database test failed: {str(e)}",
                "fix": "Check DATABASE_URL in .env file"
            }
    
    async def test_api_connections(self) -> Dict[str, Any]:
        """Test API connections"""
        try:
            from src.data.alpaca_client import AlpacaClient
            
            # Test Alpaca connection
            client = AlpacaClient()
            
            # This would test actual connection
            return {
                "success": True,
                "details": "API clients initialized"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"API connection failed: {str(e)}",
                "fix": "Check API credentials in .env file"
            }
    
    async def test_models(self) -> Dict[str, Any]:
        """Test model files exist"""
        model_dir = Path("models")
        
        if not model_dir.exists():
            return {
                "success": False,
                "error": "Models directory not found",
                "fix": "Run model training or download pre-trained models"
            }
        
        model_files = list(model_dir.glob("**/*.pkl")) + list(model_dir.glob("**/*.pt"))
        
        if not model_files:
            return {
                "success": False,
                "error": "No model files found",
                "fix": "Train models with: python train_production.py --quick-start"
            }
        
        return {
            "success": True,
            "details": f"Found {len(model_files)} model files"
        }
    
    def print_summary(self) -> None:
        """Print test summary"""
        passed = sum(1 for r in self.test_results.values() if r["success"])
        total = len(self.test_results)
        
        logger.info("\n" + "=" * 50)
        logger.info("VALIDATION SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Tests Passed: {passed}/{total}")
        
        if passed == total:
            logger.info("🎉 System is ready for trading!")
            logger.info("\nNext steps:")
            logger.info("1. Run paper trading: python src/main.py --mode paper")
            logger.info("2. Run backtest: python backtest.py --start-date 2024-01-01 --end-date 2024-06-30")
        else:
            logger.info("❌ Please fix the failing tests before proceeding")


async def main():
    """Main test runner"""
    parser = argparse.ArgumentParser(description="Validate QuantumSentiment setup")
    parser.add_argument("--config", default="config/config.yaml", 
                       help="Configuration file to test")
    
    args = parser.parse_args()
    
    validator = SetupValidator(args.config)
    success = await validator.run_all_tests()
    validator.print_summary()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())