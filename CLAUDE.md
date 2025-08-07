algorithmic-trading-bot/
├── config/                    # Configuration files
│   ├── config.yaml           # Main production config
│   ├── config_small_data.yaml # Limited data config
│   └── download_config.yaml  # Data download settings
├── src/                      # Source code
│   ├── main.py              # Main trading bot entry point
│   ├── backtesting/         # Backtesting engine
│   ├── broker/              # Trading broker integrations
│   ├── data/                # Data fetching and management
│   ├── features/            # Feature engineering
│   ├── models/              # ML models and ensemble
│   ├── portfolio/           # Portfolio optimization
│   ├── risk/                # Risk management
│   ├── sentiment/           # Sentiment analysis
│   └── training/            # Model training pipeline
├── scripts/                 # Utility scripts
│   ├── download_quality_data.py  # Download training data
│   └── prepare_quality_data.py   # Data preprocessing
├── models/                  # Trained model storage
├── data/                    # Historical data storage
├── logs/                    # Application logs
├── backups/                 # Configuration backups
├── cache/                   # Temporary cache
├── backtest.py             # Backtest entry point
├── train_production.py     # Model training entry point
├── requirements.txt        # Core dependencies
├── requirements-ml.txt     # ML dependencies
└── README.md             
```

### training
```bash
python training/train_simple_massive.py 
# or 
python training/train_binary_massive.py
```

### Model Performance

| Model | Accuracy | Performance vs Baseline | Status |
|-------|----------|------------------------|--------|
| **XGBoost** | **49.6%** | **+29.6%** | ✅ **Primary Model** |
| LSTM | ~41% | +21% | ✅ Supporting Model |
| CNN | ~41% | +21% | ✅ Pattern Recognition |  
| Ensemble | ~44% | +24% | ✅ Combined Prediction |

### Paper Trading (Recommended Start)

```bash
# Basic paper trading with default settings
python src/main.py --mode paper

# Paper trading with custom config
python src/main.py --mode paper --config config/config_small_data.yaml

# Semi-automatic mode (ask for confirmation on trades)
python src/main.py --mode semi_auto
```

### Backtesting

```bash
# Basic 6-month backtest
python backtest.py --start-date 2024-01-01 --end-date 2024-06-30

# Custom symbols and capital
python backtest.py --symbols AAPL GOOGL MSFT --start-date 2024-01-01 --end-date 2024-12-31 --capital 50000

# Using small data config
python backtest.py --config config/config_small_data.yaml --start-date 2024-01-01 --end-date 2024-06-30
```

### Model Training (Advanced)

```bash
# Quick training for testing (30 min)
python train_production.py --quick-start --symbols 10

# Full production training (3-6 hours)  
python train_production.py --symbols 50 --epochs 500

# Train specific models only
python train_production.py --models xgboost,lstm --symbols 30
```

## ⚙️ Configuration Guide

### Configuration Files

| File | Purpose | When to Use |
|------|---------|-------------|
| `config/config.yaml` | Production configuration | Live/Paper trading with full features |
| `config/config_small_data.yaml` | Limited data configuration | Testing, limited memory, quick validation |
| `config/download_config.yaml` | Data download settings | Customizing data collection |


