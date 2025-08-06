# QuantumSentiment Trading Bot - Quick Start

## Installation

```bash
# 1. Setup environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Configure API keys in .env file
cp .env.example .env  # Edit with your API keys
```

## Train Models

```bash
# Download training data
python training/scripts/download_massive_data.py --symbols 50 --years 3

# Train best performing model (49.6% accuracy, 29.6% above random)
python training/train_simple_massive.py --symbols 30
```

## Run Trading Bot

```bash
# Paper trading mode (recommended)
python src/main.py --mode paper

# Check performance
tail -f logs/trading.log
```

## Project Structure

```
├── src/main.py              # Main trading bot
├── training/                # Training scripts and data tools
│   ├── train_simple_massive.py    # Best model (49.6% accuracy)
│   └── scripts/download_*          # Data download tools
├── config/config.yaml       # Configuration
├── docs/                    # Detailed documentation
└── QUICK_START.md          # This file
```

## Performance

Current best model: **49.6% accuracy** (29.6% above 20% random baseline) with 5-class prediction using XGBoost on massive dataset.

## Status

✅ **Phase 3 Complete**: Real sentiment integration, model loading, efficient data fetching  
🔄 **Next**: Phase 4 - Advanced risk management and position sizing