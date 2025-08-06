# QuantumSentiment Trading Bot

AI-powered trading system using sentiment analysis and machine learning for Alpaca paper trading.

## Quick Start

### 1. Environment Setup

```bash
# Clone repository
git clone <repository-url>
cd algorithmic-trading-bot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Create `.env` file with your API keys:
```env
ALPACA_API_KEY=your_api_key
ALPACA_SECRET_KEY=your_secret_key
REDDIT_CLIENT_ID=your_reddit_id
REDDIT_CLIENT_SECRET=your_reddit_secret
NEWS_API_KEY=your_news_api_key
```

### 3. Download Training Data

```bash
# Download massive dataset (130K+ rows, 104 symbols, 5 years)
python scripts/download_massive_data.py --symbols 104 --years 5

# Download quality-filtered data (recommended)
python scripts/download_quality_data.py --top 50
```

### 4. Train Models

```bash
# Train 5-class model (BEST PERFORMANCE: 49.6% accuracy, 29.6% above baseline)
python train_simple_massive.py --symbols 30 --target 0.65

# Alternative: Train binary model (54.7% accuracy, 4.7% above baseline)  
python train_binary_massive.py --symbols 50 --horizon 3
```

### 5. Run Trading Bot

```bash
# Paper trading mode (recommended)
python src/main.py --mode PAPER

# Backtest mode (currently broken - see TODO.md Phase 6)
python src/main.py --mode BACKTEST
```

## System Architecture

### Data Pipeline
- **Historical Data**: 5 years of hourly price data from Alpaca
- **Sentiment Sources**: Reddit (r/wallstreetbets, r/stocks) + News APIs
- **Features**: 119 technical indicators + sentiment scores

### Models
- **Price Prediction**: XGBoost (5-class direction + magnitude)
- **Sentiment Analysis**: FinBERT (financial sentiment)
- **Ensemble**: Stacked model combining all predictions

### Risk Management
- Position sizing: Kelly Criterion
- Max drawdown: 20%
- Stop loss: 2% per position
- Portfolio exposure limits

## Performance

### Individual Model Results

| Model | Type | Accuracy | Performance vs Baseline | Status |
|-------|------|----------|------------------------|---------|
| **XGBoost** | 5-class | **49.6%** | **+29.6%** | ✅ **Best - Use for trading** |
| LSTM | 5-class | ~41% | +21% | ❌ Overfitting |
| CNN | 5-class | ~41% | +21% | ❌ Poor performance |
| Ensemble | 5-class | ~44% | +24% | ⚠️ Marginal improvement |
| XGBoost | Binary | 54.7% | +4.7% | ⚠️ Low edge |

### How to Use Models

**Option 1: Single Model (Recommended)**
```bash
# Train only XGBoost (best performance)
python train_simple_massive.py --model xgboost --symbols 30

# Use in trading
python src/main.py --mode PAPER --model xgboost
```

**Option 2: Specific Models**
```bash
# Train specific models
python scripts/train_massive_data.py --models "xgboost,lstm" --symbols 30

# Use multiple models
python src/main.py --mode PAPER --models "xgboost,lstm"
```

**Option 3: All Models (Ensemble)**
```bash
# Train all models (slow, not recommended)
python scripts/train_massive_data.py --models "all" --symbols 30

# Use ensemble
python src/main.py --mode PAPER --model ensemble
```

### Model Selection Guide

- **For Production**: Use XGBoost only (49.6% accuracy, reliable)
- **For Research**: Test individual models to find improvements
- **Avoid**: CNN/LSTM alone (poor performance on financial data)
- **Ensemble Note**: Only marginally better than XGBoost alone

## File Structure

```
├── config/
│   └── config.yaml          # Main configuration
├── src/
│   ├── main.py             # Trading bot entry point
│   ├── data/               # Data fetching modules
│   ├── features/           # Feature engineering
│   ├── models/             # ML model implementations
│   ├── portfolio/          # Portfolio management
│   ├── risk/               # Risk management
│   └── sentiment/          # Sentiment analysis
├── scripts/
│   ├── download_massive_data.py    # Data download
│   └── train_massive_data.py       # Model training
└── train_simple_massive.py         # Simplified training
```

## Common Issues

### Memory Issues
- Reduce `--symbols` parameter when training
- Use `config_small_data.yaml` for testing

### Model Not Found
- Ensure models are trained before running main.py
- Check `models/` directory for saved models

## Development Status

**Working**: 
- ✅ Model training pipeline
- ✅ Data download and preprocessing  
- ✅ Feature engineering
- ✅ 49.6% 5-class accuracy achieved

**In Progress** (see TODO.md):
- ⚠️ Phase 3: Connect real sentiment to main.py
- ⚠️ Phase 4: Integrate advanced risk management
- ⚠️ Phase 6: Fix backtesting engine

## Next Steps

1. Complete Phase 3 from TODO.md (remove mock data)
2. Run extended paper trading tests
3. Monitor performance metrics
4. Implement Phase 4-6 improvements

## Support

See CLAUDE.md for development guidelines and TODO.md for detailed refactoring plan.