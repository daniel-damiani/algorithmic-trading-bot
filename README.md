# QuantumSentiment Trading Bot 🚀

**Production-Ready Algorithmic Trading System** with sentiment analysis, machine learning predictions, and comprehensive risk management. Completely refactored from a non-functional prototype into a battle-tested trading system.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Production Ready](https://img.shields.io/badge/Status-Production%20Ready-green.svg)]()

## ⚡ Super Quick Start (5 Minutes)

**New to the system? Use the automated setup:**

```bash
# Clone and enter directory
git clone <repository-url>
cd algorithmic-trading-bot

# Create virtual environment (Windows: use python.org Python, not MSYS2)
py -3.11 -m venv .venv   # Windows if MSYS `python` is first on PATH
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Automated setup (installs dependencies, creates directories, etc.)
python quick_start.py

# Follow the prompts to set up your .env file
# Then test immediately:
python backtest.py --start-date 2024-01-01 --end-date 2024-06-30
```

## 🎯 Manual Setup (Advanced Users)

### Prerequisites
- Python 3.8+ (3.10+ recommended)
- 4GB+ RAM (8GB+ recommended for ML training)
- Alpaca Trading Account (Paper/Live)
- Reddit API credentials (optional but recommended)

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd algorithmic-trading-bot

# Create virtual environment (Windows: use python.org Python, not MSYS2)
py -3.11 -m venv .venv   # Windows if MSYS `python` is first on PATH
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install core dependencies
pip install -r requirements.txt

# Install ML dependencies (for training models)
pip install -r requirements-ml.txt
```

### 2. Environment Configuration

Create `.env` file in the project root:

```env
# === REQUIRED: Alpaca API (Paper Trading) ===
ALPACA_API_KEY=your_alpaca_api_key_here
ALPACA_API_SECRET=your_alpaca_secret_here
ALPACA_BASE_URL=https://paper-api.alpaca.markets

# === REQUIRED: Database ===
DATABASE_URL=postgresql://username:password@localhost:5432/quantumsentiment
# Alternative: DATABASE_URL=sqlite:///quantumsentiment.db

# === OPTIONAL: Sentiment Analysis ===
# Reddit API (highly recommended)
REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_CLIENT_SECRET=your_reddit_client_secret
REDDIT_USER_AGENT=QuantumSentiment/1.0

# News APIs (optional)
NEWSAPI_KEY=your_newsapi_key
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key

# === OPTIONAL: Alerts ===
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_telegram_chat_id
```

### 3. Database Setup

```bash
# Option 1: PostgreSQL (Recommended for production)
# Install PostgreSQL and create database
createdb quantumsentiment

# Option 2: SQLite (Quick start)
# Just set DATABASE_URL=sqlite:///quantumsentiment.db in .env
```

### 4. Initial Setup

```bash
# Create required directories
mkdir -p models data logs backups cache

# Download and prepare training data (optional - models included)
python scripts/download_quality_data.py --symbols 30

# Train models (optional - pre-trained models included)
python train_production.py --quick-start
```

## 🏃‍♂️ Running the System

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
python training/train_simple_massive.py 
# or 
python training/train_binary_massive.py
```

## ⚙️ Configuration Guide

### Configuration Files

| File | Purpose | When to Use |
|------|---------|-------------|
| `config/config.yaml` | Production configuration | Live/Paper trading with full features |
| `config/config_small_data.yaml` | Limited data configuration | Testing, limited memory, quick validation |
| `config/download_config.yaml` | Data download settings | Customizing data collection |

### Trading Modes

```yaml
# In config.yaml
trading:
  strategy_mode: "adaptive"  # Options: adaptive, technical_only, sentiment_only, conservative
```

#### Strategy Mode Details:

- **`adaptive`** ✅ **Recommended**: Uses any available signals, adapts to data availability
- **`technical_only`**: Requires technical indicators only (works without sentiment APIs) 
- **`sentiment_only`**: Requires sentiment data (Reddit/News APIs needed)
- **`conservative`**: Requires both technical AND sentiment confirmation (highest confidence)

### Risk Management Configuration

```yaml
risk:
  max_drawdown: 0.10        # 10% maximum portfolio drawdown
  daily_loss_limit: 0.03   # 3% daily loss limit  
  stop_loss_pct: 0.02       # 2% stop loss per position
  take_profit_pct: 0.05     # 5% take profit per position
  max_positions: 10         # Maximum concurrent positions
  max_position_size: 0.10   # 10% max per position
```

### Position Sizing (Kelly Criterion)

```yaml
# Advanced position sizing with Kelly Criterion
# Automatically enabled - calculates optimal position sizes
position_sizer:
  use_kelly_criterion: true
  kelly_fraction: 0.25      # Use 25% of Kelly recommendation (safety factor)
  max_position_size: 0.10   # Hard cap at 10% per position
```

## 🧠 Machine Learning Models

### Pre-trained Models (Included)

The system includes pre-trained models ready for paper trading:

- **XGBoost**: Primary trading model (49.6% accuracy, +29.6% vs baseline)  
- **LSTM**: Price sequence prediction
- **CNN**: Chart pattern recognition
- **FinBERT**: Sentiment analysis
- **Ensemble**: Stacked combination of all models

### Model Performance

| Model | Accuracy | Performance vs Baseline | Status |
|-------|----------|------------------------|--------|
| **XGBoost** | **49.6%** | **+29.6%** | ✅ **Primary Model** |
| LSTM | ~41% | +21% | ✅ Supporting Model |
| CNN | ~41% | +21% | ✅ Pattern Recognition |  
| Ensemble | ~44% | +24% | ✅ Combined Prediction |

### Training Your Own Models

```bash
# Quick training (1 hour, good for testing)
python train_production.py --quick-start

# Production training (4-6 hours, best performance)  
python train_production.py --symbols 50 --epochs 500 --use-all-data

# Memory-efficient training
python train_production.py --config config/config_small_data.yaml --symbols 20
```

## 📊 System Architecture

### Data Pipeline
```
Historical Data → Feature Engineering → Model Prediction → Signal Validation → Risk Assessment → Position Sizing → Order Execution
     ↓                    ↓                   ↓                ↓                ↓               ↓              ↓
  Alpaca API         119+ Features      XGBoost/LSTM      Strategy Rules    VaR Analysis   Kelly Criterion  Simulated/Live
```

### Core Components

1. **Data Sources**: Alpaca (prices), Reddit (sentiment), News APIs (sentiment)
2. **Feature Engineering**: 119+ technical indicators + sentiment scores
3. **ML Pipeline**: XGBoost primary, LSTM/CNN supporting, FinBERT sentiment
4. **Risk Management**: Kelly Criterion sizing, VaR limits, stop-loss protection
5. **Execution**: Smart order routing with slippage/commission simulation

### Real-time Processing
- **Market Data**: Updated every minute
- **Sentiment Analysis**: Updated every 5 minutes
- **Model Predictions**: Updated every 15 minutes
- **Risk Checks**: Continuous monitoring
- **Position Management**: Real-time stop-loss/take-profit

## 📈 Performance & Risk Metrics

### Key Performance Indicators
- **Sharpe Ratio**: Risk-adjusted returns
- **Sortino Ratio**: Downside risk focus
- **Max Drawdown**: Worst portfolio decline
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Gross profit / Gross loss

### Risk Controls
- **Portfolio Level**: Max drawdown (10%), daily loss limit (3%)
- **Position Level**: Stop-loss (2%), take-profit (5%), position size limits
- **Correlation Limits**: Max correlation between positions
- **Sector Concentration**: Limits on sector exposure

## 🔧 Advanced Usage

### Custom Strategy Development

1. **Create Strategy Module**:
```python
# src/strategies/my_strategy.py
class MyCustomStrategy:
    def validate_signal(self, signal):
        # Custom signal validation logic
        return signal['confidence'] > 0.8
```

2. **Configure Strategy**:
```yaml
# config/config.yaml
trading:
  strategy_mode: "custom"
  custom_strategy: "my_strategy"
```

### LAN Dashboard

Web UI for monitoring and control from any device on your local network (phone, tablet, another PC).

```powershell
# Windows
.\scripts\run_dashboard.ps1

# Or directly
python -m src.api.server
```

Open on this machine: `http://localhost:8000`

Open from another device on the same Wi‑Fi: `http://<your-pc-ip>:8000` (find IP with `ipconfig` on Windows).

Set a control API key in `.env` (required to start/stop the bot or run backtests from the UI):

```env
DASHBOARD_API_KEY=choose-a-long-random-string
```

Enter that key once in the dashboard **Control API key** field (stored in the browser only).

**Windows Firewall:** allow inbound TCP port **8000** on private networks if other devices cannot connect.

**AI Assistant:** Install [Ollama](https://ollama.com), run `ollama pull llama3.2:1b`, then use the **Assistant** tab (model picker shows all pulled models). Optional: `OLLAMA_MODEL` / `OLLAMA_BASE_URL` in `.env`. For Alpaca MCP in Cursor, see the dashboard **Help** tab.

### API Integration

REST endpoints used by the dashboard (also available via curl):

```bash
# Start dashboard server
python -m src.api.server

# Health check
curl http://localhost:8000/api/health

# Overview (account + bot status)
curl http://localhost:8000/api/overview

# Positions
curl http://localhost:8000/api/positions

# Start paper bot (requires X-API-Key header)
curl -X POST http://localhost:8000/api/bot/start -H "X-API-Key: YOUR_KEY" -H "Content-Type: application/json" -d "{}"
```

Interactive API docs: `http://localhost:8000/docs`

### Monitoring & Alerts

#### Telegram Alerts (Optional)
```env
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

## 🛠️ Troubleshooting

### Common Issues

#### 1. Model Not Found Error
```bash
# Error: No trained models found
# Solution: Train models or check model path
python train_production.py --quick-start
```

#### 2. API Authentication Error
```bash
# Error: Alpaca authentication failed
# Solution: Check .env file has correct API keys
# Verify: ALPACA_API_KEY and ALPACA_API_SECRET are set
```

#### 3. Memory Issues During Training
```bash
# Error: Out of memory during model training
# Solution: Use smaller configuration
python train_production.py --config config/config_small_data.yaml --symbols 10
```

#### 4. Redis Connection Error
```yaml
# In config.yaml, change cache type to memory:
cache:
  type: "memory"  # Instead of "redis"
```

#### 5. Database Connection Issues
```env
# Try SQLite instead of PostgreSQL:
DATABASE_URL=sqlite:///quantumsentiment.db
```

### Debug Mode

```bash
# Run with debug logging
python src/main.py --mode paper --log-level DEBUG

# Enable profiling
python src/main.py --mode paper --profile
```

### Performance Tuning

```yaml
# In config.yaml - reduce frequency for slower systems:
scheduler:
  market_data_update: "*/5 * * * *"    # Every 5 minutes instead of 1
  sentiment_update: "*/15 * * * *"     # Every 15 minutes instead of 5
  model_prediction: "*/30 * * * *"     # Every 30 minutes instead of 15
```

## 📁 Directory Structure

```
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
└── README.md              # This file
```

## 📚 Documentation

### Additional Resources
- **`CLAUDE.md`**: Development guidelines and system context
- **`TODO.md`**: Completed refactoring plan and system analysis  
- **`NOTES.md`**: Technical decisions and implementation notes
- **`TRAINING_GUIDE.md`**: Detailed model training instructions

### API Documentation
- **Web dashboard**: `http://localhost:8000` (see LAN Dashboard section above)
- **OpenAPI docs**: `http://localhost:8000/docs` (when dashboard server is running)
- **Configuration**: See `config/config.yaml` with inline comments
- **Model Architecture**: See `src/models/` for implementation details

## 🧪 Testing

```bash
# Run unit tests
pytest tests/

# Run integration tests
pytest tests/integration/

# Test configuration validation
python -c "from src.configuration import load_config; print('Config OK')"

# Test database connection
python -c "from src.database import DatabaseManager; db = DatabaseManager(); print('DB OK')"

# Test API credentials
python scripts/test_apis.py
```

## 🚀 Deployment

### Local Production Deployment

1. **Setup Production Environment**:
```bash
# Use production config
cp config/config.yaml config/config_production.yaml

# Set production environment
export ENVIRONMENT=production
```

2. **Run with Process Manager**:
```bash
# Using systemd (Linux)
sudo systemctl enable quantumsentiment.service
sudo systemctl start quantumsentiment.service

# Using PM2 (Node.js process manager)
pm2 start ecosystem.config.js
```

3. **Monitoring**:
```bash
# View logs
tail -f logs/trading.log

# Monitor performance
python scripts/monitor_performance.py
```

### Cloud Deployment (Advanced)

The system can be deployed on cloud platforms:

- **AWS**: EC2 + RDS + ElastiCache
- **Google Cloud**: Compute Engine + Cloud SQL + Memorystore
- **Azure**: Virtual Machines + Database + Cache
- **Docker**: Containerized deployment available

## ⚠️ Risk Disclaimer

**IMPORTANT**: This software is for educational and research purposes. Algorithmic trading involves substantial risk of financial loss. 

- **Start with Paper Trading**: Always test thoroughly before risking real money
- **Understand the Risks**: Past performance does not guarantee future results
- **Monitor Continuously**: Automated systems require active monitoring
- **Risk Management**: Never risk more than you can afford to lose

The authors are not responsible for any financial losses incurred from using this software.

## 🤝 Contributing

1. **Fork the Repository**
2. **Create Feature Branch**: `git checkout -b feature/amazing-feature`
3. **Commit Changes**: `git commit -m 'Add amazing feature'`
4. **Push to Branch**: `git push origin feature/amazing-feature`
5. **Open Pull Request**

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run linting and tests
ruff check src/
black src/
pytest tests/
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/your-repo/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-repo/discussions)  
- **Documentation**: Check `CLAUDE.md` and inline code comments
- **Configuration Help**: Review `config/config.yaml` comments

---

**Happy Trading! 📈🎯**

*Built with ❤️ by the QuantumSentiment team*