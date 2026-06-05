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

# Or start/stop from the LAN dashboard (see below)
python -m src.api.server
```

## LAN Dashboard

Monitor account, positions, models, and run backtests from a browser — including phones on the same Wi‑Fi.

```powershell
# Windows
.\scripts\run_dashboard.ps1

# macOS/Linux
python -m src.api.server
```

1. Add to `.env`: `DASHBOARD_API_KEY=your-secret-key`
2. Open `http://localhost:8000` or `http://<your-pc-ip>:8000` from another device
3. Save your API key in the dashboard to use **Start/Stop** and **Run backtest**

If other devices cannot connect, allow inbound TCP **8000** in Windows Firewall (private network only).

### AI Assistant (Ollama)

Ask live questions about paper trades, bot status, and how the app works:

1. Install [Ollama](https://ollama.com) on the PC running the dashboard
2. `ollama pull llama3.2:1b` (default; change model in the Assistant tab dropdown)
3. Open the dashboard → **Assistant** tab

Optional env: `OLLAMA_BASE_URL=http://127.0.0.1:11434`. For Alpaca MCP inside **Cursor**, see the **Help** tab.

```
├── src/main.py              # Main trading bot
├── src/api/                 # LAN dashboard (FastAPI + static UI)
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