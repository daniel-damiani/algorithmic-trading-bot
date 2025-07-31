# QuantumSentiment Trading Bot - Setup Guide

## 🚀 **Quick Start**

### **1. Set Up API Credentials**

You need Alpaca Markets API credentials to download market data and execute trades.

1. **Sign up for Alpaca Markets:**
   - Go to https://alpaca.markets/
   - Create a free account
   - Navigate to your dashboard and get your API credentials

2. **Create a `.env` file in the project root:**
   ```bash
   # Create .env file
   touch .env
   ```

3. **Add your credentials to `.env`:**
   ```bash
   # Alpaca API Credentials (Paper Trading)
   ALPACA_API_KEY=your_api_key_here
   ALPACA_API_SECRET=your_secret_key_here
   ALPACA_BASE_URL=https://paper-api.alpaca.markets

   # Reddit API (Optional - for sentiment analysis)
   REDDIT_CLIENT_ID=your_reddit_client_id
   REDDIT_CLIENT_SECRET=your_reddit_client_secret
   REDDIT_USER_AGENT=QuantumSentiment:1.0

   # NewsAPI (Optional - for news sentiment)
   NEWSAPI_KEY=your_news_api_key

   # Database (Optional - defaults to SQLite)
   DATABASE_URL=sqlite:///quantumsentiment.db
   ```

### **2. Install Dependencies**

Make sure you have Python 3.8+ and activate your virtual environment:

```bash
# Activate virtual environment
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate     # On Windows

# Install requirements
pip install -r requirements.txt
```

### **3. Download Historical Data**

```bash
# Download 4+ years of historical data for training
python scripts/download_historical_data.py
```

This will download data for: AAPL, GOOGL, MSFT, TSLA, NVDA, SPY, QQQ, META, AMZN

### **4. Test the Training Pipeline**

```bash
# Check data availability
python scripts/test_training.py --check-data

# Run a quick test
python scripts/test_training.py --quick

# Full training test
python scripts/test_training.py
```

### **5. Train All Models**

```bash
# Train all models with current data
python src/train_models.py
```

---

## 📊 **Expected Results**

### **With Downloaded Data:**
- **PriceLSTM**: R² > 0.4, Directional Accuracy > 60%
- **ChartPatternCNN**: Pattern Recognition Accuracy > 70%
- **MarketRegimeXGBoost**: Regime Classification Accuracy > 65%
- **FinBERT**: Sentiment Classification Accuracy > 75%
- **Ensemble**: Overall Performance > Individual Models

### **Training Time:**
- **Small Data** (<1000 samples): 5-10 minutes
- **Medium Data** (1000-20000 samples): 20-40 minutes  
- **Large Data** (20000+ samples): 1-3 hours

---

## 🔧 **Troubleshooting**

### **Common Issues:**

1. **"Missing Alpaca API credentials"**
   - Ensure `.env` file exists with correct credentials
   - Check that API key and secret are valid

2. **"No data retrieved"**
   - Check internet connection
   - Verify Alpaca API status
   - Try different symbols or date ranges

3. **Training fails with small data**
   - Models automatically adapt to data size
   - Download more historical data for better performance

4. **Memory issues during training**
   - Reduce batch size in configuration
   - Close other applications
   - Consider using CPU instead of GPU for small models

### **Performance Optimization:**

1. **For Better Results:**
   ```bash
   # Download more data (recommended)
   python scripts/download_historical_data.py
   
   # Use the enhanced configuration
   cp config/config.yaml config/config_backup.yaml
   # Then update config.yaml with settings from MODEL_IMPROVEMENT_STRATEGY.md
   ```

2. **For Faster Training:**
   - Reduce epochs in configuration (50-100 instead of 300-500)
   - Use smaller batch sizes
   - Disable complex features

---

## 📈 **Next Steps**

1. **Paper Trading:**
   ```bash
   python src/main.py --mode paper
   ```

2. **Backtesting:**
   ```bash
   python src/backtest.py --start 2023-01-01 --end 2024-01-01
   ```

3. **Live Monitoring:**
   ```bash
   python src/dashboard.py
   ```

---

## 💡 **Tips for Success**

1. **Start Small:** Begin with paper trading and a few symbols
2. **Monitor Performance:** Check model metrics and adjust as needed
3. **Regular Retraining:** Retrain models weekly/monthly with new data
4. **Risk Management:** Always use proper position sizing and stop losses
5. **Stay Updated:** Monitor market conditions and model performance

---

## 🆘 **Support**

If you encounter issues:

1. Check the logs in `logs/` directory
2. Review the FIXES_SUMMARY.md for common solutions
3. Ensure all dependencies are installed correctly
4. Verify API credentials and network connectivity

---

**Ready to start trading with AI! 🤖📈**