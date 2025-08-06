#!/bin/bash
# Script to download extended historical data for training

echo "📊 Downloading Extended Historical Data for Training"
echo "=================================================="

# Define quality symbols
SYMBOLS="AAPL,MSFT,GOOGL,AMZN,NVDA,META,TSLA,BRK.B,JPM,JNJ,V,PG,UNH,HD,MA,BAC,XOM,DIS,CVX,ABBV,KO,PEP,MRK,WMT,PFE,TMO,CSCO,VZ,CMCSA,ADBE,NFLX,AMD,CRM,BA,SPY,QQQ,IWM,DIA,XLF,XLE,XLK,XLV"

# Use the existing download script
echo "Downloading 2 years of data for quality symbols..."
python scripts/download_historical_data.py \
    --symbols "$SYMBOLS" \
    --days 730 \
    --timeframe "1Hour"

echo ""
echo "✅ Data download complete!"
echo ""
echo "Next steps:"
echo "1. Check the downloaded data in the data/ directory"
echo "2. Run training with the new data:"
echo "   python src/train_models.py --days 730 --models all"