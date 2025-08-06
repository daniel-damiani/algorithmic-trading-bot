# QuantumSentiment Experiment Log

## Objective
Achieve production-ready trading models with >60% accuracy for real trading.

## Experiment Tracking

### Experiment #1 - Baseline (2025-08-05 12:13)
- **Data**: 5 symbols, 1000 rows each (5k total)
- **Models**: LSTM, XGBoost, Ensemble (CNN failed)
- **Results**:
  - LSTM: Val loss 0.93 (overfitting)
  - XGBoost: F1 0.195 (poor)
  - Ensemble: Accuracy 41.5%
- **Issues**: CNN config error, poor performance
- **Status**: ❌ Failed

### Experiment #2 - Fixed CNN (2025-08-05 12:29)
- **Data**: 3 symbols, 1000 rows each (3k total)
- **Models**: All models trained successfully
- **Results**:
  - LSTM: Val loss 1.03
  - CNN: Val F1 0.24, Accuracy 41.2%
  - XGBoost: F1 0.44 (improved!)
  - Ensemble: Accuracy 42.9%
- **Improvements**: Fixed CNN config, all models working
- **Status**: ❌ Still below target

## Next Steps
1. Massively expand data (100+ symbols, 5+ years)
2. Implement hyperparameter optimization
3. Add advanced features
4. Track all experiments systematically

## Performance Targets
- Minimum: 60% accuracy
- Target: 65%+ accuracy
- Stretch: 70%+ accuracy
### Experiment #progressive_1_5symbols - Progressive Training (2025-08-05 13:41)
- **Data**: 5,000 rows, 5 symbols
- **Configuration**: Small test (5 symbols)
- **Results**:
  - Training: FAILED (Traceback (most recent call last):
  File "/Users/karlovrancic/Documents/projects/algorithmic-trading-bot/train_production.py", line 32, in <module>
    import structlog
ModuleNotFoundError: No module named 'structlog'
)
- **Status**: ❌ Poor Performance

### Experiment #progressive_2_10symbols - Progressive Training (2025-08-05 13:41)
- **Data**: 10,000 rows, 10 symbols
- **Configuration**: Medium test (10 symbols)
- **Results**:
  - Training: FAILED (Traceback (most recent call last):
  File "/Users/karlovrancic/Documents/projects/algorithmic-trading-bot/train_production.py", line 32, in <module>
    import structlog
ModuleNotFoundError: No module named 'structlog'
)
- **Status**: ❌ Poor Performance

### Experiment #progressive_3_20symbols - Progressive Training (2025-08-05 13:41)
- **Data**: 20,000 rows, 20 symbols
- **Configuration**: Large test (20 symbols)
- **Results**:
  - Training: FAILED (Traceback (most recent call last):
  File "/Users/karlovrancic/Documents/projects/algorithmic-trading-bot/train_production.py", line 32, in <module>
    import structlog
ModuleNotFoundError: No module named 'structlog'
)
- **Status**: ❌ Poor Performance

### Experiment #progressive_4_40symbols - Progressive Training (2025-08-05 13:41)
- **Data**: 40,000 rows, 40 symbols
- **Configuration**: Very large test (40 symbols)
- **Results**:
  - Training: FAILED (Traceback (most recent call last):
  File "/Users/karlovrancic/Documents/projects/algorithmic-trading-bot/train_production.py", line 32, in <module>
    import structlog
ModuleNotFoundError: No module named 'structlog'
)
- **Status**: ❌ Poor Performance

### Experiment #3 - Working Model Training (2025-08-05 13:42)
- **Data**: 3,000 rows, 3 symbols
- **Models**: LSTM, CNN, XGBoost, Ensemble
- **Results**:
  - LSTM: Val loss 1.045 (early stop epoch 16)
  - CNN: Val F1 0.240, Accuracy 41.2% (early stop epoch 11)
  - XGBoost: F1 0.439 (93 iterations)
  - Ensemble: Accuracy 42.9%
- **Best Model**: Ensemble (42.9%)
- **Status**: ⚠️ Needs Improvement - Working baseline established

### Experiment #4 - Scaled Training (2025-08-05 13:45)
- **Data**: 10,000 rows, 10 symbols  
- **Models**: LSTM, CNN, XGBoost, Ensemble
- **Results**:
  - LSTM: Val loss 0.828 (28 epochs), R² -1.09
  - CNN: Val F1 0.374, Accuracy 40% (16 epochs)
  - XGBoost: F1 0.144 (5 iterations) - Poor performance
  - Ensemble: Accuracy ~44% (calculated from outputs)
- **Best Model**: Ensemble (44%)
- **Status**: ⚠️ Slight improvement with more data but still far from target

### Experiment #enhanced_massive - Enhanced Massive Training (2025-08-05 13:51)
- **Data**: 37,620 rows, 30 symbols (massive dataset with quality filtering)
- **Features**: 103 enhanced features (technical + volume + momentum)
- **Target**: 5-class future returns prediction (5-day horizon)
- **Model**: Enhanced XGBoost (1000 trees, depth 8, regularized)
- **Results**:
  - Validation Accuracy: 0.4956 (49.6%)
  - Validation F1 Score: 0.4914 (49.1%)
  - Target: 65.0%
- **Status**: ❌ Poor Performance

### Experiment #binary_massive - Binary Classification (2025-08-05 13:53)
- **Data**: 62,700 rows, 50 symbols (quality-filtered massive dataset)
- **Features**: 119 advanced trading features
- **Target**: Binary up/down prediction (3-day horizon)  
- **Model**: Optimized XGBoost Binary Classifier (1500 trees, depth 6)
- **Results**:
  - Accuracy: 0.5470 (54.7%)
  - F1 Score: 0.2688 (26.9%)
  - AUC-ROC: 0.5780 (57.8%)
  - Target: 65.0%
- **Status**: ⚠️ Needs Improvement
