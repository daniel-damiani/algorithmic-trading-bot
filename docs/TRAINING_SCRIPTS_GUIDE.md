# Training Scripts Guide

## Primary Training Script ✅

### `train_simple_massive.py` - RECOMMENDED
**Best Performance: 49.6% accuracy (29.6% above random)**

```bash
# Train the best performing model
python train_simple_massive.py --symbols 30 --target 0.65

# Key features:
# - 5-class prediction (Strong Down, Down, Neutral, Up, Strong Up)
# - XGBoost classifier with 103 advanced features
# - Quality-based symbol selection
# - Simple, reliable, fast training
```

**Use this script for:**
- Production model training
- Best trading performance
- Reliable results

## Alternative Training Script ⚠️

### `train_binary_massive.py` - BACKUP OPTION
**Performance: 54.7% accuracy (4.7% above random)**

```bash
# Train binary classification model
python train_binary_massive.py --symbols 50 --horizon 3

# Key features:
# - Binary up/down prediction
# - Lower edge over random (4.7% vs 29.6%)
# - More data efficient
```

**Use this script for:**
- When you need binary signals only
- Testing different approaches
- Comparison with 5-class model

## Removed Training Scripts 🗑️

The following scripts were removed during cleanup:

- `train_production.py` - Complex pipeline with dependency issues
- `train_massive.py` - Redundant, fewer features than simple_massive
- `scripts/train_massive_data.py` - Complex pipeline variant
- `scripts/train_progressive.py` - Experimental approach that didn't improve results

## Quick Start

```bash
# 1. Download training data (if not done yet)
python scripts/download_massive_data.py --symbols 104 --years 5

# 2. Train the best model
python train_simple_massive.py --symbols 30

# 3. Check results in EXPERIMENT_LOG.md
cat EXPERIMENT_LOG.md

# 4. Run paper trading
python src/main.py --mode paper
```

## Model Comparison

| Script | Accuracy | Above Random | Use Case |
|--------|----------|--------------|----------|
| `train_simple_massive.py` | 49.6% | +29.6% | **Production** |
| `train_binary_massive.py` | 54.7% | +4.7% | Alternative |

**Recommendation:** Always use `train_simple_massive.py` for real trading as it has significantly better performance relative to chance.