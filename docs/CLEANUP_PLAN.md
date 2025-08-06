# Project Cleanup Plan

## Training Scripts Analysis

### ✅ KEEP - Primary Training Scripts
1. **train_simple_massive.py** - BEST PERFORMING (49.6% accuracy, 5-class)
   - Direct XGBoost training with massive dataset
   - Achieved 29.6% above random baseline
   - Simple, fast, reliable

2. **train_binary_massive.py** - Alternative approach (54.7% accuracy, binary)
   - Binary classification variant
   - Only 4.7% above random baseline
   - Keep for comparison

### 🗑️ DELETE - Outdated/Redundant Training Scripts
1. **train_production.py** - Complex pipeline with issues
   - Based on old architecture
   - Complex dependencies that fail
   - Superseded by simple_massive approach

2. **train_massive.py** - Incomplete implementation
   - Similar to simple_massive but less features
   - Redundant

3. **scripts/train_massive_data.py** - Complex pipeline variant
   - Similar issues as train_production.py
   - Redundant with working approaches

4. **scripts/train_progressive.py** - Experimental approach
   - Progressive training didn't show benefits
   - Adds unnecessary complexity

## Files to Clean Up

### 🗑️ DELETE - Old/Unused Directories
- `backups/` - Old code backups (superseded)
- `archive/` - Archived documentation 
- `checkpoints/` - Old training checkpoints
- `logs/` - Old training logs (hundreds of files)
- `test_models/` - Test model checkpoints
- `walkthroughs/` - Example code that's outdated

### 🗑️ DELETE - Redundant Model Directories
- `models/*/20250729_*` - Very old model versions (keep only latest 2-3)
- `models/*/20250730_*` - Old model versions
- `models/*/20250731_*` - Old model versions
- Keep only `models/*/20250803_*` (latest working models)

### 🗑️ DELETE - Unused Script Files
- `scripts/apply_model_fixes.py`
- `scripts/fix_model_performance.py` 
- `scripts/hyperparameter_optimizer.py`
- `scripts/test_*.py` (most testing scripts)
- Duplicate data files in `data/historical/*/` (keep only latest)

### ✅ KEEP - Essential Files
- Main documentation: `README.md`, `TODO.md`, `CLAUDE.md`
- Core source code: `src/` directory
- Configuration: `config/` directory  
- Latest training data: `data/massive/massive_training_data.parquet`
- Working training scripts: `train_simple_massive.py`, `train_binary_massive.py`

## Recommended File Structure After Cleanup

```
├── README.md
├── TODO.md  
├── CLAUDE.md
├── config/
│   └── config.yaml
├── src/           # Core application
├── data/
│   └── massive/   # Keep only latest training data
├── models/        # Keep only 2-3 latest versions
├── train_simple_massive.py    # PRIMARY training script
├── train_binary_massive.py    # Alternative training script
└── scripts/
    ├── download_massive_data.py
    └── validate_setup.py
```

This reduces project size by ~80% while keeping all essential functionality.