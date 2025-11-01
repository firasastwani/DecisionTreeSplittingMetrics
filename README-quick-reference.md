# QUICK REFERENCE: Decision Tree Splitting Metrics

## 🎯 All 6 Metrics Overview

Your implementation should support these metrics:

1. **correlation** - Linear relationship strength (baseline)
2. **gini** - Distribution purity using binning (StatQuest method)
3. **variance_reduction** - Variance minimization (standard CART)
4. **mse_reduction** - Mean squared error minimization
5. **mae_reduction** - Mean absolute error minimization (robust)
6. **information_gain** - Entropy reduction with binning

---

## 🎮 Two Evaluation Modes

Your `main.py` supports two modes via the `--test-set` flag:

### 📊 Mode 1: Insample (Default - Good for Visualization)
```bash
python main.py --data data/bikes.csv -i --leaf-size 1 --max-depth 4 --metric gini
```

**What it does:**
- Trains on **ALL 729 samples** (full dataset)
- **Use for:** Tree visualization, understanding splits, debugging

**Expected Results (Bikes Dataset, leaf_size=1, max_depth=4):**
```
correlation         → Train RMSE: ~218.18
gini                → Train RMSE: ~239.83
variance_reduction  → Train RMSE: ~217.56
mse_reduction       → Train RMSE:  ? (you tell us!)
mae_reduction       → Train RMSE: ~217.78
information_gain    → Train RMSE: ~241.24
```

### 🎯 Mode 2: Outsample (Proper Evaluation)
```bash
python main.py --data data/bikes.csv -i --leaf-size 1 --max-depth 4 --metric gini --test-set outsample
```

**What it does:**
- Trains on **583 samples** (80% of data, randomly shuffled with seed=42)
- Tests on **146 samples** (remaining 20%)
- **Use for:** Proper model evaluation, matching unit tests

**Expected Results (Bikes Dataset, leaf_size=1, max_depth=4):**
```
correlation         → Train RMSE: ~214.59
gini                → Train RMSE: ~239.70
variance_reduction  → Train RMSE: ~214.17
mse_reduction       → Train RMSE: ~214.17
mae_reduction       → Train RMSE: ~214.12
information_gain    → Train RMSE: ~233.23
```

### 🤔 Why Different Results?

**Insample Mode:** More training data (729 samples) but same max_depth=4
- Tree must fit more diverse patterns with limited depth
- Results in slightly **higher** training RMSE

**Outsample Mode:** Less training data (583 samples) 
- Tree can fit these specific samples better
- Results in slightly **lower** training RMSE
- **But** this is more realistic for evaluating generalization!

---

## 🔥 FOCUS: Gini Impurity (StatQuest Method)

### ⚠️ Key Difference: Gini ≠ Variance

**Traditional Gini for Regression** (NOT what we use):
- Uses variance formula
- Same results as variance_reduction
- Train RMSE: ~217.56 (insample) or ~214.17 (outsample)

**StatQuest Gini** (WHAT WE USE):
- Bins Y values into quartiles for continuous data
- Uses Gini impurity formula: 1 - Σ(p²)
- Tries multiple thresholds (adjacent averages)
- Train RMSE: ~239.83 (insample) or ~239.70 (outsample)

### 🎓 Why This Method?

This demonstrates a crucial ML lesson: **different optimization objectives produce different results.**

Gini performs ~10% worse (RMSE ~239 vs ~217) because it optimizes for 
**distribution purity**, not **prediction error**. This shows that 
metric choice has real consequences!

### ✅ Verification Checklist

**Your Gini implementation is correct if:**
- ✅ Train RMSE ≈ 239.83 (insample) or 239.70 (outsample)
- ✅ Gini RMSE ≠ Variance RMSE (should differ by ~25 points)
- ✅ Gini RMSE ≈ Information Gain RMSE (both use binning, should be within 10 points)

**Your Gini implementation is WRONG if:**
- ❌ Train RMSE ≈ 217.56 (insample) or 214.17 (outsample) - you're using variance!
- ❌ Gini RMSE = Variance RMSE - you're not implementing Gini correctly

---

## 📊 Quick Comparison Table

### Insample Mode (Full Dataset: 729 samples)

| Metric | Train RMSE | Key Characteristic |
|--------|------------|-------------------|
| **correlation** | ~218.18 | Linear relationship |
| **gini** | ~239.83 | Bins Y, tries multiple thresholds |
| **variance_reduction** | ~217.56 | Minimizes variance |
| **mse_reduction** | ~217.56 | Same as variance mathematically |
| **mae_reduction** | ~217.78 | Robust to outliers |
| **information_gain** | ~241.24 | Bins Y, uses entropy |

### Outsample Mode (Train: 583, Test: 146 samples)

| Metric | Train RMSE | Key Characteristic |
|--------|------------|-------------------|
| **correlation** | ~214.59 | Linear relationship |
| **gini** | ~239.70 | Bins Y, tries multiple thresholds |
| **variance_reduction** | ~214.17 | Minimizes variance |
| **mse_reduction** | ~214.17 | Same as variance mathematically |
| **mae_reduction** | ~214.12 | Robust to outliers |
| **information_gain** | ~233.23 | Bins Y, uses entropy |

### Key Observations:

- **Gini ≈ Information Gain** (~239-241 insample, ~233-240 outsample): Both bin Y values
- **MSE = Variance = MAE** (±1 point): All measure central tendency reduction
- **Gini performs ~10% worse**: Optimizes for distribution, not prediction

---

## 📖 Where to Find Implementation Details

**For Gini implementation:**
- See **README-metrics.md** section: "Gini Impurity (StatQuest Method)"
- Contains detailed mathematical formulas
- Includes worked examples with step-by-step calculations
- Shows how to bin continuous Y values
- Explains adjacent averages threshold selection

**For all metrics:**
- See **README-metrics.md** for comprehensive guide
- Each metric has its own detailed section
- Worked examples for bikes and wine datasets

---

## 🎓 Teaching Note: Which Mode to Use?

**For Unit Tests:**
- Use **outsample mode** (`--test-set outsample`)
- Tests use 80/20 split with random seed=42
- This matches industry standard practice

**For Visualization:**
- Use **insample mode** (default, no flag)
- See the full tree structure
- Good for debugging and understanding

**For Model Evaluation:**
- Use **outsample mode** 
- Proper train/test split
- Realistic performance metrics

---

## 🤓 Discovery Questions

As you implement the metrics, you'll discover some patterns:
- Why do MSE and Variance give identical results?
- Why does Gini perform worse than Variance?
- Which metrics bin continuous Y values?
- Which metrics try multiple thresholds vs just median?

These discoveries are part of the learning process!

---

## ✅ Summary

### To Match Unit Tests:
```bash
python main.py --data data/bikes.csv -i --leaf-size 1 --max-depth 4 --metric gini --test-set outsample
```
**Expected: Gini Train RMSE ≈ 239.70** ✅

### Key Takeaways:
- Two modes exist: insample (729 samples) vs outsample (583 samples)
- Gini ≠ Variance (should differ by ~25 RMSE points in both modes)
- Use outsample mode to match tests
- See README-metrics.md for detailed implementation guidance

### Expected Results Summary:

**Gini Implementation Targets:**
- Insample: ~239.83 (trains on 729 samples)
- Outsample: ~239.70 (trains on 583 samples)

**Not these values:**
- ~217.56 or ~214.17 means you're using variance, not Gini!

Good luck! 🎉