# Decision Tree Splitting Metrics: A Mathematical Guide

**Course**: Machine Learning for Trading  
**Topic**: Feature Selection Metrics for Regression Trees  
**Author**: Prof. Hybinette

---

## Table of Contents

1. [Overview](#overview)
2. [Correlation](#correlation)
3. [Gini Impurity (StatQuest Method)](#gini-impurity-statquest-method)
4. [Variance Reduction](#variance-reduction)
5. [MAE Reduction](#mae-reduction)
6. [MSE Reduction](#mse-reduction)
7. [Information Gain (Entropy)](#information-gain-entropy)
8. [Comparison & When to Use Each](#comparison--when-to-use-each)
9. [Worked Examples](#worked-examples)

---

## Overview

When building a decision tree for regression, we must decide:
1. **Which feature** to split on at each node
2. **Where to split** (what threshold value)

Different **metrics** help us make this decision by measuring how "good" a split is. This guide covers six important metrics:

- **Correlation** - Measures linear relationship strength (simple baseline)
- **Gini Impurity** (StatQuest method) - Measures distribution purity
- **Variance Reduction** - Measures reduction in target variance (standard CART)
- **MAE Reduction** - Measures reduction in absolute error (robust to outliers)
- **MSE Reduction** - Measures reduction in mean squared error
- **Information Gain** (Entropy) - Measures reduction in uncertainty/randomness

Each metric has different properties and is best suited for different scenarios.

### 🎯 Expected Results on Bikes Dataset

When you run these commands with `bikes.csv` (leaf_size=1, max_depth=4):

```bash
# Correlation (baseline)
python main.py --data data/bikes.csv -i --metric correlation
→ Train RMSE: ~218.18

# Gini method (StatQuest - binning approach)
python main.py --data data/bikes.csv -i --metric gini
→ Train RMSE: ~239.83

# Variance reduction (standard CART)
python main.py --data data/bikes.csv -i --metric variance_reduction
→ Train RMSE: ~217.56

# MSE reduction 
python main.py --data data/bikes.csv -i --metric mse_reduction
→ Train RMSE: ??? (you tell us!)

# MAE reduction (robust to outliers)
python main.py --data data/bikes.csv -i --metric mae_reduction
→ Train RMSE: 217.78 

# Information Gain (entropy with quartile binning)
python main.py --data data/bikes.csv -i --metric information_gain
→ Train RMSE: ~241.24
```

**Key Observations**:
- **Gini ≈ Information Gain** (~239-241): Both use binning and measure distribution purity
- **MSE, Variance, and MAE** (215-220): All perform similarly on bikes dataset
```

**Notice**: Gini performs ~10% worse because it optimizes for distribution purity, not prediction error!

### Key Concept: What Makes a Good Split?

A good split should:
- Separate the data into groups with **similar** target values within each group
- Create groups that are **different** from each other
- Reduce **uncertainty** or **error** in predictions

---

## Gini Impurity (StatQuest Method)

> **🎯 IMPORTANT FOR IMPLEMENTATION**: This is the method used when you run:
> ```bash
> python main.py --data data/bikes.csv -i --metric gini
> ```
> This produces different results than `--metric variance_reduction` because it looks at the **distribution** of Y values (binned into quartiles for continuous data), NOT the variance.

### 📚 Concept

The **Gini impurity** measures how "mixed" or "impure" a group of data is based on the **distribution** of target values (Y). 

**Key idea**: After splitting, we want each group to have target values that are concentrated in specific categories or ranges (high purity = low Gini).

### 📐 Mathematical Definition

For a group of data with target values **Y**:

#### Step 1: Categorize Y Values

**For Discrete Y** (e.g., wine quality ratings: 5, 6, 7, 8):
- Use the actual categories directly
- Example: Y = [5, 5, 6, 6, 7, 8]

**For Continuous Y** (e.g., bike rentals: 23.5, 112.3, 178.9):
- Bin Y into quartiles (4 equal-frequency bins)
- Example: Y = [23.5, 45.2, 67.8, 89.1] → Bins [Q1, Q2, Q3, Q4]

#### Step 2: Calculate Proportions

For each category/bin **i**, calculate the proportion:

```
pᵢ = (number of samples in category i) / (total number of samples)
```

#### Step 3: Compute Gini Impurity

```
Gini = 1 - Σ(pᵢ²)
```

Where:
- **Gini = 0**: Perfect purity (all Y values in one category)
- **Gini = 0.5**: Maximum impurity for binary (50/50 split)
- **Higher Gini** = More mixed/impure
- **Lower Gini** = More pure/concentrated

### 🎯 Split Selection Process

For each feature **Xⱼ**:

1. **Sort** the unique values of Xⱼ
2. **Calculate adjacent averages** (midpoints between consecutive values):
   ```
   threshold = (Xⱼ[i] + Xⱼ[i+1]) / 2
   ```
   Example: If Xⱼ has values [1.0, 2.0, 5.0], try thresholds [1.5, 3.5]

3. For each threshold, **split** the data:
   - Left: samples where Xⱼ ≤ threshold
   - Right: samples where Xⱼ > threshold

4. **Calculate Gini** for left and right groups using their Y distributions:
   - For **discrete Y**: Count frequency of each unique Y value
   - For **continuous Y**: Bin Y into 4 quartiles, count frequency in each bin

5. **Compute weighted Gini**:
   ```
   Weighted_Gini = (nₗₑfₜ/nₜₒₜₐₗ) × Giniₗₑfₜ + (nᵣᵢgₕₜ/nₜₒₜₐₗ) × Giniᵣᵢgₕₜ
   ```

6. **Calculate gain** (improvement from split):
   ```
   Gain = Gini_before - Weighted_Gini_after
   ```

7. **Choose** the threshold with maximum gain

**Key Difference from Variance Reduction**: 
- Gini tries **many thresholds** (all adjacent averages)
- Variance tries **one threshold** (the median)
- Gini examines **Y distribution** (bins/categories)
- Variance examines **Y spread** (distance from mean)
   ```
   Weighted Gini = (nₗₑfₜ/nₜₒₜₐₗ) × Giniₗₑfₜ + (nᵣᵢgₕₜ/nₜₒₜₐₗ) × Giniᵣᵢgₕₜ
   ```

6. **Calculate gain** (improvement from split):
   ```
   Gain = Gini_before - Weighted_Gini_after
   ```

7. **Choose** the threshold with maximum gain

### 💡 Key Properties

- ✅ Uses Y values (examines their distribution)
- ✅ Works for discrete and continuous targets
- ✅ Focuses on "purity" of distribution
- ⚠️ May not minimize prediction error
- ⚠️ Binning continuous Y can lose information

---

## Variance Reduction

### 📚 Concept

**Variance reduction** measures how much we reduce the **spread** (variance) of target values by splitting the data.

**Key idea**: A good split creates groups where Y values are close together (low variance within groups).

### 📐 Mathematical Definition

**Variance** measures how far values are from their mean:

```
Var(Y) = (1/n) Σ(yᵢ - ȳ)²
```

Where:
- **yᵢ** = individual target values
- **ȳ** = mean of target values
- **n** = number of samples

### 🎯 Split Selection Process

For each feature **Xⱼ**:

1. **Calculate current variance**:
   ```
   Var_before = Var(Y)
   ```

2. **Try split at median** of Xⱼ:
   ```
   threshold = median(Xⱼ)
   ```

3. **Split** the data:
   - Left: samples where Xⱼ ≤ threshold → get Yₗₑfₜ
   - Right: samples where Xⱼ > threshold → get Yᵣᵢgₕₜ

4. **Calculate variance** for each group:
   ```
   Var_left = Var(Yₗₑfₜ)
   Var_right = Var(Yᵣᵢgₕₜ)
   ```

5. **Compute weighted variance** after split:
   ```
   Weighted_Var = (nₗₑfₜ/nₜₒₜₐₗ) × Var_left + (nᵣᵢgₕₜ/nₜₒₜₐₗ) × Var_right
   ```

6. **Calculate reduction** (gain):
   ```
   Reduction = Var_before - Weighted_Var
   ```

7. **Choose** the feature with maximum reduction

### 💡 Key Properties

- ✅ Directly minimizes variance of predictions
- ✅ Standard approach in ML (CART, sklearn)
- ✅ Works well for continuous targets
- ✅ Mathematically optimal for MSE
- ⚠️ Sensitive to outliers (uses squared differences)
- ⚠️ Always splits at median (doesn't try multiple thresholds)

---

## MAE Reduction

### 📚 Concept

**MAE reduction** measures how much we reduce the **Mean Absolute Error** by splitting the data.

**Key idea**: A good split creates groups where Y values are close to their group's median (minimizes absolute deviations).

### 📐 Mathematical Definition

**Mean Absolute Error** measures average distance from the median:

```
MAE(Y) = (1/n) Σ|yᵢ - median(Y)|
```

Where:
- **yᵢ** = individual target values
- **median(Y)** = median of target values
- **| |** = absolute value
- **n** = number of samples

### 🎯 Split Selection Process

For each feature **Xⱼ**:

1. **Calculate current MAE**:
   ```
   MAE_before = (1/n) Σ|yᵢ - median(Y)|
   ```

2. **Try split at median** of Xⱼ:
   ```
   threshold = median(Xⱼ)
   ```

3. **Split** the data:
   - Left: samples where Xⱼ ≤ threshold → get Yₗₑfₜ
   - Right: samples where Xⱼ > threshold → get Yᵣᵢgₕₜ

4. **Calculate MAE** for each group:
   ```
   MAE_left = (1/nₗₑfₜ) Σ|yᵢ - median(Yₗₑfₜ)|
   MAE_right = (1/nᵣᵢgₕₜ) Σ|yᵢ - median(Yᵣᵢgₕₜ)|
   ```

5. **Compute weighted MAE** after split:
   ```
   Weighted_MAE = (nₗₑfₜ/nₜₒₜₐₗ) × MAE_left + (nᵣᵢgₕₜ/nₜₒₜₐₗ) × MAE_right
   ```

6. **Calculate reduction** (gain):
   ```
   Reduction = MAE_before - Weighted_MAE
   ```

7. **Choose** the feature with maximum reduction

### 💡 Key Properties

- ✅ Robust to outliers (uses absolute value, not squares)
- ✅ More stable with noisy data
- ✅ Uses median instead of mean (less affected by extremes)
- ✅ Often gives best test performance
- ⚠️ Slightly more complex to compute
- ⚠️ Less common in standard ML libraries

---

## MSE Reduction

### What is MSE Reduction?

MSE (Mean Squared Error) Reduction measures how much splitting on a feature reduces the mean squared error of predictions.

**Key insight**: By minimizing MSE, we find splits that reduce prediction error when using the mean as our predictor.

### Mathematical Formula

```
MSE_before = mean((Y - mean(Y))²)
MSE_after = weighted average of MSE in left and right groups
MSE_reduction = MSE_before - MSE_after
```

### What is the relationship between MSE, and Variance?

```
?
```

### Algorithm Steps

1. Calculate **MSE before split**: `mean((Y - mean(Y))²)`
2. Find split point (use median of feature)
3. Split data into left and right groups
4. Calculate **MSE for each group**: Use group's mean as prediction
5. Calculate **weighted MSE**: `(n_left/n_total) × MSE_left + (n_right/n_total) × MSE_right`
6. **Reduction** = MSE_before - weighted_MSE

### Implementation Notes

- Implement using squared errors: `(Y - mean(Y))²`
- Split at median of feature values
- Calculate weighted average of MSE for left and right groups
- Return reduction (MSE_before - weighted_MSE)

### 💡 Key Properties

- ✅ Intuitive measure of prediction error
- ✅ Fast to compute
- ✅ Standard metric for regression trees
- ✅ Directly optimizes what we care about (error)
- ⚠️ Not robust to outliers (uses squared errors)
- ⚠️ Penalizes large errors more heavily

---

## Information Gain (Entropy)

### What is Information Gain?

Information Gain measures the reduction in **entropy** (uncertainty/randomness) achieved by splitting on a feature. Originally designed for classification, it can be adapted for regression by discretizing continuous target values.

**Key concept**: Entropy measures how "mixed" or "uncertain" a set of values is. Lower entropy = more predictable, higher entropy = more random.

### 🎯 THE MOST IMPORTANT PART: Regression Adaptation

**For regression**, we must convert continuous Y values into discrete categories (bins) before calculating entropy:

#### Quartile Binning (REQUIRED)

**You MUST use quartile binning** to ensure consistent, gradeable results:

```python
# Bin Y into quartiles (4 categories)
bins = np.percentile(dataY, [0, 25, 50, 75, 100])
y_binned = np.digitize(dataY, bins[1:-1])  # Creates 4 bins: 0,1,2,3
```

This creates 4 bins where:
- **Bin 0**: Bottom 25% of Y values (Q1)
- **Bin 1**: 25th-50th percentile (Q2)
- **Bin 2**: 50th-75th percentile (Q3)
- **Bin 3**: Top 25% of Y values (Q4)

**Why quartiles?**
- ✅ Each bin has equal number of samples (balanced)
- ✅ Adapts to data distribution
- ✅ Standard discretization practice
- ✅ Consistent results for grading
- ✅ Matches worked example below

### Mathematical Formula

**Entropy** (measures uncertainty):
```
Entropy(Y) = -Σ pᵢ × log₂(pᵢ)
```
where pᵢ is the proportion of samples in bin i.

**Information Gain**:
```
IG = Entropy_before - Weighted_Entropy_after
```

**Weighted Entropy After Split**:
```
Weighted_Entropy = (n_left/n_total) × Entropy_left + (n_right/n_total) × Entropy_right
```

### Step-by-Step Algorithm for Regression

1. **Bin the target values** (Y) into discrete categories
2. Calculate **entropy before split** using binned Y
3. For each feature:
   - Try splitting at median (or multiple thresholds)
   - Bin Y values in left and right groups
   - Calculate entropy for each group
   - Calculate weighted entropy
4. **Information gain** = entropy_before - weighted_entropy_after
5. Choose feature with **highest information gain**

### Implementation Notes

**Key Challenge**: Must discretize continuous Y values into bins

**Recommended Binning Strategy**:
```python
# Quartile binning (4 bins)
bins = np.percentile(dataY, [0, 25, 50, 75, 100])
y_binned = np.digitize(dataY, bins[1:-1])
```

**Entropy Calculation**:
```python
# For each bin, calculate proportion and entropy term
# Entropy = -Σ(p * log₂(p)) where p > 0
```

**Algorithm Outline**:
1. Bin Y values into quartiles
2. Calculate entropy on binned Y
3. Split at median of feature
4. Calculate entropy for left/right binned Y
5. Compute weighted entropy
6. Return gain (entropy reduction)

### 📖 Worked Example: Information Gain (Regression)

**Dataset**:
```
X = [10, 12, 14, 16, 18, 20, 22, 24]
Y = [5,  5,  6,  6,  7,  7,  8,  8]
```

Split at X = 16 (between 14 and 18)

#### Step 1: Bin Y values into quartiles

```python
Y = [5, 5, 6, 6, 7, 7, 8, 8]
Quartiles: [5, 6, 7, 8]  # 0%, 25%, 50%, 75%, 100%

Binning:
- Y=5 → bin 0 (lowest quartile)
- Y=6 → bin 1 
- Y=7 → bin 2
- Y=8 → bin 3 (highest quartile)

Y_binned = [0, 0, 1, 1, 2, 2, 3, 3]
```

#### Step 2: Calculate entropy before split

```
Y_binned = [0, 0, 1, 1, 2, 2, 3, 3]

Bin counts:
- Bin 0: 2 samples → p₀ = 2/8 = 0.25
- Bin 1: 2 samples → p₁ = 2/8 = 0.25
- Bin 2: 2 samples → p₂ = 2/8 = 0.25
- Bin 3: 2 samples → p₃ = 2/8 = 0.25

Entropy_before = -Σ(pᵢ × log₂(pᵢ))
               = -(0.25×log₂(0.25) + 0.25×log₂(0.25) + 0.25×log₂(0.25) + 0.25×log₂(0.25))
               = -(4 × 0.25 × (-2))
               = -(4 × -0.5)
               = 2.0
```

#### Step 3: Split data

```
Split at X = 16:

Left group (X ≤ 16):
X_left = [10, 12, 14, 16]
Y_left = [5, 5, 6, 6]
Y_binned_left = [0, 0, 1, 1]

Right group (X > 16):
X_right = [18, 20, 22, 24]
Y_right = [7, 7, 8, 8]
Y_binned_right = [2, 2, 3, 3]
```

#### Step 4: Calculate entropy for left group

```
Y_binned_left = [0, 0, 1, 1]

Bin counts:
- Bin 0: 2 samples → p₀ = 2/4 = 0.5
- Bin 1: 2 samples → p₁ = 2/4 = 0.5

Entropy_left = -(0.5×log₂(0.5) + 0.5×log₂(0.5))
             = -(2 × 0.5 × (-1))
             = 1.0
```

#### Step 5: Calculate entropy for right group

```
Y_binned_right = [2, 2, 3, 3]

Bin counts:
- Bin 2: 2 samples → p₂ = 2/4 = 0.5
- Bin 3: 2 samples → p₃ = 2/4 = 0.5

Entropy_right = -(0.5×log₂(0.5) + 0.5×log₂(0.5))
              = 1.0
```

#### Step 6: Calculate weighted entropy

```
Weighted_Entropy = (n_left/n_total) × Entropy_left + (n_right/n_total) × Entropy_right
                 = (4/8) × 1.0 + (4/8) × 1.0
                 = 0.5 × 1.0 + 0.5 × 1.0
                 = 1.0
```

#### Step 7: Calculate information gain

```
Information_Gain = Entropy_before - Weighted_Entropy
                 = 2.0 - 1.0
                 = 1.0
```

✅ **Result**: Information gain of 1.0 indicates this is a **perfect split** for this feature!

### 📊 Comparison with Other Metrics (Same Data)

Using the same split (X ≤ 16):

| Metric | Score | Interpretation |
|--------|-------|----------------|
| **Information Gain** | 1.0 | Perfect reduction in entropy |
| **Variance Reduction** | 0.817 | Reduces variance by 0.817 |
| **Gini Gain** | 0.183 | Reduces impurity by 0.183 |
| **MAE Reduction** | 0.75 | Reduces MAE by 0.75 |

All metrics agree this is a good split, but they measure different aspects!

### 🎓 Information Gain for Classification (For Reference)

In **classification** (not regression), Information Gain works directly without binning:

```python
# For classification (Y is already discrete: e.g., 0, 1, 2)
def calculate_entropy_classification(Y):
    unique, counts = np.unique(Y, return_counts=True)
    proportions = counts / len(Y)
    return -np.sum([p * np.log2(p) for p in proportions if p > 0])

# Example: Binary classification
Y = [0, 0, 0, 1, 1, 1, 1, 1]  # Already discrete!
Entropy = -(3/8)×log₂(3/8) - (5/8)×log₂(5/8)
        ≈ 0.954 bits
```

**Key difference**: Classification Y values are already discrete (classes), so no binning needed!

### 💡 Key Properties

- ✅ Well-founded in information theory
- ✅ Handles multi-way splits naturally (in classification)
- ✅ Measures uncertainty reduction
- ✅ Popular in classification (ID3, C4.5 algorithms)
- ⚠️ Requires binning for regression (loses some precision)
- ⚠️ Sensitive to number of bins chosen
- ⚠️ Can be biased toward features with more unique values
- ⚠️ More complex to compute than variance

### 🤔 Why Use Information Gain for Regression?

**Advantages**:
1. Provides theoretical foundation (information theory)
2. Can handle discrete-like continuous targets well
3. Connects regression trees to classification concepts
4. Educational value (understanding entropy)

**Disadvantages**:
1. Requires discretization (binning) which loses information
2. Results depend on binning strategy
3. Usually performs worse than variance reduction
4. More computationally expensive

**Recommendation**: Use variance_reduction or mae_reduction for regression in practice. Use information_gain for learning or when target is naturally discrete.

---

## Comparison & When to Use Each

### 📊 Quick Comparison Table

| Property | Correlation | Gini | Variance | MAE | MSE | Info Gain |
|----------|-------------|------|----------|-----|-----|-----------|
| **What it measures** | Linear relationship | Distribution purity | Variance of Y | Absolute error | Squared error | Entropy/uncertainty |
| **Split threshold** | Median | Adjacent averages | Median | Median | Median | Median |
| **Best for** | Baseline | Discrete Y | Standard regression | Noisy data | Prediction error | Teaching concepts |
| **Robust to outliers** | Somewhat | Somewhat | No | Yes | No | Depends on binning |
| **Speed** | Very fast | Slow | Fast | Fast | Fast | Medium |
| **Typical RMSE (bikes)** | ~218 | ~239 | ~217 | ??? | ~217 | ~241 |

### 🎯 When to Use Each Metric

#### Use **Correlation** when:
- Quick baseline implementation needed
- Understanding linear relationships
- Simplest possible splitting criterion
- **Recommended** for initial implementation

#### Use **Gini (StatQuest)** when:
- Teaching classification concepts in regression
- You have discrete target values (ratings, categories)
- You want to understand distribution-based splitting
- Exploring alternative splitting criteria
- **Not recommended** for production regression models

#### Use **Variance Reduction** when:
- Building standard regression trees
- You want to minimize MSE (squared error)
- Your data has no outliers
- You need fast computation
- Following standard ML practices (sklearn, CART)
- **Recommended** as default for regression

#### Use **MAE Reduction** when:
- Your data has outliers
- You want robust predictions
- Minimizing absolute error is more important than squared error
- You need stable performance across different datasets
- **Recommended** for real-world noisy data

#### Use **MSE Reduction** when:
- You want to explicitly minimize prediction error
- You prefer thinking in terms of squared errors
- **Good choice** for standard regression problems

#### Use **Information Gain** when:
- Learning about information theory concepts
- Target is naturally discrete (ratings, scores)
- Educational purposes
- Want theoretical foundation
- **Not recommended** for production regression (variance/MAE better)

### 🔄 Mathematical Relationships

**Variance Reduction ≡ MSE Reduction**
```
Variance = MSE when predicting the mean
Therefore: Var(Y) = (1/n) Σ(yᵢ - mean(Y))² = MSE
```

**Why Gini ≠ Variance**
```
Gini focuses on: Distribution across categories
Variance focuses on: Distance from mean

Example:
Y = [5, 5, 6, 6]

Gini approach:
  Categories: 5→50%, 6→50%
  Gini = 1 - (0.5² + 0.5²) = 0.5

Variance approach:
  Mean = 5.5
  Var = ((5-5.5)² + (5-5.5)² + (6-5.5)² + (6-5.5)²) / 4 = 0.25
```

---

## Worked Examples

### Example 1: Wine Quality Data (Discrete Y)

**Data**: 8 wine samples

| Sample | Alcohol (X) | Quality (Y) |
|--------|-------------|-------------|
| 1 | 11.5 | 5 |
| 2 | 12.0 | 5 |
| 3 | 12.5 | 6 |
| 4 | 13.0 | 6 |
| 5 | 13.5 | 7 |
| 6 | 14.0 | 7 |
| 7 | 14.5 | 8 |
| 8 | 15.0 | 8 |

**Question**: Should we split at Alcohol = 12.75?

---

#### Solution 1a: Gini Method

**Step 1: Split the data**
- Left (X ≤ 12.75): Y = [5, 5, 6]
- Right (X > 12.75): Y = [6, 7, 7, 8, 8]

**Step 2: Calculate Gini for left group**
```
Categories: 5→2 samples, 6→1 sample
Total: 3 samples

p₅ = 2/3 = 0.667
p₆ = 1/3 = 0.333

Gini_left = 1 - (0.667² + 0.333²)
          = 1 - (0.445 + 0.111)
          = 1 - 0.556
          = 0.444
```

**Step 3: Calculate Gini for right group**
```
Categories: 6→1, 7→2, 8→2
Total: 5 samples

p₆ = 1/5 = 0.2
p₇ = 2/5 = 0.4
p₈ = 2/5 = 0.4

Gini_right = 1 - (0.2² + 0.4² + 0.4²)
           = 1 - (0.04 + 0.16 + 0.16)
           = 1 - 0.36
           = 0.640
```

**Step 4: Weighted Gini**
```
Weighted_Gini = (3/8) × 0.444 + (5/8) × 0.640
              = 0.375 × 0.444 + 0.625 × 0.640
              = 0.167 + 0.400
              = 0.567
```

**Step 5: Calculate gain**
```
Before split - all Y = [5,5,6,6,7,7,8,8]:
Categories: 5→2, 6→2, 7→2, 8→2
p = 0.25 for each

Gini_before = 1 - (0.25² + 0.25² + 0.25² + 0.25²)
            = 1 - 0.25
            = 0.75

Gain = 0.75 - 0.567 = 0.183
```

✅ **This is a good split** (positive gain)

---

#### Solution 1b: Variance Reduction Method

**Step 1: Calculate variance before split**
```
Y = [5, 5, 6, 6, 7, 7, 8, 8]
Mean = (5+5+6+6+7+7+8+8)/8 = 6.5

Var_before = [(5-6.5)² + (5-6.5)² + (6-6.5)² + (6-6.5)² + 
              (7-6.5)² + (7-6.5)² + (8-6.5)² + (8-6.5)²] / 8
           = [2.25 + 2.25 + 0.25 + 0.25 + 0.25 + 0.25 + 2.25 + 2.25] / 8
           = 10.0 / 8
           = 1.25
```

**Step 2: Calculate variance for left group**
```
Y_left = [5, 5, 6]
Mean_left = (5+5+6)/3 = 5.333

Var_left = [(5-5.333)² + (5-5.333)² + (6-5.333)²] / 3
         = [0.111 + 0.111 + 0.444] / 3
         = 0.667 / 3
         = 0.222
```

**Step 3: Calculate variance for right group**
```
Y_right = [6, 7, 7, 8, 8]
Mean_right = (6+7+7+8+8)/5 = 7.2

Var_right = [(6-7.2)² + (7-7.2)² + (7-7.2)² + (8-7.2)² + (8-7.2)²] / 5
          = [1.44 + 0.04 + 0.04 + 0.64 + 0.64] / 5
          = 2.8 / 5
          = 0.56
```

**Step 4: Weighted variance**
```
Weighted_Var = (3/8) × 0.222 + (5/8) × 0.56
             = 0.375 × 0.222 + 0.625 × 0.56
             = 0.083 + 0.350
             = 0.433
```

**Step 5: Reduction (gain)**
```
Reduction = 1.25 - 0.433 = 0.817
```

✅ **This is a good split** (positive reduction)

---

#### Solution 1c: MAE Reduction Method

**Step 1: Calculate MAE before split**
```
Y = [5, 5, 6, 6, 7, 7, 8, 8]
Median = (6 + 7) / 2 = 6.5

MAE_before = [|5-6.5| + |5-6.5| + |6-6.5| + |6-6.5| + 
              |7-6.5| + |7-6.5| + |8-6.5| + |8-6.5|] / 8
           = [1.5 + 1.5 + 0.5 + 0.5 + 0.5 + 0.5 + 1.5 + 1.5] / 8
           = 8.0 / 8
           = 1.0
```

**Step 2: Calculate MAE for left group**
```
Y_left = [5, 5, 6]
Median_left = 5

MAE_left = [|5-5| + |5-5| + |6-5|] / 3
         = [0 + 0 + 1] / 3
         = 0.333
```

**Step 3: Calculate MAE for right group**
```
Y_right = [6, 7, 7, 8, 8]
Median_right = 7

MAE_right = [|6-7| + |7-7| + |7-7| + |8-7| + |8-7|] / 5
          = [1 + 0 + 0 + 1 + 1] / 5
          = 3 / 5
          = 0.6
```

**Step 4: Weighted MAE**
```
Weighted_MAE = (3/8) × 0.333 + (5/8) × 0.6
             = 0.375 × 0.333 + 0.625 × 0.6
             = 0.125 + 0.375
             = 0.5
```

**Step 5: Reduction (gain)**
```
Reduction = 1.0 - 0.5 = 0.5
```

✅ **This is a good split** (positive reduction)

---

#### Solution 1d: MSE Reduction Method

**Step 1: Calculate MSE before split**
```
Y = [5, 5, 6, 6, 7, 7, 8, 8]
Mean = (5+5+6+6+7+7+8+8)/8 = 6.5

MSE_before = [(5-6.5)² + (5-6.5)² + (6-6.5)² + (6-6.5)² + 
              (7-6.5)² + (7-6.5)² + (8-6.5)² + (8-6.5)²] / 8
           = [2.25 + 2.25 + 0.25 + 0.25 + 0.25 + 0.25 + 2.25 + 2.25] / 8
           = 10.0 / 8
           = 1.25
```

**Step 2: Calculate MSE for left group**
```
Y_left = [5, 5, 6]
Mean_left = (5+5+6)/3 = 5.333

MSE_left = [(5-5.333)² + (5-5.333)² + (6-5.333)²] / 3
         = [0.111 + 0.111 + 0.444] / 3
         = 0.667 / 3
         = 0.222
```

**Step 3: Calculate MSE for right group**
```
Y_right = [6, 7, 7, 8, 8]
Mean_right = (6+7+7+8+8)/5 = 7.2

MSE_right = [(6-7.2)² + (7-7.2)² + (7-7.2)² + (8-7.2)² + (8-7.2)²] / 5
          = [1.44 + 0.04 + 0.04 + 0.64 + 0.64] / 5
          = 2.8 / 5
          = 0.56
```

**Step 4: Weighted MSE**
```
Weighted_MSE = (3/8) × 0.222 + (5/8) × 0.56
             = 0.375 × 0.222 + 0.625 × 0.56
             = 0.083 + 0.350
             = 0.433
```

**Step 5: Reduction (gain)**
```
Reduction = 1.25 - 0.433 = 0.817
```

✅ **This is a good split** (positive reduction)

---

#### Solution 1e: Information Gain Method

**Step 1: Bin Y values into categories**
```
Y = [5, 5, 6, 6, 7, 7, 8, 8]

Using quartiles as bins:
- Bin 0 (lowest 25%): Y = 5
- Bin 1 (25-50%): Y = 6
- Bin 2 (50-75%): Y = 7
- Bin 3 (highest 25%): Y = 8

Y_binned = [0, 0, 1, 1, 2, 2, 3, 3]
```

**Step 2: Calculate entropy before split**
```
Y_binned = [0, 0, 1, 1, 2, 2, 3, 3]

Bin counts:
- Bin 0: 2 samples → p₀ = 2/8 = 0.25
- Bin 1: 2 samples → p₁ = 2/8 = 0.25
- Bin 2: 2 samples → p₂ = 2/8 = 0.25
- Bin 3: 2 samples → p₃ = 2/8 = 0.25

Entropy_before = -Σ(pᵢ × log₂(pᵢ))
               = -(0.25×log₂(0.25) + 0.25×log₂(0.25) + 0.25×log₂(0.25) + 0.25×log₂(0.25))
               = -(4 × 0.25 × (-2))
               = 2.0 bits
```

**Step 3: Split data and bin Y for each group**
```
Left (X ≤ 12.75):
Y_left = [5, 5, 6]
Y_binned_left = [0, 0, 1]

Right (X > 12.75):
Y_right = [6, 7, 7, 8, 8]
Y_binned_right = [1, 2, 2, 3, 3]
```

**Step 4: Calculate entropy for left group**
```
Y_binned_left = [0, 0, 1]

Bin counts:
- Bin 0: 2 samples → p₀ = 2/3 = 0.667
- Bin 1: 1 sample → p₁ = 1/3 = 0.333

Entropy_left = -(0.667×log₂(0.667) + 0.333×log₂(0.333))
             = -(0.667×(-0.585) + 0.333×(-1.585))
             = -(-0.390 - 0.528)
             = 0.918 bits
```

**Step 5: Calculate entropy for right group**
```
Y_binned_right = [1, 2, 2, 3, 3]

Bin counts:
- Bin 1: 1 sample → p₁ = 1/5 = 0.2
- Bin 2: 2 samples → p₂ = 2/5 = 0.4
- Bin 3: 2 samples → p₃ = 2/5 = 0.4

Entropy_right = -(0.2×log₂(0.2) + 0.4×log₂(0.4) + 0.4×log₂(0.4))
              = -(0.2×(-2.322) + 0.4×(-1.322) + 0.4×(-1.322))
              = -(-0.464 - 0.529 - 0.529)
              = 1.522 bits
```

**Step 6: Calculate weighted entropy**
```
Weighted_Entropy = (3/8) × 0.918 + (5/8) × 1.522
                 = 0.375 × 0.918 + 0.625 × 1.522
                 = 0.344 + 0.951
                 = 1.295 bits
```

**Step 7: Calculate information gain**
```
Information_Gain = Entropy_before - Weighted_Entropy
                 = 2.0 - 1.295
                 = 0.705 bits
```

✅ **This is a good split** (positive information gain)

**Note**: Higher information gain = more uncertainty reduction

---

#### Solution 1f: Correlation Method

The correlation method measures the strength of the linear relationship between feature X and target Y.

**Step 1: Calculate correlation coefficient**

```
X = [11.5, 12.0, 12.5, 13.0, 13.5, 14.0, 14.5, 15.0]
Y = [5, 5, 6, 6, 7, 7, 8, 8]

Using Pearson correlation formula:
r = Σ((X - X̄)(Y - Ȳ)) / √(Σ(X - X̄)² × Σ(Y - Ȳ)²)
```

**Step 2: Calculate means**
```
X̄ = (11.5 + 12.0 + 12.5 + 13.0 + 13.5 + 14.0 + 14.5 + 15.0) / 8
  = 104.0 / 8
  = 13.0

Ȳ = (5 + 5 + 6 + 6 + 7 + 7 + 8 + 8) / 8
  = 52 / 8
  = 6.5
```

**Step 3: Calculate deviations and products**
```
Sample 1: (11.5 - 13.0) × (5 - 6.5) = (-1.5) × (-1.5) = 2.25
Sample 2: (12.0 - 13.0) × (5 - 6.5) = (-1.0) × (-1.5) = 1.50
Sample 3: (12.5 - 13.0) × (6 - 6.5) = (-0.5) × (-0.5) = 0.25
Sample 4: (13.0 - 13.0) × (6 - 6.5) = (0.0) × (-0.5) = 0.00
Sample 5: (13.5 - 13.0) × (7 - 6.5) = (0.5) × (0.5) = 0.25
Sample 6: (14.0 - 13.0) × (7 - 6.5) = (1.0) × (0.5) = 0.50
Sample 7: (14.5 - 13.0) × (8 - 6.5) = (1.5) × (1.5) = 2.25
Sample 8: (15.0 - 13.0) × (8 - 6.5) = (2.0) × (1.5) = 3.00

Sum of products = 2.25 + 1.50 + 0.25 + 0.00 + 0.25 + 0.50 + 2.25 + 3.00
                = 10.0
```

**Step 4: Calculate standard deviations**
```
Σ(X - X̄)² = 1.5² + 1.0² + 0.5² + 0² + 0.5² + 1.0² + 1.5² + 2.0²
          = 2.25 + 1.0 + 0.25 + 0 + 0.25 + 1.0 + 2.25 + 4.0
          = 11.0

Σ(Y - Ȳ)² = 1.5² + 1.5² + 0.5² + 0.5² + 0.5² + 0.5² + 1.5² + 1.5²
          = 2.25 + 2.25 + 0.25 + 0.25 + 0.25 + 0.25 + 2.25 + 2.25
          = 10.0
```

**Step 5: Calculate correlation**
```
r = 10.0 / √(11.0 × 10.0)
  = 10.0 / √110.0
  = 10.0 / 10.488
  = 0.954
```

**Step 6: Feature score**
```
Score = |r| = |0.954| = 0.954
```

✅ **This is an excellent feature** (very high correlation = strong linear relationship!)

**Note**: The correlation coefficient ranges from -1 to +1:
- Values near ±1 indicate strong linear relationship
- Values near 0 indicate weak/no linear relationship
- We use absolute value because both positive and negative correlations are useful for prediction

---

### Example 2: Bike Rentals (Continuous Y)

**Data**: 6 samples

| Sample | Temperature (X) | Rentals (Y) |
|--------|----------------|-------------|
| 1 | 0.2 | 23.5 |
| 2 | 0.4 | 67.8 |
| 3 | 0.6 | 112.3 |
| 4 | 0.7 | 145.6 |
| 5 | 0.8 | 178.9 |
| 6 | 0.9 | 201.4 |

**Question**: Should we split at Temperature = 0.5?

---

#### Solution 2a: Gini Method

**Step 1: Split the data**
- Left (X ≤ 0.5): Y = [23.5, 67.8]
- Right (X > 0.5): Y = [112.3, 145.6, 178.9, 201.4]

**Step 2: Bin Y into quartiles for left group**
```
Y_left = [23.5, 67.8]
Since only 2 values, we get 2 bins (50% each)

p₁ = 0.5
p₂ = 0.5

Gini_left = 1 - (0.5² + 0.5²)
          = 1 - 0.5
          = 0.5
```

**Step 3: Bin Y into quartiles for right group**
```
Y_right = [112.3, 145.6, 178.9, 201.4]
Quartiles: [112.3, 129.0, 162.3, 201.4]

All 4 values fall in different quartiles (25% each)

Gini_right = 1 - (0.25² + 0.25² + 0.25² + 0.25²)
           = 1 - 0.25
           = 0.75
```

**Step 4: Weighted Gini**
```
Weighted_Gini = (2/6) × 0.5 + (4/6) × 0.75
              = 0.333 × 0.5 + 0.667 × 0.75
              = 0.167 + 0.500
              = 0.667
```

---

#### Solution 2b: Variance Reduction Method

**Step 1: Calculate variance before split**
```
Y = [23.5, 67.8, 112.3, 145.6, 178.9, 201.4]
Mean = (23.5 + 67.8 + 112.3 + 145.6 + 178.9 + 201.4) / 6 = 121.6

Var_before = [(23.5-121.6)² + (67.8-121.6)² + (112.3-121.6)² + 
              (145.6-121.6)² + (178.9-121.6)² + (201.4-121.6)²] / 6
           = [9625 + 2894 + 86 + 576 + 3283 + 6368] / 6
           = 22832 / 6
           = 3805.3
```

**Step 2: Variance for left group**
```
Y_left = [23.5, 67.8]
Mean_left = 45.65

Var_left = [(23.5-45.65)² + (67.8-45.65)²] / 2
         = [490.2 + 490.2] / 2
         = 490.2
```

**Step 3: Variance for right group**
```
Y_right = [112.3, 145.6, 178.9, 201.4]
Mean_right = 159.55

Var_right = [(112.3-159.55)² + (145.6-159.55)² + 
             (178.9-159.55)² + (201.4-159.55)²] / 4
          = [2232 + 194 + 374 + 1751] / 4
          = 4551 / 4
          = 1137.8
```

**Step 4: Weighted variance**
```
Weighted_Var = (2/6) × 490.2 + (4/6) × 1137.8
             = 0.333 × 490.2 + 0.667 × 1137.8
             = 163.3 + 759.1
             = 922.4
```

**Step 5: Reduction**
```
Reduction = 3805.3 - 922.4 = 2882.9
```

✅ **Excellent split** (large reduction)

---

#### Solution 2c: MAE Reduction Method

**Step 1: MAE before split**
```
Y = [23.5, 67.8, 112.3, 145.6, 178.9, 201.4]
Median = (112.3 + 145.6) / 2 = 129.0

MAE_before = [|23.5-129| + |67.8-129| + |112.3-129| + 
              |145.6-129| + |178.9-129| + |201.4-129|] / 6
           = [105.5 + 61.2 + 16.7 + 16.6 + 49.9 + 72.4] / 6
           = 322.3 / 6
           = 53.7
```

**Step 2: MAE for left group**
```
Y_left = [23.5, 67.8]
Median_left = (23.5 + 67.8) / 2 = 45.65

MAE_left = [|23.5-45.65| + |67.8-45.65|] / 2
         = [22.15 + 22.15] / 2
         = 22.15
```

**Step 3: MAE for right group**
```
Y_right = [112.3, 145.6, 178.9, 201.4]
Median_right = (145.6 + 178.9) / 2 = 162.25

MAE_right = [|112.3-162.25| + |145.6-162.25| + 
             |178.9-162.25| + |201.4-162.25|] / 4
          = [49.95 + 16.65 + 16.65 + 39.15] / 4
          = 122.4 / 4
          = 30.6
```

**Step 4: Weighted MAE**
```
Weighted_MAE = (2/6) × 22.15 + (4/6) × 30.6
             = 0.333 × 22.15 + 0.667 × 30.6
             = 7.4 + 20.4
             = 27.8
```

**Step 5: Reduction**
```
Reduction = 53.7 - 27.8 = 25.9
```

✅ **Good split** (positive reduction)

---

## Summary & Key Takeaways

### 🎯 Main Differences

1. **What they measure**:
   - Gini: Distribution of Y across categories/bins
   - Variance: Spread of Y from mean
   - MAE: Absolute distance of Y from median

2. **How they split**:
   - Gini: Try many thresholds (adjacent averages)
   - Variance: Split at median of X
   - MAE: Split at median of X

3. **Performance**:
   - Gini: Moderate (focuses on purity, not error)
   - Variance: Good (standard approach)
   - MAE: Often best (robust to outliers)

### 📚 For Students

**Practice Exercise**: Given a dataset, calculate all three metrics for the same split and compare:
1. Which gives the best gain/reduction?
2. Why might they differ?
3. Which would you choose for your application?

**Discussion Questions**:
1. Why does Gini bin continuous Y values?
2. When would MAE be better than Variance?
3. How do outliers affect each metric?
4. What's the connection between variance and MSE?

### 🔗 Additional Resources

- **StatQuest Videos** (Josh Starmer): Excellent visual explanations
- **CART Algorithm** (Breiman et al.): Original decision tree paper
- **sklearn Documentation**: Implementation details
- **ISL Chapter 8**: Decision trees for regression

---

**Remember**: The "best" metric depends on your data, your goal, and what you're trying to optimize. Try multiple metrics and use cross-validation to choose!

---

*End of Guide*