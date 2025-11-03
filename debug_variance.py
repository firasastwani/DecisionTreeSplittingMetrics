import numpy as np
import pandas as pd
from DTLearner_skeleton import DTLearner

# Load bikes data with same split as tests
df = pd.read_csv('data/bikes.csv')
X = df.iloc[:, 1:-1].values
y = df.iloc[:, -1].values

np.random.seed(42)
n_samples = len(y)
n_train = int(n_samples * 0.8)
indices = np.random.permutation(n_samples)
train_idx = indices[:n_train]
test_idx = indices[n_train:]

X_train = X[train_idx]
y_train = y[train_idx]

print("=" * 70)
print("Testing Variance Reduction")
print("=" * 70)
print(f"Training samples: {len(X_train)}")
print(f"Training features: {X_train.shape[1]}")

# Create learner with verbose mode
learner = DTLearner(leaf_size=1, max_depth=4, metric='variance_reduction', verbose=True)

print("\nTesting variance calculation on first 10 samples:")
X_sample = X_train[:10]
y_sample = y_train[:10]

print(f"\nSample X shape: {X_sample.shape}")
print(f"Sample y shape: {y_sample.shape}")
print(f"Sample y values: {y_sample}")

# Test variance reduction calculation
for i in range(3):
    print(f"\n--- Feature {i} ---")
    reduction = learner.calculate_variance_reduction(X_sample, y_sample, i)
    print(f"Variance reduction: {reduction:.6f}")

print("\n" + "=" * 70)
print("Building tree on full training set...")
print("=" * 70)
learner.add_evidence(X_train, y_train)

print("\n" + "=" * 70)
print("Making predictions...")
print("=" * 70)
train_pred = learner.query(X_train)
train_rmse = np.sqrt(np.mean((y_train - train_pred) ** 2))

print(f"\nTrain RMSE: {train_rmse:.4f}")
print(f"Expected: ~214.17")
print(f"Difference: {abs(train_rmse - 214.17):.4f}")

# Check if tree has splits
if isinstance(learner.tree, np.ndarray):
    print("\nTree structure: Single leaf (NO SPLITS!)")
    print(f"Tree: {learner.tree}")
else:
    print("\nTree structure: Has splits ✓")

