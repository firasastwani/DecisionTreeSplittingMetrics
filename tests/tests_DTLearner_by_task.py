"""
Unit Tests for DTLearner - Organized by Homework Tasks
Tests follow the task structure from HW08 assignment

Run with:
    pytest tests_DTLearner_by_task.py -v
    or
    python -m pytest tests_DTLearner_by_task.py -v

For specific task:
    pytest tests_DTLearner_by_task.py::TestTask0 -v
    pytest tests_DTLearner_by_task.py::TestTask1 -v
"""

import pytest
import numpy as np
import pandas as pd
# from DTLearner_solution import DTLearner
from DTLearner_skeleton import DTLearner


# ============================================================================
# TASK 0: Basic Tree Structure and Leaf Creation (5 pts)
# Lab Week 1 - Already Done
# ============================================================================

class TestTask0_BasicTreeStructure:
    """Task 0: Complete basic tree structure and leaf creation"""

    def test_task0a_initialization(self):
        """Task 0a: Test that DTLearner initializes correctly"""
        learner = DTLearner(leaf_size=5, max_depth=10, verbose=False)
        assert learner.leaf_size == 5
        assert learner.max_depth == 10
        assert learner.verbose == 0
        assert learner.tree is None
        print("✓ Task 0a: Initialization works correctly")

    def test_task0b_tree_is_built(self):
        """Task 0b: Test that tree is created after training"""
        X = np.random.rand(20, 3)
        Y = np.random.rand(20)

        learner = DTLearner()
        assert learner.tree is None

        learner.add_evidence(X, Y)
        assert learner.tree is not None
        print("✓ Task 0b: Tree builds successfully")

    def test_task0c_single_sample_leaf(self):
        """Task 0c: Test with a single data point creates leaf"""
        X = np.array([[1.0]])
        Y = np.array([5.0])

        learner = DTLearner(leaf_size=1, max_depth=5)
        learner.add_evidence(X, Y)

        predictions = learner.query(X)
        assert predictions[0] == 5.0
        print("✓ Task 0c: Single sample creates correct leaf")

    def test_task0d_all_same_target(self):
        """Task 0d: Test when all target values are identical"""
        X = np.array([[1.0], [2.0], [3.0], [4.0]])
        Y = np.array([5.0, 5.0, 5.0, 5.0])

        learner = DTLearner(leaf_size=1, max_depth=5)
        learner.add_evidence(X, Y)

        predictions = learner.query(X)
        assert np.all(predictions == 5.0)
        print("✓ Task 0d: Handles identical targets correctly")

    def test_task0e_leaf_size_constraint(self):
        """Task 0e: Test that leaf_size is respected"""
        X = np.array([[1], [2], [3], [4], [5]])
        Y = np.array([1, 2, 3, 4, 5])

        learner = DTLearner(leaf_size=10, max_depth=None)
        learner.add_evidence(X, Y)

        # With leaf_size=10 and only 5 samples, should create a leaf immediately
        assert not isinstance(learner.tree, list)
        print("✓ Task 0e: Leaf size constraint works")

    def test_task0f_max_depth_constraint(self):
        """Task 0f: Test that max_depth is respected"""
        X = np.random.rand(100, 3)
        Y = np.random.rand(100)

        learner = DTLearner(leaf_size=1, max_depth=0)
        learner.add_evidence(X, Y)

        # With max_depth=0, should create leaf immediately
        assert not isinstance(learner.tree, list)
        print("✓ Task 0f: Max depth constraint works")


# ============================================================================
# TASK 1: Implement Correlation-Based Splitting (5 pts)
# Lab Week 1 - Already Done
# ============================================================================

class TestTask1_CorrelationSplitting:
    """Task 1: Implement correlation-based splitting"""

    def test_task1a_correlation_selects_best_feature(self):
        """Task 1a: Test that correlation metric selects the best correlated feature"""
        # X1 has perfect correlation with Y, X2 has no correlation
        X = np.array([[1, 10], [2, 20], [3, 15], [4, 25], [5, 30]])
        Y = np.array([1, 2, 3, 4, 5])  # Perfect correlation with X[:, 0]

        learner = DTLearner(leaf_size=1, max_depth=1, metric='correlation')
        learner.add_evidence(X, Y)

        # Tree should split on feature 0 (first feature)
        assert isinstance(learner.tree, list)
        assert learner.tree[0] == 0  # First element is feature index
        print("✓ Task 1a: Correlation selects best feature")

    def test_task1b_correlation_perfect_split(self):
        """Task 1b: Test data that splits perfectly"""
        X = np.array([[1.0], [1.0], [5.0], [5.0]])
        Y = np.array([10.0, 10.0, 20.0, 20.0])

        learner = DTLearner(leaf_size=1, max_depth=2, metric='correlation')
        learner.add_evidence(X, Y)

        predictions = learner.query(X)
        assert np.allclose(predictions, Y)
        print("✓ Task 1b: Correlation handles perfect split")

    def test_task1c_correlation_linear_relationship(self):
        """Task 1c: Test on data with strong linear relationship"""
        np.random.seed(42)
        X = np.random.rand(100, 1) * 10
        Y = 2 * X[:, 0] + 5 + np.random.randn(100) * 0.1  # Y = 2X + 5 + noise

        learner = DTLearner(leaf_size=5, max_depth=10, metric='correlation')
        learner.add_evidence(X, Y)

        predictions = learner.query(X)

        # Should have low error on training data
        mae = np.mean(np.abs(predictions - Y))
        assert mae < 2.0  # Should approximate well
        print(f"✓ Task 1c: Correlation approximates linear relationship (MAE: {mae:.4f})")


# ============================================================================
# TASK 2: Implement Query/Prediction Method (5 pts)
# Lab Week 1 - Already Done
# ============================================================================

class TestTask2_QueryPrediction:
    """Task 2: Implement query/prediction method"""

    def test_task2a_query_returns_predictions(self):
        """Task 2a: Test that query returns predictions"""
        X = np.random.rand(30, 3)
        Y = np.random.rand(30)

        learner = DTLearner(leaf_size=5, max_depth=3, metric='correlation')
        learner.add_evidence(X, Y)

        predictions = learner.query(X)

        assert predictions is not None
        assert len(predictions) == len(Y)
        assert not np.any(np.isnan(predictions))
        print("✓ Task 2a: Query returns valid predictions")

    def test_task2b_query_pandas_compatibility(self):
        """Task 2b: Test that query works with pandas DataFrames"""
        X = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [2, 4, 6, 8, 10]
        })
        Y = pd.Series([1, 2, 3, 4, 5])

        learner = DTLearner(metric='correlation')
        learner.add_evidence(X, Y)

        predictions = learner.query(X)
        assert predictions is not None
        assert len(predictions) == 5
        print("✓ Task 2b: Query works with pandas DataFrames")

    def test_task2c_query_numpy_compatibility(self):
        """Task 2c: Test that query works with numpy arrays"""
        X = np.array([[1, 2], [2, 4], [3, 6], [4, 8], [5, 10]])
        Y = np.array([1, 2, 3, 4, 5])

        learner = DTLearner(metric='correlation')
        learner.add_evidence(X, Y)

        predictions = learner.query(X)
        assert predictions is not None
        assert len(predictions) == 5
        print("✓ Task 2c: Query works with numpy arrays")

    def test_task2d_query_large_dataset(self):
        """Task 2d: Test query with larger dataset"""
        np.random.seed(42)
        X = np.random.rand(1000, 5) * 100
        Y = 2 * X[:, 0] + 3 * X[:, 1] - X[:, 2] + np.random.randn(1000) * 5

        learner = DTLearner(leaf_size=10, max_depth=5, metric='correlation')
        learner.add_evidence(X, Y)

        X_test = np.random.rand(100, 5) * 100
        predictions = learner.query(X_test)

        assert len(predictions) == 100
        assert not np.any(np.isnan(predictions))
        print("✓ Task 2d: Query handles large datasets")


# ============================================================================
# TASK 3: Implement Gini Impurity Splitting (10 pts)
# Lab Week 2 - THIS WEEK
# ============================================================================

class TestTask3_GiniImplementation:
    """Task 3: Implement Gini impurity splitting (StatQuest method)"""

    def bikes_data_no_shuffling(self):
        """Load bikes dataset with 60/40 un-shuffled split (matching main.py)"""
        """Load bikes dataset for Gini tests"""
        df = pd.read_csv('data/bikes.csv')
        X = df.iloc[:, 1:-1].values  # Skip first column (date)
        y = df.iloc[:, -1].values
        split_idx = int(0.6 * len(X))
        return X[:split_idx], y[:split_idx], X[split_idx:], y[split_idx:]

    @pytest.fixture(scope="class")
    # with SHUFFLING --- as we did in main.py
    def bikes_data(self):
        """Load bikes dataset with 80/20 shuffled split (matching main.py)"""
        df = pd.read_csv('data/bikes.csv')
        X = df.iloc[:, 1:-1].values  # Skip first column (date)
        y = df.iloc[:, -1].values

        # Use the same split as main.py: 80/20 with random_state=42
        np.random.seed(42)
        n_samples = len(y)
        n_train = int(n_samples * 0.8)
        indices = np.random.permutation(n_samples)
        train_idx = indices[:n_train]
        test_idx = indices[n_train:]

        return X[train_idx], y[train_idx], X[test_idx], y[test_idx]

    def test_task3a_gini_runs_without_error(self):
        """Task 3a: Test that Gini metric runs without errors"""
        X = np.random.rand(30, 3) * 10
        Y = 2 * X[:, 0] + 3 * X[:, 1] - X[:, 2] + np.random.randn(30)

        learner = DTLearner(leaf_size=5, max_depth=3, metric='gini')
        learner.add_evidence(X, Y)
        predictions = learner.query(X)

        assert predictions is not None
        assert len(predictions) == len(Y)
        assert not np.any(np.isnan(predictions))
        print("✓ Task 3a: Gini runs without error")

    def test_task3b_gini_produces_correct_rmse(self, bikes_data):
        """Task 3b: Test Gini produces expected RMSE on bikes dataset"""
        X_train, y_train, X_test, y_test = bikes_data

        learner = DTLearner(leaf_size=1, max_depth=4, metric='gini')
        learner.add_evidence(X_train, y_train)

        train_pred = learner.query(X_train)
        train_rmse = np.sqrt(np.mean((y_train - train_pred) ** 2))

        expected = 239.70
        tolerance = 1.0

        print(f"✓ Task 3b: Gini RMSE = {train_rmse:.4f} (expected {expected:.2f})")

        assert abs(train_rmse - expected) < tolerance, \
            f"Gini Train RMSE: expected {expected:.2f}, got {train_rmse:.2f}"

        print(f"✓ Task 3b: Gini RMSE = {train_rmse:.4f} (expected {expected:.2f})")

    def test_task3c_gini_different_from_variance(self, bikes_data):
        """Task 3c: CRITICAL - Gini must be different from Variance (uses binning)"""
        X_train, y_train, X_test, y_test = bikes_data

        gini_learner = DTLearner(leaf_size=1, max_depth=4, metric='gini')
        var_learner = DTLearner(leaf_size=1, max_depth=4, metric='variance_reduction')

        gini_learner.add_evidence(X_train, y_train)
        var_learner.add_evidence(X_train, y_train)

        gini_pred = gini_learner.query(X_train)
        var_pred = var_learner.query(X_train)

        gini_rmse = np.sqrt(np.mean((y_train - gini_pred) ** 2))
        var_rmse = np.sqrt(np.mean((y_train - var_pred) ** 2))

        difference = abs(gini_rmse - var_rmse)
        assert difference > 15, \
            f"Gini ({gini_rmse:.2f}) too close to Variance ({var_rmse:.2f}). " \
            f"Check that you're binning Y values in Gini calculation. " \
            f"Difference should be >15, got {difference:.2f}"

        print(f"✓ Task 3c: Gini ≠ Variance: {gini_rmse:.2f} vs {var_rmse:.2f} (diff: {difference:.2f})")


# ============================================================================
# TASK 4: Generate Comparison Results on Bikes Dataset (5 pts)
# Lab Week 2 - THIS WEEK
# ============================================================================

class TestTask4_ComparisonResults:
    """Task 4: Generate comparison results on bikes dataset"""

    @pytest.fixture(scope="class")
    def bikes_data(self):
        """Load bikes dataset with 80/20 shuffled split (matching main.py)"""
        df = pd.read_csv('data/bikes.csv')
        X = df.iloc[:, 1:-1].values  # Skip first column (date)
        y = df.iloc[:, -1].values

        # Use same split as main.py: 80/20 with random_state=42
        np.random.seed(42)
        n_samples = len(y)
        n_train = int(n_samples * 0.8)
        indices = np.random.permutation(n_samples)
        train_idx = indices[:n_train]
        test_idx = indices[n_train:]

        return X[train_idx], y[train_idx], X[test_idx], y[test_idx]

    def test_task4a_compare_gini_vs_correlation(self, bikes_data):
        """Task 4a: Compare Gini vs Correlation"""
        X_train, y_train, X_test, y_test = bikes_data

        results = {}
        for metric in ['gini', 'correlation']:
            learner = DTLearner(leaf_size=1, max_depth=4, metric=metric)
            learner.add_evidence(X_train, y_train)
            train_pred = learner.query(X_train)
            train_rmse = np.sqrt(np.mean((y_train - train_pred) ** 2))
            results[metric] = train_rmse

        print(f"\n✓ Task 4a Comparison:")
        print(f"  Correlation RMSE: {results['correlation']:.2f}")
        print(f"  Gini RMSE:        {results['gini']:.2f}")
        print(f"  Difference:       {abs(results['gini'] - results['correlation']):.2f}")

    def test_task4b_compare_gini_vs_variance(self, bikes_data):
        """Task 4b: Compare Gini vs Variance Reduction"""
        X_train, y_train, X_test, y_test = bikes_data

        results = {}
        for metric in ['gini', 'variance_reduction']:
            learner = DTLearner(leaf_size=1, max_depth=4, metric=metric)
            learner.add_evidence(X_train, y_train)
            train_pred = learner.query(X_train)
            train_rmse = np.sqrt(np.mean((y_train - train_pred) ** 2))
            results[metric] = train_rmse

        print(f"\n✓ Task 4b Comparison:")
        print(f"  Variance RMSE: {results['variance_reduction']:.2f}")
        print(f"  Gini RMSE:     {results['gini']:.2f}")
        print(f"  Difference:    {abs(results['gini'] - results['variance_reduction']):.2f}")

        # This is the KEY comparison for lab checkoff
        assert abs(results['gini'] - results['variance_reduction']) > 15, \
            "Gini and Variance should differ by >15 RMSE"


# ============================================================================
# TASK 5a: Correlation Metric (Homework)
# ============================================================================

class TestTask5a_Correlation:
    """Task 5a: Correlation metric implementation"""

    @pytest.fixture(scope="class")
    def bikes_data(self):
        """Load bikes dataset with 80/20 shuffled split (matching main.py)"""
        df = pd.read_csv('data/bikes.csv')
        X = df.iloc[:, 1:-1].values  # Skip first column (date)
        y = df.iloc[:, -1].values

        # Use same split as main.py: 80/20 with random_state=42
        np.random.seed(42)
        n_samples = len(y)
        n_train = int(n_samples * 0.8)
        indices = np.random.permutation(n_samples)
        train_idx = indices[:n_train]
        test_idx = indices[n_train:]

        return X[train_idx], y[train_idx], X[test_idx], y[test_idx]

    def test_task5a_correlation_bikes_rmse(self, bikes_data):
        """Task 5a: Correlation on bikes dataset produces correct RMSE"""
        X_train, y_train, X_test, y_test = bikes_data

        learner = DTLearner(leaf_size=1, max_depth=4, metric='correlation')
        learner.add_evidence(X_train, y_train)

        train_pred = learner.query(X_train)
        test_pred = learner.query(X_test)

        train_rmse = np.sqrt(np.mean((y_train - train_pred) ** 2))
        test_rmse = np.sqrt(np.mean((y_test - test_pred) ** 2))

        expected_train = 214.59
        tolerance = 1.0

        assert abs(train_rmse - expected_train) < tolerance, \
            f"Correlation Train RMSE: expected {expected_train:.2f}, got {train_rmse:.2f}"

        print(f"✓ Task 5a: Correlation - Train: {train_rmse:.4f}, Test: {test_rmse:.4f}")


# ============================================================================
# TASK 5b: Gini Metric (Homework)
# ============================================================================

class TestTask5b_Gini:
    """Task 5b: Gini metric implementation"""

    @pytest.fixture(scope="class")
    def bikes_data(self):
        """Load bikes dataset with 80/20 shuffled split (matching main.py)"""
        df = pd.read_csv('data/bikes.csv')
        X = df.iloc[:, 1:-1].values  # Skip first column (date)
        y = df.iloc[:, -1].values

        # Use same split as main.py: 80/20 with random_state=42
        np.random.seed(42)
        n_samples = len(y)
        n_train = int(n_samples * 0.8)
        indices = np.random.permutation(n_samples)
        train_idx = indices[:n_train]
        test_idx = indices[n_train:]

        return X[train_idx], y[train_idx], X[test_idx], y[test_idx]

    def test_task5b_gini_bikes_rmse(self, bikes_data):
        """Task 5b: Gini on bikes dataset produces correct RMSE"""
        X_train, y_train, X_test, y_test = bikes_data

        learner = DTLearner(leaf_size=1, max_depth=4, metric='gini')
        learner.add_evidence(X_train, y_train)

        train_pred = learner.query(X_train)
        test_pred = learner.query(X_test)

        train_rmse = np.sqrt(np.mean((y_train - train_pred) ** 2))
        test_rmse = np.sqrt(np.mean((y_test - test_pred) ** 2))

        expected_train = 239.70
        tolerance = 1.0

        assert abs(train_rmse - expected_train) < tolerance, \
            f"Gini Train RMSE: expected {expected_train:.2f}, got {train_rmse:.2f}"

        print(f"✓ Task 5b: Gini - Train: {train_rmse:.4f}, Test: {test_rmse:.4f}")


# ============================================================================
# TASK 5c: Variance Reduction Metric (Homework)
# ============================================================================

class TestTask5c_VarianceReduction:
    """Task 5c: Variance reduction metric implementation"""

    @pytest.fixture(scope="class")
    def bikes_data(self):
        """Load bikes dataset with 80/20 shuffled split (matching main.py)"""
        df = pd.read_csv('data/bikes.csv')
        X = df.iloc[:, 1:-1].values  # Skip first column (date)
        y = df.iloc[:, -1].values

        # Use same split as main.py: 80/20 with random_state=42
        np.random.seed(42)
        n_samples = len(y)
        n_train = int(n_samples * 0.8)
        indices = np.random.permutation(n_samples)
        train_idx = indices[:n_train]
        test_idx = indices[n_train:]

        return X[train_idx], y[train_idx], X[test_idx], y[test_idx]

    def test_task5c_variance_bikes_rmse(self, bikes_data):
        """Task 5c: Variance reduction on bikes dataset produces correct RMSE"""
        X_train, y_train, X_test, y_test = bikes_data

        learner = DTLearner(leaf_size=1, max_depth=4, metric='variance_reduction')
        learner.add_evidence(X_train, y_train)

        train_pred = learner.query(X_train)
        test_pred = learner.query(X_test)

        train_rmse = np.sqrt(np.mean((y_train - train_pred) ** 2))
        test_rmse = np.sqrt(np.mean((y_test - test_pred) ** 2))

        expected_train = 214.17
        tolerance = 1.0

        assert abs(train_rmse - expected_train) < tolerance, \
            f"Variance Train RMSE: expected {expected_train:.2f}, got {train_rmse:.2f}"

        print(f"✓ Task 5c: Variance - Train: {train_rmse:.4f}, Test: {test_rmse:.4f}")


# ============================================================================
# TASK 5d: MAE Reduction Metric (Homework)
# ============================================================================

class TestTask5d_MAEReduction:
    """Task 5d: MAE reduction metric implementation"""

    @pytest.fixture(scope="class")
    def bikes_data(self):
        """Load bikes dataset with 80/20 shuffled split (matching main.py)"""
        df = pd.read_csv('data/bikes.csv')
        X = df.iloc[:, 1:-1].values  # Skip first column (date)
        y = df.iloc[:, -1].values

        # Use same split as main.py: 80/20 with random_state=42
        np.random.seed(42)
        n_samples = len(y)
        n_train = int(n_samples * 0.8)
        indices = np.random.permutation(n_samples)
        train_idx = indices[:n_train]
        test_idx = indices[n_train:]

        return X[train_idx], y[train_idx], X[test_idx], y[test_idx]

    def test_task5d_mae_bikes_rmse(self, bikes_data):
        """Task 5d: MAE reduction on bikes dataset produces reasonable RMSE"""
        X_train, y_train, X_test, y_test = bikes_data

        learner = DTLearner(leaf_size=1, max_depth=4, metric='mae_reduction')
        learner.add_evidence(X_train, y_train)

        train_pred = learner.query(X_train)
        test_pred = learner.query(X_test)

        train_rmse = np.sqrt(np.mean((y_train - train_pred) ** 2))
        test_rmse = np.sqrt(np.mean((y_test - test_pred) ** 2))

        # MAE expected value kept flexible
        assert 210 < train_rmse < 218, \
            f"MAE Train RMSE {train_rmse:.2f} outside expected range (210-218)"

        print(f"✓ Task 5d: MAE - Train: {train_rmse:.4f}, Test: {test_rmse:.4f}")


# ============================================================================
# TASK 5e: MSE Reduction Metric (Homework)
# ============================================================================

class TestTask5e_MSEReduction:
    """Task 5e: MSE reduction metric implementation"""

    @pytest.fixture(scope="class")
    def bikes_data(self):
        """Load bikes dataset with 80/20 shuffled split (matching main.py)"""
        df = pd.read_csv('data/bikes.csv')
        X = df.iloc[:, 1:-1].values  # Skip the first column (date)
        y = df.iloc[:, -1].values

        # Use same split as main.py: 80/20 with random_state=42
        np.random.seed(42)
        n_samples = len(y)
        n_train = int(n_samples * 0.8)
        indices = np.random.permutation(n_samples)
        train_idx = indices[:n_train]
        test_idx = indices[n_train:]

        return X[train_idx], y[train_idx], X[test_idx], y[test_idx]

    def test_task5e_mse_bikes_rmse(self, bikes_data):
        """Task 5e: MSE reduction on bikes dataset produces correct RMSE"""
        X_train, y_train, X_test, y_test = bikes_data

        learner = DTLearner(leaf_size=1, max_depth=4, metric='mse_reduction')
        learner.add_evidence(X_train, y_train)

        train_pred = learner.query(X_train)
        test_pred = learner.query(X_test)

        train_rmse = np.sqrt(np.mean((y_train - train_pred) ** 2))
        test_rmse = np.sqrt(np.mean((y_test - test_pred) ** 2))

        expected_train = 214.17  # Should equal variance
        tolerance = 1.0

        assert abs(train_rmse - expected_train) < tolerance, \
            f"MSE Train RMSE: expected {expected_train:.2f}, got {train_rmse:.2f}"

        print(f"✓ Task 5e: MSE - Train: {train_rmse:.4f}, Test: {test_rmse:.4f}")

    def test_task5e_mse_equals_variance(self, bikes_data):
        """Task 5e: CRITICAL - MSE must equal Variance mathematically"""
        X_train, y_train, X_test, y_test = bikes_data

        mse_learner = DTLearner(leaf_size=1, max_depth=4, metric='mse_reduction')
        var_learner = DTLearner(leaf_size=1, max_depth=4, metric='variance_reduction')

        mse_learner.add_evidence(X_train, y_train)
        var_learner.add_evidence(X_train, y_train)

        mse_pred = mse_learner.query(X_train)
        var_pred = var_learner.query(X_train)

        mse_rmse = np.sqrt(np.mean((y_train - mse_pred) ** 2))
        var_rmse = np.sqrt(np.mean((y_train - var_pred) ** 2))

        difference = abs(mse_rmse - var_rmse)
        assert difference < 0.01, \
            f"MSE ({mse_rmse:.4f}) must equal Variance ({var_rmse:.4f}). " \
            f"Difference: {difference:.4f}"

        print(f"✓ Task 5e: MSE = Variance: {mse_rmse:.4f} = {var_rmse:.4f}")


# ============================================================================
# TASK 5f: Information Gain Metric (Homework)
# ============================================================================

class TestTask5f_InformationGain:
    """Task 5f: Information gain metric implementation"""

    @pytest.fixture(scope="class")
    def bikes_data(self):
        """Load bikes dataset with 80/20 shuffled split (matching main.py)"""
        df = pd.read_csv('data/bikes.csv')
        X = df.iloc[:, 1:-1].values  # Skip first column (date)
        y = df.iloc[:, -1].values

        # Use same split as main.py: 80/20 with random_state=42
        np.random.seed(42)
        n_samples = len(y)
        n_train = int(n_samples * 0.8)
        indices = np.random.permutation(n_samples)
        train_idx = indices[:n_train]
        test_idx = indices[n_train:]

        return X[train_idx], y[train_idx], X[test_idx], y[test_idx]

    def test_task5f_infogain_bikes_rmse(self, bikes_data):
        """Task 5f: Information gain on bikes dataset produces correct RMSE"""
        X_train, y_train, X_test, y_test = bikes_data

        learner = DTLearner(leaf_size=1, max_depth=4, metric='information_gain')
        learner.add_evidence(X_train, y_train)

        train_pred = learner.query(X_train)
        test_pred = learner.query(X_test)

        train_rmse = np.sqrt(np.mean((y_train - train_pred) ** 2))
        test_rmse = np.sqrt(np.mean((y_test - test_pred) ** 2))

        expected_train = 233.23
        tolerance = 1.0

        assert abs(train_rmse - expected_train) < tolerance, \
            f"Info Gain Train RMSE: expected {expected_train:.2f}, got {train_rmse:.2f}"

        print(f"✓ Task 5f: Info Gain - Train: {train_rmse:.4f}, Test: {test_rmse:.4f}")

    def test_task5f_infogain_similar_to_gini(self, bikes_data):
        """Task 5f: Info Gain should be similar to Gini (both use binning)"""
        X_train, y_train, X_test, y_test = bikes_data

        info_learner = DTLearner(leaf_size=1, max_depth=4, metric='information_gain')
        gini_learner = DTLearner(leaf_size=1, max_depth=4, metric='gini')

        info_learner.add_evidence(X_train, y_train)
        gini_learner.add_evidence(X_train, y_train)

        info_pred = info_learner.query(X_train)
        gini_pred = gini_learner.query(X_train)

        info_rmse = np.sqrt(np.mean((y_train - info_pred) ** 2))
        gini_rmse = np.sqrt(np.mean((y_train - gini_pred) ** 2))

        difference = abs(info_rmse - gini_rmse)
        assert difference < 10, \
            f"Info Gain ({info_rmse:.2f}) should be similar to Gini ({gini_rmse:.2f}). " \
            f"Check that you're using quartile binning. Difference should be <10, got {difference:.2f}"

        print(f"✓ Task 5f: Info Gain ≈ Gini: {info_rmse:.2f} vs {gini_rmse:.2f}")


# ============================================================================
# SUMMARY: All Tasks Complete
# ============================================================================

class TestSummary:
    """Summary of all tasks"""

    @pytest.fixture(scope="class")
    def bikes_data(self):
        """Load bikes dataset with 80/20 shuffled split (matching main.py)"""
        df = pd.read_csv('data/bikes.csv')
        X = df.iloc[:, 1:-1].values  # Skip first column (date)
        y = df.iloc[:, -1].values

        # Use same split as main.py: 80/20 with random_state=42
        np.random.seed(42)
        n_samples = len(y)
        n_train = int(n_samples * 0.8)
        indices = np.random.permutation(n_samples)
        train_idx = indices[:n_train]
        test_idx = indices[n_train:]

        return X[train_idx], y[train_idx], X[test_idx], y[test_idx]

    def test_summary_all_metrics(self, bikes_data):
        """Summary: Test all metrics and print results table"""
        X_train, y_train, X_test, y_test = bikes_data

        metrics = [
            'correlation',
            'gini',
            'variance_reduction',
            'mse_reduction',
            'mae_reduction',
            'information_gain'
        ]

        print("\n" + "=" * 80)
        print("SUMMARY: All Tasks - Bikes Dataset Results")
        print("=" * 80)
        print(f"{'Task':<10} {'Metric':<20} {'Train RMSE':<15} {'Test RMSE':<15}")
        print("-" * 80)

        task_names = {
            'correlation': 'Task 5a',
            'gini': 'Task 5b',
            'variance_reduction': 'Task 5c',
            'mse_reduction': 'Task 5e',
            'mae_reduction': 'Task 5d',
            'information_gain': 'Task 5f'
        }

        for metric in metrics:
            learner = DTLearner(leaf_size=1, max_depth=4, metric=metric)
            learner.add_evidence(X_train, y_train)

            train_pred = learner.query(X_train)
            test_pred = learner.query(X_test)

            train_rmse = np.sqrt(np.mean((y_train - train_pred) ** 2))
            test_rmse = np.sqrt(np.mean((y_test - test_pred) ** 2))

            task = task_names.get(metric, 'Task ?')
            print(f"{task:<10} {metric:<20} {train_rmse:<15.4f} {test_rmse:<15.4f}")

        print("=" * 80)
        print("✓ ALL TASKS COMPLETE!")
        print("=" * 80)


if __name__ == "__main__":
    """Run tests directly with python"""
    import sys

    print("\n" + "=" * 80)
    print("Running DTLearner Tests by Task")
    print("=" * 80 + "\n")

    # Run with pytest
    pytest.main([__file__, "-v", "--tb=short"])