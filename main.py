"""
Main script for DTLearner workflow

This script demonstrates the complete workflow:
1. Read the dataset
2. Learn and build a decision tree model
3. Visualize the tree (text or graphviz)
4. Test/query the model
5. Evaluate the model with error metrics

Usage:
    python main.py                              # Use default settings
    python main.py --data data/wine.csv         # Specify data file
    python main.py --ignore-first-col or -i     # Ignore first column (row numbers/index)
    python main.py --metric correlation         # Use correlation metric (default)
    python main.py --metric gini                # Use Gini metric
    python main.py --viz text                   # Use text visualization
    python main.py --viz graphviz               # Use Graphviz visualization
    python main.py --verbose or -v              # Enable verbose tree building output
    python main.py --leaf-size 5                # Set minimum samples at leaf
    python main.py --max-depth 10               # Set maximum tree depth

    # Custom parameters
    python main.py --data data/bike.csv --metric gini --leaf-size 5 --verbose

    # All together
    python main.py --data data/wine.csv --ignore-first-col --metric correlation --viz graphviz -v --leaf-size 3 --max-depth 7

# Earlier Lab - Recreate the Wine Tree with correlation default/and larger data set
python main.py --data data/wine-simple.csv --i --viz graphviz --leaf-size 1 --max-depth 7 --metric correlation
python main.py --data data/winequality-red.csv --viz graphviz --leaf-size 1 --max-depth 4 --metric correlation
python main.py --data data/winequality-white.csv --viz graphviz --leaf-size 1 --max-depth 4 --metric correlation

# New Lab HW08b - Recreate the Wine Tree using GINI as metric
python main.py --data data/wine-simple.csv -i --viz graphviz -v --leaf-size 1 --max-depth 4 --metric gini
python main.py --data data/winequality-white.csv --viz graphviz -v --leaf-size 1 --max-depth 4 --metric gini
python main.py --data data/bikes.csv -i --viz graphviz -v --leaf-size 1 --max-depth 4 --metric gini
python main.py --data data/bikes.csv -i --viz graphviz -v --leaf-size 1 --max-depth 4 --metric variance_reduction


Flags:
    --data FILE                                         # Path to CSV data file (default: data/wine-simple.csv)
    --ignore-first-col                                  # Ignore first column (e.g., row index column)
    --metric {correlation,gini,information_gain,        # Feature selection metric (default: correlation)
              mse_reduction,mae_reduction,
              variance_reduction}
    --viz {text,k}                               # Visualization method (default: text)
    --verbose or -v                                     # Enable verbose tree building output
    --leaf-size N                                       # Minimum samples at leaf (default: 1)
    --max-depth N                                       # Maximum tree depth (default: 5)


(c) 2025 Hybinette

"""
import sys
import os
import numpy as np
import pandas as pd
from DTLearner_skeleton import DTLearner
from utils import process_dataset
from tree_visualization import visualize_tree, check_graphviz_available


def calculate_metrics(y_true, y_pred):
    """
    Calculate various error metrics.

    Args:
        y_true: Actual target values
        y_pred: Predicted target values

    Returns:
        dict: Dictionary of metrics
    """
    errors = y_pred - y_true

    metrics = {
        'MAE': np.mean(np.abs(errors)),
        'MSE': np.mean(errors ** 2),
        'RMSE': np.sqrt(np.mean(errors ** 2)),
        'Max Error': np.max(np.abs(errors)),
        'R²': 1 - (np.sum(errors ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))
    }

    return metrics


def print_metrics(metrics, dataset_name=""):
    """
    Print error metrics in a formatted table.

    Args:
        metrics: Dictionary of metric names and values
        dataset_name: Name of dataset for display
    """
    print(f"\n{'=' * 60}")
    print(f"Error Metrics{' - ' + dataset_name if dataset_name else ''}")
    print(f"{'=' * 60}")

    for metric_name, value in metrics.items():
        print(f"{metric_name:>15s}: {value:>10.4f}")

    print(f"{'=' * 60}")


def split_data(X, Y, train_ratio=0.8, random_state=None):
    """
    Split data into training and testing sets.

    Args:
        X: Feature data
        Y: Target data
        train_ratio: Proportion of data for training
        random_state: Random seed for reproducibility

    Returns:
        X_train, X_test, Y_train, Y_test
    """
    if random_state is not None:
        np.random.seed(random_state)

    n_samples = len(Y)
    n_train = int(n_samples * train_ratio)

    # Shuffle indices
    indices = np.random.permutation(n_samples)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    if isinstance(X, pd.DataFrame):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
    else:
        X_train = X[train_idx]
        X_test = X[test_idx]

    if isinstance(Y, pd.Series):
        Y_train = Y.iloc[train_idx]
        Y_test = Y.iloc[test_idx]
    else:
        Y_train = Y[train_idx]
        Y_test = Y[test_idx]

    return X_train, X_test, Y_train, Y_test


def main(data_file='data/wine-simple.csv',
         ignore_first_col=True,
         visualization_method='text',
         verbose=False,
         leaf_size=1,
         max_depth=5,
         metric='correlation',
         test_set_mode='insample'
         ):
    """
    Main workflow for DTLearner:
    1. Read dataset
    2. Build model
    3. Visualize tree
    4. Test model
    5. Evaluate model

    Args:
        data_file: Path to CSV data file
        ignore_first_col: Whether to ignore first column (e.g., row index)
        visualization_method: 'text' or 'graphviz'
        verbose: Enable verbose output during tree building
        leaf_size: Minimum samples at leaf node
        max_depth: Maximum tree depth
        metric: Feature selection metric ('correlation' or 'gini')
    """
    print("=" * 70)
    print("DTLearner Complete Workflow")
    print("=" * 70)
    print(f"Data file: {data_file}")
    print(f"Visualization method: {visualization_method}")
    # Check graphviz availability if requested
    if visualization_method == 'graphviz':
        if not check_graphviz_available():
            print("\nWarning: Graphviz not available, will fall back to text")
    print(f"Evaluation mode: {test_set_mode}")  # <-- Add this line for clarity

    # ========================================================================
    # STEP 1: READ THE DATASET
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 1: Reading Dataset")
    print("=" * 70)

    # Load data - automatically creates synthetic data if file not found
    data, features, target = process_dataset(
        data_file,
        ignore_first_column=ignore_first_col,
        create_synthetic_if_missing=True
    )

    print("\nDataset preview:")
    print(data.head(10))

    # Extract X and Y
    X = data[features]
    Y = target

    print(f"\nDataset statistics:")
    print(f"  Number of samples: {len(X)}")
    print(f"  Number of features: {len(features)}")
    print(f"  Target range: [{Y.min():.2f}, {Y.max():.2f}]")
    print(f"  Target mean: {Y.mean():.2f}")
    print(f"  Target std: {Y.std():.2f}")

    # Split data into train/test
    print("\n" + "-" * 70)
    print("Splitting data into train/test sets (80/20)...")
    X_train, X_test, Y_train, Y_test = split_data(X, Y, train_ratio=0.8, random_state=42)
    print(f"  Training set: {len(X_train)} samples")
    print(f"  Test set: {len(X_test)} samples")

    # ========================================================================
    # STEP 2: LEARN & BUILD MODEL (DECISION TREE)
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 2: Building Decision Tree Model")
    print("=" * 70)

    # Create learner with parameters from function arguments
    learner = DTLearner(
        leaf_size=leaf_size,
        max_depth=max_depth,
        verbose=verbose,
        metric=metric,
        tie_breaker='first'
    )

    print(f"\nModel parameters:")
    print(f"  Leaf size: {learner.leaf_size}")
    print(f"  Max depth: {learner.max_depth}")
    print(f"  Metric: {learner.metric}")
    print(f"  Tie breaker: {learner.tie_breaker}")

    # Train the model --- OLD METHOD -- Making it Right!
    # Option 1: Train on full dataset (for visualization/debugging)
    # print("\nTraining model on full dataset...")
    # learner.add_evidence(X, Y)

    # Op# tion 2: Train on training set only (for proper evaluation)
    # print("\nTraining model on training set...")
    # learner.add_evidence(X_train, Y_train)

    # Train the model based on the test_set_mode
    if test_set_mode == 'insample':
        # Option 1: Train on full dataset (for visualization/debugging)
        print("\nTraining model on full dataset (in-sample mode)...")
        learner.add_evidence(X, Y)
    else:  # test_set_mode == 'outsample'
        # Option 2: Train on a training set only (for proper evaluation)
        print("\nTraining model on training set (out-of-sample mode)...")
        learner.add_evidence(X_train, Y_train)

    print("✓ Training complete!")

    # ========================================================================
    # STEP 3: VISUALIZE THE TREE
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 3: Tree Visualization")
    print("=" * 70)

    print(f"\nUsing {visualization_method} visualization:")
    print("-" * 70)

    if visualization_method == 'text':
        visualize_tree(learner.tree, feature_names=features, method='text', max_depth=6)
    elif visualization_method == 'graphviz':
        # Create an output directory if it doesn't exist
        output_dir = 'out'
        os.makedirs(output_dir, exist_ok=True)

        # Generate output filename based on input file and max_depth
        # Extract base filename without extension
        base_filename = os.path.splitext(os.path.basename(data_file))[0]
        depth_str = f"{max_depth:02d}" if max_depth is not None else "full"
        output_filename = os.path.join(output_dir, f"{base_filename}-depth-{depth_str}")

        print(f"Output files will be: {output_filename}.pdf and {output_filename}.png")

        graph = visualize_tree(learner.tree, feature_names=features,
                               method='graphviz', filename=output_filename, view=True)
        if graph is not None:
            print("\n(Graph saved and will open automatically if system supports it)")
    else:
        print(f"Unknown method '{visualization_method}', using text...")
        visualize_tree(learner.tree, feature_names=features, method='text', max_depth=6)

    print("-" * 70)

    # ========================================================================
    # STEP 4: TEST/QUERY THE MODEL
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 4: Making Predictions")
    print("=" * 70)

    # Make predictions on the training set
    print("\nPredicting on training set...")
    train_predictions = learner.query(X_train)

    # Make predictions on the test set
    print("Predicting on test set...")
    test_predictions = learner.query(X_test)

    print("✓ Predictions complete!")

    # Show sample predictions
    print("\nSample predictions (first 10 test samples):")
    print(f"{'Actual':>10s} {'Predicted':>10s} {'Error':>10s}")
    print("-" * 35)
    for i in range(min(10, len(Y_test))):
        actual = Y_test.iloc[i] if isinstance(Y_test, pd.Series) else Y_test[i]
        predicted = test_predictions[i]
        error = predicted - actual
        print(f"{actual:>10.4f} {predicted:>10.4f} {error:>10.4f}")

    # ========================================================================
    # STEP 5: EVALUATE THE MODEL
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 5: Model Evaluation")
    print("=" * 70)

    # Calculate metrics for training set
    Y_train_array = Y_train.values if isinstance(Y_train, pd.Series) else Y_train
    train_metrics = calculate_metrics(Y_train_array, train_predictions)
    print_metrics(train_metrics, "Training Set")

    # Calculate metrics for test set
    Y_test_array = Y_test.values if isinstance(Y_test, pd.Series) else Y_test
    test_metrics = calculate_metrics(Y_test_array, test_predictions)
    print_metrics(test_metrics, "Test Set")

    # Check for overfitting
    print("\n" + "=" * 70)
    if test_set_mode == 'insample':
        print("STEP 5: Overfitting Analysis (Warning - In-Sample Mode)")
    else:
        print("STEP 5: Overfitting Analysis (Out-of-Sample Mode)")
    print("=" * 70)
    train_rmse = train_metrics['RMSE']
    test_rmse = test_metrics['RMSE']
    ratio = test_rmse / train_rmse if train_rmse > 0 else float('inf')

    print(f"Train RMSE: {train_rmse:.4f}")
    print(f"Test RMSE:  {test_rmse:.4f}")
    print(f"Ratio (Test/Train): {ratio:.4f}")

    if ratio < 1.1:
        print("✓ Model generalizes well (minimal overfitting)")
    elif ratio < 1.5:
        print("⚠ Model shows some overfitting")
    else:
        print("✗ Model is overfitting significantly")

    # ========================================================================
    # BONUS: COMPARE DIFFERENT CONFIGURATIONS
    # ========================================================================
    print("\n\n" + "=" * 70)
    print("BONUS: Comparing Different Configurations (Out of Sample)")
    print("=" * 70)

    configs = [
        {'leaf_size': 1, 'max_depth': 3, 'metric': 'correlation'},
        {'leaf_size': 5, 'max_depth': 5, 'metric': 'correlation'},
        {'leaf_size': 10, 'max_depth': 10, 'metric': 'correlation'},
        {'leaf_size': 5, 'max_depth': 5, 'metric': 'gini'},

        {'leaf_size': 1, 'max_depth': 4, 'metric': 'correlation'},
        {'leaf_size': 1, 'max_depth': 4, 'metric': 'gini'},

    ]

    print("\nComparing configurations on test set:")
    print(f"{'Config':>30s} {'Test RMSE':>12s} {'Test R²':>12s}")
    print("-" * 60)

    for config in configs:
        learner_temp = DTLearner(verbose=False, **config)
        learner_temp.add_evidence(X_train, Y_train)
        preds = learner_temp.query(X_test)
        metrics = calculate_metrics(Y_test_array, preds)

        config_str = f"ls={config['leaf_size']}, md={config['max_depth']}, {config['metric'][:4]}"
        print(f"{config_str:>30s} {metrics['RMSE']:>12.4f} {metrics['R²']:>12.4f}")

    print("\n" + "=" * 70)
    print("Workflow Complete!")
    print("=" * 70)


if __name__ == "__main__":
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='DTLearner Complete Workflow')
    parser.add_argument('--data', type=str, default='data/wine-simple.csv',
                        help='Path to CSV data file (default: data/wine-simple.csv)')
    parser.add_argument('--ignore-first-col', '-i',
                        action='store_true',
                        help='Ignore first column (e.g., row index column)')
    parser.add_argument('--metric', type=str,
                        choices=['correlation', 'gini', 'information_gain',
                                'mse_reduction', 'mae_reduction', 'variance_reduction'],
                        default='correlation',
                        help='Feature selection metric (default: correlation)')
    parser.add_argument('--viz', '--visualization',
                        choices=['text', 'graphviz'],
                        default='text',
                        help='Visualization method: text or graphviz (default: text)')
    parser.add_argument('--verbose', '-v',
                        action='store_true',
                        help='Enable verbose output during tree building')
    parser.add_argument('--leaf-size', type=int, default=1,
                        help='Minimum samples at leaf node (default: 1)')
    parser.add_argument('--max-depth', type=int, default=5,
                        help='Maximum tree depth (default: 5)')

    parser.add_argument('--test-set',
                        choices=['insample', 'outsample'],
                        default='insample',
                        help="Evaluation mode: "
                             "\n\t'insample' (train on full data, test on test set - good for viz) or "\
                             "\n\t'outsample' (train on train set, test on test set - correct evaluation) (default: insample)")

    args = parser.parse_args()

    # Run the main workflow with all arguments
    main(
        data_file=args.data,
        ignore_first_col=args.ignore_first_col,
        visualization_method=args.viz,
        verbose=args.verbose,
        leaf_size=args.leaf_size,
        max_depth=args.max_depth,
        metric=args.metric,
        test_set_mode=args.test_set
    )

"""
Quick Command Line Examples:
python main.py --data data/bikes.csv -i --leaf-size 1 --max-depth 4 --metric correlation --verbose
python main.py --data data/bikes.csv -i --leaf-size 1 --max-depth 4 --metric correlation
python main.py --data data/bikes.csv -i --leaf-size 1 --max-depth 4 --metric gini 
python main.py --data data/bikes.csv -i --leaf-size 1 --max-depth 4 --metric variance_reduction
python main.py --data data/bikes.csv -i --leaf-size 1 --max-depth 4 --metric mse_reduction
python main.py --data data/bikes.csv -i --leaf-size 1 --max-depth 4 --metric mae_reduction
python main.py --data data/bikes.csv -i --leaf-size 1 --max-depth 4 --metric information_gain
python --help
python main.py --data data/wine-simple.csv -i --leaf-size 1 --max-depth 4 --metric correlation --viz graphviz
"""