"""
Utility functions for DTLearner lab

This file contains helper functions for loading and creating data.
Students do not need to modify this file.

(c) 2025 Hybinette
"""

import pandas as pd
import numpy as np


def create_fake_data():
    """
    Creates fake data for testing when CSV files are not available.

    Returns:
    - X: pd.DataFrame with feature data
    - Y: pd.Series with target data
    """
    # Create simple fake data
    fake_data = pd.DataFrame({
        'X1': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        'X2': [1.0, 1.5, 2.0, 3.0, 4.0, 5.0],
        'X3': [5.0, 4.5, 4.0, 3.0, 2.0, 1.0]
    })
    fake_target = pd.Series([10, 15, 20, 25, 30, 35], name='Y')

    print(f"Created fake dataset with {len(fake_data)} rows and {len(fake_data.columns)} features.")
    print(f"Feature columns: {fake_data.columns.tolist()}")

    return fake_data, fake_target


def create_synthetic_data(n_samples=100, n_features=3, random_state=42):
    """
    Creates synthetic data with known relationships.

    Parameters:
    - n_samples: Number of samples to generate
    - n_features: Number of features
    - random_state: Random seed for reproducibility

    Returns:
    - data: pd.DataFrame with all data (features + target)
    - features: list of feature column names
    - target: pd.Series with target values
    """
    np.random.seed(random_state)

    # Create feature data
    feature_names = [f'feature_{i + 1}' for i in range(n_features)]
    X = pd.DataFrame(
        np.random.rand(n_samples, n_features) * 10,
        columns=feature_names
    )

    # Create target with known relationship to features
    # y = 2*x1 + 3*x2 - x3 + ... + noise
    coefficients = [2, 3, -1] + [0.5] * (n_features - 3)
    Y = pd.Series(0.0, index=range(n_samples), name='target')

    for i, coef in enumerate(coefficients[:n_features]):
        Y += coef * X.iloc[:, i]

    # Add noise
    Y += np.random.randn(n_samples) * 2

    # Combine into single dataframe
    data = X.copy()
    data['target'] = Y

    print(f"Created synthetic dataset with {n_samples} samples and {n_features} features.")
    print(f"Feature columns: {feature_names}")
    print(f"Target column: target")

    return data, feature_names, Y


def process_dataset(file_path, ignore_first_column=False, target_column=None,
                    create_synthetic_if_missing=True):
    """
    Loads and processes the dataset.

    Parameters:
    - file_path: str, the path to the dataset CSV file
    - ignore_first_column: bool, if True, the first column will be ignored (e.g., row indices)
    - target_column: str, the name of the target column (if None, assumes the last column)
    - create_synthetic_if_missing: bool, if True, creates synthetic data when file not found

    Returns:
    - data: pd.DataFrame, the loaded data
    - features: list, the list of feature column names
    - target: pd.Series, the target values
    """
    try:
        # Load the dataset from the CSV file
        data = pd.read_csv(file_path)

        # Optionally ignore the first column
        if ignore_first_column:
            data = data.iloc[:, 1:]

        # Remove duplicate rows
        original_len = len(data)
        data = data.drop_duplicates()
        if len(data) < original_len:
            print(f"Removed {original_len - len(data)} duplicate rows.")

        # If no target column is specified, use the last column as the target
        if target_column is None:
            target_column = data.columns[-1]

        # Split the data into features and target
        features = data.drop(columns=[target_column]).columns.tolist()
        target = data[target_column]

        print(f"Loaded dataset with {len(data)} rows and {len(features)} features.")
        print(f"Target column: {target_column}")
        print(f"Feature columns: {features}")

        return data, features, target

    except FileNotFoundError:
        print(f"\nError: Could not find '{file_path}'")

        if create_synthetic_if_missing:
            print("Creating synthetic data for demonstration...\n")
            return create_synthetic_data(n_samples=100, n_features=3)
        else:
            raise


if __name__ == "__main__":
    # Demo the utility functions
    print("=" * 70)
    print("Utility Functions Demo")
    print("=" * 70)

    # Demo 1: Create fake data
    print("\n--- Creating Fake Data ---")
    X, Y = create_fake_data()
    print("\nX:")
    print(X)
    print("\nY:")
    print(Y)

    # Demo 2: Try loading CSV
    print("\n\n--- Loading CSV Data ---")
    try:
        data, features, target = process_dataset('data/wine-simple.csv', ignore_first_column=True)
        print("\nFirst few rows:")
        print(data.head())
    except FileNotFoundError:
        print("CSV file not found. Place wine-simple.csv in data/ folder to test.")