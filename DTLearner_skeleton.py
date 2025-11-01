import numpy as np
import pandas as pd

"""
DTLearner Skeleton for HW08

Complete the TODO sections to implement a decision tree learner with multiple splitting metrics.
See README-metrics.md for mathematical details on each metric.

Test your implementation by running: python main.py

(c) 2025 Hybinette
"""


class DTLearner:
    def __init__(self, leaf_size=1, max_depth=None, verbose=False,
                 metric='correlation', tie_breaker='first', tie_tolerance=1e-9):
        """
        Initialize the decision tree learner.

        Args:
            leaf_size: Minimum number of samples required to be at a leaf node
            max_depth: Maximum depth of the tree (None = no limit)
            verbose: Print debug information
            metric: Feature selection metric - one of:
                'correlation' - Absolute correlation with target (default)
                'gini' - Gini impurity (StatQuest method with binning)
                'variance_reduction' - Standard CART approach
                'mae_reduction' - Mean absolute error reduction
                'mse_reduction' - Mean squared error reduction
                'information_gain' - Entropy-based splitting
            tie_breaker: How to break ties ('first', 'random', 'last')
            tie_tolerance: Tolerance for considering scores as tied (default: 1e-9)
        """
        self.leaf_size = leaf_size
        self.max_depth = max_depth
        self.verbose = verbose
        self.metric = metric
        self.tie_breaker = tie_breaker
        self.tie_tolerance = tie_tolerance
        self.tree = None

        # Validate metric
        valid_metrics = ['correlation', 'gini', 'variance_reduction',
                         'mae_reduction', 'mse_reduction', 'information_gain']
        if self.metric not in valid_metrics:
            raise ValueError(f"metric must be one of {valid_metrics}")

        # Validate tie_breaker
        valid_tie_breakers = ['first', 'last', 'random']
        if self.tie_breaker not in valid_tie_breakers:
            raise ValueError(f"tie_breaker must be one of {valid_tie_breakers}")

        if self.verbose:
            print(f"Initialized DTLearner with leaf_size={leaf_size}, max_depth={max_depth}")
            print(f"  metric={metric}, tie_breaker={tie_breaker}, tie_tolerance={tie_tolerance}")

    # ========================================================================
    # CORRELATION METRIC
    # ========================================================================

    def calculate_correlation(self, dataX, dataY, feature_idx):
        """
        Calculate absolute correlation between a feature and target.

        Args:
            dataX: Feature data
            dataY: Target data
            feature_idx: Index of feature to evaluate

        Returns:
            correlation: Absolute correlation (0 to 1, higher is better)
        """
        if self.verbose:
            print(f"    TODO: Calculate correlation for feature {feature_idx}")
            print("    Hint: Use np.corrcoef() and handle NaN values")

        # TODO: Calculate correlation using np.corrcoef
        # Hint: corr_matrix = np.corrcoef(dataX[:, feature_idx], dataY)
        # Hint: correlation = abs(corr_matrix[0, 1])
        # Hint: Handle NaN (zero variance) by returning 0.0
        # Hint: Use np.isnan() to check for NaN

        corr_matrix = np.corrcoef(dataX[:, feature_idx], dataY)
        correlation = abs(corr_matrix[0,1])

        if np.isnan(correlation):
            return 0        

        return correlation

    # ========================================================================
    # GINI IMPURITY (StatQuest Method)
    # See README-metrics.md and README-quick-reference.md
    # ========================================================================

    def calculate_gini_gain(self, dataX, dataY, feature_idx):
        """
        Calculate Gini gain using StatQuest method.

        This method:
        1. Tries splits at adjacent averages (midpoints between consecutive X values)
        2. For each split, bins Y values (quartiles for continuous, categories for discrete)
        3. Calculates Gini = 1 - Σ(p_i²) where p_i is proportion in each bin
        4. Chooses split with minimum weighted Gini

        See README-metrics.md for detailed mathematical explanation.

        Args:
            dataX: Feature data
            dataY: Target data
            feature_idx: Index of feature to evaluate

        Returns:
            gain: Reduction in Gini impurity (higher is better)
        """
        if self.verbose:
            print(f"    TODO: Calculate Gini gain for feature {feature_idx}")
            print("    Hint: See README-quick-reference.md for step-by-step guide")

        # TODO: Implement StatQuest Gini method
        # Step 1: Get unique sorted feature values
        # Step 2: Calculate adjacent averages as thresholds
        # Step 3: For each threshold, split data and calculate Gini
        # Step 4: Return best gain

        # Hint: See README-quick-reference.md for common mistakes to avoid!
        # Hint: For continuous Y, bin into quartiles using np.percentile
        # Hint: For discrete Y (few unique values), use categories directly

        gain = 0.0  # Placeholder
        return gain

    # ========================================================================
    # VARIANCE REDUCTION (Standard CART)
    # ========================================================================

    def calculate_variance_reduction(self, dataX, dataY, feature_idx):
        """
        Calculate variance reduction (standard CART approach).

        Args:
            dataX: Feature data
            dataY: Target data
            feature_idx: Index of feature to evaluate

        Returns:
            reduction: Reduction in variance (higher is better)
        """
        if self.verbose:
            print(f"    TODO: Calculate variance reduction for feature {feature_idx}")
            print("    Hint: Split at median, calculate weighted variance")

        # TODO: Implement variance reduction
        # Hint: current_variance = np.var(dataY)
        # Hint: Split at median: split_val = np.median(dataX[:, feature_idx])
        # Hint: Calculate weighted variance after split
        # Hint: reduction = current_variance - weighted_variance

        reduction = 0.0  # Placeholder
        return reduction

    # ========================================================================
    # MAE REDUCTION (Robust to outliers)
    # ========================================================================

    def calculate_mae_reduction(self, dataX, dataY, feature_idx):
        """
        Calculate MAE (Mean Absolute Error) reduction.

        Args:
            dataX: Feature data
            dataY: Target data
            feature_idx: Index of feature to evaluate

        Returns:
            reduction: Reduction in MAE (higher is better)
        """
        if self.verbose:
            print(f"    TODO: Calculate MAE reduction for feature {feature_idx}")
            print("    Hint: Use median instead of mean for robustness")

        # TODO: Implement MAE reduction
        # Hint: current_mae = np.mean(np.abs(dataY - np.median(dataY)))
        # Hint: Split at median of feature
        # Hint: Calculate MAE for left and right groups using their medians
        # Hint: reduction = current_mae - weighted_mae

        reduction = 0.0  # Placeholder
        return reduction

    # ========================================================================
    # MSE REDUCTION (Mathematically equivalent to variance)
    # ========================================================================

    def calculate_mse_reduction(self, dataX, dataY, feature_idx):
        """
        Calculate MSE (Mean Squared Error) reduction.

        Note: This is mathematically equivalent to variance_reduction.

        Args:
            dataX: Feature data
            dataY: Target data
            feature_idx: Index of feature to evaluate

        Returns:
            reduction: Reduction in MSE (higher is better)
        """
        if self.verbose:
            print(f"    TODO: Calculate MSE reduction for feature {feature_idx}")
            print("    Hint: Should give same results as variance_reduction")

        # TODO: Implement MSE reduction
        # Hint: This should give same results as variance_reduction
        # Hint: current_mse = np.mean((dataY - np.mean(dataY))**2)
        # Hint: Split at median of feature
        # Hint: Calculate weighted MSE after split

        reduction = 0.0  # Placeholder
        return reduction

    # ========================================================================
    # INFORMATION GAIN (Entropy-based)
    # ========================================================================

    def calculate_information_gain(self, dataX, dataY, feature_idx):
        """
        Calculate information gain (entropy reduction).

        Args:
            dataX: Feature data
            dataY: Target data
            feature_idx: Index of feature to evaluate

        Returns:
            gain: Information gain (higher is better)
        """
        if self.verbose:
            print(f"    TODO: Calculate information gain for feature {feature_idx}")
            print("    Hint: Discretize Y into bins, calculate entropy")

        # TODO: Implement information gain
        # Hint: For regression, discretize Y into bins
        # Hint: Calculate entropy: -Σ(p * log2(p)) for each bin
        # Hint: Split at median of feature
        # Hint: gain = entropy_before - weighted_entropy_after

        gain = 0.0  # Placeholder
        return gain

    # ========================================================================
    # FEATURE SELECTION
    # ========================================================================

    def calculate_feature_scores(self, dataX, dataY):
        """
        Calculate scores for all features based on the selected metric.

        Args:
            dataX: Feature data as numpy array (N samples x M features)
            dataY: Target data as numpy array (N samples)

        Returns:
            scores: Array of scores for each feature (higher is better)
        """
        n_features = dataX.shape[1]
        scores = np.zeros(n_features)

        for i in range(n_features):
            if self.metric == 'correlation':
                scores[i] = self.calculate_correlation(dataX, dataY, i)

            elif self.metric == 'gini':
                scores[i] = self.calculate_gini_gain(dataX, dataY, i)

            elif self.metric == 'variance_reduction':
                scores[i] = self.calculate_variance_reduction(dataX, dataY, i)

            elif self.metric == 'mae_reduction':
                scores[i] = self.calculate_mae_reduction(dataX, dataY, i)

            elif self.metric == 'mse_reduction':
                scores[i] = self.calculate_mse_reduction(dataX, dataY, i)

            elif self.metric == 'information_gain':
                scores[i] = self.calculate_information_gain(dataX, dataY, i)

            else:
                raise ValueError(f"Unknown metric: {self.metric}")

        return scores

    def select_best_feature(self, dataX, dataY):
        """
        Select the best feature to split on, with tie-breaking strategy.

        Args:
            dataX: Feature data as numpy array (N samples x M features)
            dataY: Target data as numpy array (N samples)

        Returns:
            best_feature: Index of the best feature to split on
            split_val: The median value of that feature to use as split point
        """
        if self.verbose:
            print(f"\nSelecting best feature from {dataX.shape[1]} features (metric={self.metric})")

        # Calculate scores for all features
        scores = self.calculate_feature_scores(dataX, dataY)

        if self.verbose:
            for i, score in enumerate(scores):
                print(f"  Feature {i}: score = {score:.4f}")

        # TODO: Find the maximum score
        if self.verbose:
            print("  TODO: Find maximum score")
        max_score = 0.0  # Placeholder - replace with np.max(scores)

        # TODO: Find all features with maximum score (handle ties with tolerance)
        if self.verbose:
            print("  TODO: Identify tied features using tolerance")
        # Hint: np.abs(scores - max_score) < self.tie_tolerance gives a boolean array
        # Hint: You need to return INDICES, not booleans. Use np.where() to get indices.
        #   1) Hint: Use np.where() to find indices of tied features
        #   2) Hint: Use np.array() to convert list to numpy array
        tied_features = np.array([0])  # Placeholder - replace with proper tie detection

        if self.verbose and len(tied_features) > 1:
            print(f"  Tie detected! Features {tied_features} all have score {max_score:.4f}")

        # TODO: Break ties according to tie_breaker strategy
        if self.verbose:
            print(f"  TODO: Apply tie-breaking strategy '{self.tie_breaker}'")

        if self.tie_breaker == 'first':
            best_feature = 0  # Placeholder - select tied_features[0]
        elif self.tie_breaker == 'last':
            best_feature = 0  # Placeholder - select tied_features[-1]
        elif self.tie_breaker == 'random':
            best_feature = 0  # Placeholder - use np.random.choice(tied_features)

        # TODO: Calculate split value as median of best feature
        if self.verbose:
            print("  TODO: Calculate split value as median")
        split_val = 0.0  # Placeholder - replace with np.median(dataX[:, best_feature])

        if self.verbose:
            print(f"  --> Selected feature {best_feature} (tie_breaker={self.tie_breaker})")
            print(f"      Split value: {split_val:.4f}")

        return best_feature, split_val

    # ========================================================================
    # TREE BUILDING
    # ========================================================================

    def build_tree(self, dataX, dataY, depth=0):
        """
        Recursively build the decision tree.

        Args:
            dataX: Feature data (N x M numpy array)
            dataY: Target data (N numpy array)
            depth: Current depth in tree (used for max_depth check)

        Returns:
            Tree structure as list: [feature_idx, split_val, left_tree, right_tree]
            or leaf value if it's a leaf node
        """
        if self.verbose:
            print(f"\n{'  ' * depth}Building tree at depth {depth}, samples={dataX.shape[0]}")
            print(f"{'  ' * depth}TODO: Check base cases (max_depth, leaf_size, all same Y)")

        # TODO: Base case 1 - Check if we've reached max depth
        if self.verbose and self.max_depth is not None and depth >= self.max_depth:
            print(f"{'  ' * depth}TODO: Max depth reached, should create leaf")
        # Hint: if self.max_depth is not None and depth >= self.max_depth:
        #           return np.mean(dataY)

        # TODO: Base case 2 - Check if we have too few samples
        if self.verbose and dataX.shape[0] <= self.leaf_size:
            print(f"{'  ' * depth}TODO: Too few samples ({dataX.shape[0]} <= {self.leaf_size}), should create leaf")
        # Hint: if dataX.shape[0] <= self.leaf_size:
        #           return np.mean(dataY)

        # TODO: Base case 3 - Check if all Y values are the same
        if self.verbose and len(np.unique(dataY)) == 1:
            print(f"{'  ' * depth}TODO: All Y values same ({dataY[0]}), should create leaf")
        # Hint: if np.all(dataY == dataY[0]):
        #           return dataY[0]

        # Find best feature to split on
        best_feature, split_val = self.select_best_feature(dataX, dataY)

        # TODO: Split the data based on best_feature and split_val
        if self.verbose:
            print(f"{'  ' * depth}TODO: Split data on feature {best_feature} at value {split_val:.4f}")
        # Hint: left_idx = dataX[:, best_feature] <= split_val
        # Hint: right_idx = dataX[:, best_feature] > split_val
        left_idx = np.ones(len(dataY), dtype=bool)  # Placeholder
        right_idx = np.zeros(len(dataY), dtype=bool)  # Placeholder

        # TODO: Check if split actually separates the data
        if self.verbose:
            print(f"{'  ' * depth}TODO: Check if split separates data (left={left_idx.sum()}, right={right_idx.sum()})")
        # Hint: if left_idx.sum() == 0 or right_idx.sum() == 0:
        #           return np.mean(dataY)

        if self.verbose:
            print(f"{'  ' * depth}Splitting on feature {best_feature}, " +
                  f"left={left_idx.sum()}, right={right_idx.sum()}")
            print(f"{'  ' * depth}TODO: Recursively build left and right subtrees")

        # TODO: Recursively build left and right subtrees
        # Hint: left_tree = self.build_tree(dataX[left_idx], dataY[left_idx], depth + 1)
        # Hint: right_tree = self.build_tree(dataX[right_idx], dataY[right_idx], depth + 1)
        left_tree = np.mean(dataY)  # Placeholder
        right_tree = np.mean(dataY)  # Placeholder

        # Return node as a list: [best_feature, split_val, left_tree, right_tree]
        return [best_feature, split_val, left_tree, right_tree]

    def add_evidence(self, dataX, dataY):
        """
        Train the model by building the decision tree.

        Args:
            dataX: Feature data (can be pandas DataFrame or numpy array)
            dataY: Target data (can be pandas Series or numpy array)
        """
        # Convert to numpy arrays if needed
        if isinstance(dataX, pd.DataFrame):
            dataX = dataX.values
        if isinstance(dataY, pd.Series):
            dataY = dataY.values

        if self.verbose:
            print(f"\n=== Training DTLearner ===")
            print(f"Data shape: {dataX.shape}")
            print(f"Target shape: {dataY.shape}")

        # Build the tree
        self.tree = self.build_tree(dataX, dataY, depth=0)

        if self.verbose:
            print(f"\n=== Training Complete ===")

    # ========================================================================
    # PREDICTION
    # ========================================================================

    def query_point(self, point, tree):
        """
        Traverse tree to make prediction for a single point.

        Args:
            point: Single data point (1D numpy array)
            tree: Current tree/subtree to traverse

        Returns:
            Predicted value
        """
        # TODO: If tree is a leaf (not a list, just a number), return it
        if self.verbose and not isinstance(tree, list):
            print(f"    TODO: Reached leaf node, should return value")
        # Hint: if not isinstance(tree, list):
        #           return tree

        # TODO: Extract node components
        if self.verbose and isinstance(tree, list):
            print(f"    TODO: Extract feature, split_val, left_tree, right_tree from tree")
        # Hint: feature = tree[0]
        # Hint: split_val = tree[1]
        # Hint: left_tree = tree[2]
        # Hint: right_tree = tree[3]
        feature = 0  # Placeholder
        split_val = 0.0  # Placeholder
        left_tree = None  # Placeholder
        right_tree = None  # Placeholder

        # TODO: Traverse left or right based on feature value
        if self.verbose:
            print(f"    TODO: Compare point[{feature}] with split_val {split_val:.4f} to decide direction")
        # Hint: if point[feature] <= split_val:
        #           return self.query_point(point, left_tree)
        # Hint: else:
        #           return self.query_point(point, right_tree)

        return 0.0  # Placeholder

    def query(self, points):
        """
        Make predictions by traversing the decision tree.

        Args:
            points: Test data (N x M numpy array or pandas DataFrame)

        Returns:
            predictions: Numpy array of predictions
        """
        # Convert to numpy if needed
        if isinstance(points, pd.DataFrame):
            points = points.values

        # TODO: Make prediction for each point
        if self.verbose:
            print(f"\nTODO: Query {len(points)} points through the tree")
            print("Hint: Use list comprehension with query_point()")
        # Hint: predictions = np.array([self.query_point(point, self.tree) for point in points])
        predictions = np.zeros(len(points))  # Placeholder

        return predictions