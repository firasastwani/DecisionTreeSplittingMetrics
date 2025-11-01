"""
Tree Visualization Utilities

Provides both text-based and Graphviz-based visualization
for decision trees.

Emulating scikit-learn's visualization functions.

(c) 2025 Hybinette -- Tree Visualization Utilities
"""

def visualize_tree_text(tree, feature_names=None, indent="\t", depth=0, max_depth=10):
    """
    Visualize the decision tree structure in text format.

    Args:
        tree: The tree structure (list or float)
        feature_names: List of feature names for display
        indent: Current indentation string
        depth: Current depth in tree
        max_depth: Maximum depth to display
    """
    # another_indent = "   "
    another_indent = "\t"

    if depth > max_depth:
        print(f"{indent}... (tree continues)")
        return

    # If it's a leaf node (just a number)
    if not isinstance(tree, list):
        print(f"{indent}Leaf: predict {tree:.4f}")
        return

    # Otherwise it's an internal node [feature, split_val, left, right]
    feature_idx, split_val, left_tree, right_tree = tree

    # Get a feature name if available
    if feature_names is not None and feature_idx < len(feature_names):
        feature_name = feature_names[feature_idx]
    else:
        feature_name = f"Feature {feature_idx}"

    print(f"{indent}if {feature_name} <= {split_val:.4f}:")
    visualize_tree_text(left_tree, feature_names, indent + another_indent, depth + 1, max_depth)

    print(f"{indent}else ({feature_name} > {split_val:.4f}):")
    visualize_tree_text(right_tree, feature_names, indent + another_indent, depth + 1, max_depth)


def visualize_tree_graphviz(tree, feature_names=None, filename='decision_tree',
                            view=True, format='pdf'):
    """
    Visualize the decision tree using Graphviz.

    Args:
        tree: The tree structure (list or float)
        feature_names: List of feature names for display
        filename: Output filename (without extension)
        view: If True, automatically open the visualization
        format: Output format ('pdf', 'png', 'svg', etc.)

    Returns:
        graphviz.Digraph object (or None if graphviz not available)
    """
    try:
        from graphviz import Digraph
    except ImportError:
        print("\n" + "=" * 70)
        print("ERROR: graphviz package not installed")
        print("=" * 70)
        print("\nTo use Graphviz visualization, you need to:")
        print("\n1. Install Python package:")
        print("   pip install graphviz")
        print("\n2. Install Graphviz system package:")
        print("   - Ubuntu/Debian: sudo apt-get install graphviz")
        print("   - macOS: brew install graphviz")
        print("   - Windows: Download from https://graphviz.org/download/")
        print("\n3. After installation, restart your terminal/IDE")
        print("\nFalling back to text visualization...")
        print("=" * 70 + "\n")
        return None

    dot = Digraph(comment='Decision Tree')
    dot.attr(rankdir='TB')  # Top to Bottom layout

    # Set graph aesthetics
    dot.attr('node', shape='box', style='rounded,filled', fontname='Arial', fontsize='10')
    dot.attr('edge', fontname='Arial', fontsize='9')

    node_counter = [0]  # Use list to allow modification in nested function

    def add_nodes(subtree, parent_id=None, edge_label=""):
        """Recursively add nodes to the graph"""
        current_id = node_counter[0]
        node_counter[0] += 1

        # If it's a leaf node
        if not isinstance(subtree, list):
            label = f"Predict\n{subtree:.4f}"
            dot.node(str(current_id), label, fillcolor='lightblue', shape='ellipse')

            if parent_id is not None:
                dot.edge(str(parent_id), str(current_id), label=edge_label)

            return current_id

        # Otherwise it's an internal node [feature, split_val, left, right]
        feature_idx, split_val, left_tree, right_tree = subtree

        # Get feature name
        if feature_names is not None and feature_idx < len(feature_names):
            feature_name = feature_names[feature_idx]
        else:
            feature_name = f"X{feature_idx}"

        # Create node label
        label = f"{feature_name}\n<= {split_val:.4f}"
        dot.node(str(current_id), label, fillcolor='lightgreen')

        # Connect to parent if exists
        if parent_id is not None:
            dot.edge(str(parent_id), str(current_id), label=edge_label)

        # Add left and right subtrees
        add_nodes(left_tree, current_id, "Yes")
        add_nodes(right_tree, current_id, "No")

        return current_id

    # Build the graph
    add_nodes(tree)

    # Render the graph
    try:
        output_file = dot.render(filename, format=format, cleanup=True, view=view)
        print(f"✓ Tree visualization saved as '{output_file}'")

        # Also save as PNG if format is PDF (for easy viewing)
        if format == 'pdf':
            try:
                png_file = dot.render(filename, format='png', cleanup=True, view=False)
                print(f"✓ Also saved as '{png_file}'")
            except:
                pass

    except Exception as e:
        print(f"\n" + "=" * 70)
        print(f"ERROR: Could not render graph")
        print("=" * 70)
        print(f"\nError details: {e}")
        print("\nPossible issues:")
        print("1. Graphviz system package not installed")
        print("2. Graphviz not in system PATH")
        print("3. Permission issues with output directory")
        print("\nFalling back to text visualization...")
        print("=" * 70 + "\n")
        return None

    return dot


def visualize_tree(tree, feature_names=None, method='text', filename='decision_tree',
                   view=True, max_depth=10, format='pdf'):
    """
    Visualize the decision tree using the specified method.

    Args:
        tree: The tree structure (list or float)
        feature_names: List of feature names for display
        method: 'text' for text-based or 'graphviz' for graph visualization
        filename: Output filename for graphviz (without extension)
        view: If True, automatically open graphviz visualization
        max_depth: Maximum depth to display for text visualization
        format: Output format for graphviz ('pdf', 'png', 'svg', etc.)

    Returns:
        Graphviz object if method='graphviz', None otherwise
    """
    if method == 'text':
        visualize_tree_text(tree, feature_names, max_depth=max_depth)
        return None
    elif method == 'graphviz':
        graph = visualize_tree_graphviz(tree, feature_names, filename, view, format)
        # If graphviz failed, fall back to text
        if graph is None:
            print("\nShowing text visualization instead:\n")
            visualize_tree_text(tree, feature_names, max_depth=max_depth)
        return graph
    else:
        print(f"Unknown visualization method: '{method}'")
        print("Valid options: 'text' or 'graphviz'")
        print("Using text visualization instead...\n")
        visualize_tree_text(tree, feature_names, max_depth=max_depth)
        return None


def check_graphviz_available():
    """
    Check if Graphviz is available on the system.

    Returns:
        bool: True if graphviz is available, False otherwise
    """
    try:
        from graphviz import Digraph
        # Try to create a simple graph to verify system installation
        test_dot = Digraph()
        test_dot.node('test', 'Test')
        # This will fail if system graphviz is not installed
        test_dot.pipe(format='png')
        return True
    except ImportError:
        return False
    except Exception:
        return False


if __name__ == "__main__":
    # Demo the visualization functions
    print("=" * 70)
    print("Tree Visualization Utilities - Demo")
    print("=" * 70)

    # Create a simple demo tree
    demo_tree = [
        0,  # Split on feature 0
        5.5,  # Split value
        [  # Left subtree
            1,  # Split on feature 1
            3.0,
            10.5,  # Left leaf
            15.2  # Right leaf
        ],
        20.0  # Right leaf
    ]

    feature_names = ['Temperature', 'Humidity', 'Windspeed']

    # Test text visualization
    print("\n1. Text Visualization:")
    print("-" * 70)
    visualize_tree(demo_tree, feature_names, method='text')

    # Check if graphviz is available
    print("\n\n2. Checking Graphviz availability:")
    print("-" * 70)
    if check_graphviz_available():
        print("✓ Graphviz is available!")
        print("\nCreating Graphviz visualization...")
        visualize_tree(demo_tree, feature_names, method='graphviz',
                       filename='demo_tree', view=True)
    else:
        print("✗ Graphviz is not available")
        print("Install it to use graph visualizations")

    print("\n" + "=" * 70)
    print("Demo complete!")
    print("=" * 70)