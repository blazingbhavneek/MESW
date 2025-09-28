# --- Imports ---
import json
import os
from itertools import combinations

import numpy as np
import plotly.graph_objects as go

# --- Configuration Variables ---
WORKING_DIR = "./graph_rag_cache_small"
ENTITY_DEFINITIONS_FILE = os.path.join(WORKING_DIR, "entity_definitions.json")
ENTITY_CONTEXTS_FILE = os.path.join(WORKING_DIR, "entity_contexts.json")

# --- Visualization Parameters ---
EDGE_COLOR = "black"
EDGE_OPACITY = 0.3  # Transparency for edges
EDGE_WIDTH = 1.5  # Width for edges
NODE_SIZE = 15  # Increased size for nodes

# ---------------------------------------
# Data Loading Section
# ---------------------------------------


def load_graph_data():
    """Loads entity definitions and contexts from saved JSON files."""
    if not os.path.exists(ENTITY_DEFINITIONS_FILE) or not os.path.exists(
        ENTITY_CONTEXTS_FILE
    ):
        print(
            f"Error: Could not find {ENTITY_DEFINITIONS_FILE} or {ENTITY_CONTEXTS_FILE}. Run the nano_graphrag build process first."
        )
        return None, None

    with open(ENTITY_DEFINITIONS_FILE, "r", encoding="utf-8") as f:
        definitions = json.load(f)
    with open(ENTITY_CONTEXTS_FILE, "r", encoding="utf-8") as f:
        contexts = json.load(f)

    return definitions, contexts


# ---------------------------------------
# Graph Inference Section
# ---------------------------------------


def infer_edges_from_contexts(contexts_data):
    """Infers edges based on co-occurrence of entities within the same context."""
    edges = set()  # Use a set to avoid duplicate edges (e.g., A-B and B-A)
    for term, context_list in contexts_data.items():
        for context in context_list:
            # Find all terms present in this specific context string
            # This is a simple keyword search; refine based on your term extraction logic if needed
            connected_terms = [
                t
                for t in contexts_data.keys()
                if t != term and t.lower() in context.lower()
            ]
            for connected_term in connected_terms:
                # Create an edge tuple with terms sorted alphabetically to ensure consistency (A-B same as B-A)
                edge = tuple(sorted([term, connected_term]))
                edges.add(edge)
    return list(edges)


# ---------------------------------------
# 3D Visualization Section
# ---------------------------------------


def create_3d_visualization(nodes, edges):
    """Creates a 3D scatter plot using Plotly."""
    if not nodes:
        print("No nodes to visualize.")
        return

    # Assign simple coordinates for demonstration; ideally, use a 3D layout algorithm like Fruchterman-Reingold
    # For now, we'll assign arbitrary 3D positions or use a layout engine implicitly handled by plotly if possible.
    # Using a simple distribution for initial visualization.
    import numpy as np

    num_nodes = len(nodes)
    # Generate random or simple 3D coordinates
    # Using a spherical distribution for a more spread-out look
    indices = np.arange(0, num_nodes, dtype=float)
    z = (2 * indices / (num_nodes - 1)) - 1  # z from -1 to 1
    theta = np.arccos(z)  # Spherical coordinate theta
    phi = np.sqrt(num_nodes * np.pi) * theta  # Spherical coordinate phi

    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    # Create the trace for nodes
    node_trace = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="markers+text",
        marker=dict(
            size=NODE_SIZE,  # Use the parameter defined above
            color="lightblue",
            line=dict(width=2, color="black"),  # Add a border to make nodes stand out
        ),
        text=nodes,  # Node labels
        textposition="middle center",
        hoverinfo="text",
        name="Entities",
    )

    # Prepare coordinates for edges
    Xe, Ye, Ze = [], [], []
    for edge in edges:
        term1, term2 = edge
        if term1 in nodes and term2 in nodes:
            idx1 = nodes.index(term1)
            idx2 = nodes.index(term2)
            # Add coordinates for the edge, followed by None to separate segments
            Xe += [x[idx1], x[idx2], None]
            Ye += [y[idx1], y[idx2], None]
            Ze += [z[idx1], z[idx2], None]

    # Create the trace for edges
    edge_trace = go.Scatter3d(
        x=Xe,
        y=Ye,
        z=Ze,
        mode="lines",
        line=dict(
            color=EDGE_COLOR,  # Use the base color defined above
            width=EDGE_WIDTH,  # Use the parameter defined above
        ),
        hoverinfo="none",
        name="Relationships",
        opacity=EDGE_OPACITY,  # Apply opacity to the whole trace
    )

    # Create the figure
    fig = go.Figure(data=[edge_trace, node_trace])

    # Update layout for better visualization
    fig.update_layout(
        title=dict(
            text="3D Knowledge Graph Visualization (Co-occurrence in Contexts)",
            font=dict(size=16),  # Use 'font.size' instead of 'titlefont_size'
        ),
        showlegend=True,
        scene=dict(
            xaxis=dict(showticklabels=False, showspikes=False, title_text=""),
            yaxis=dict(showticklabels=False, showspikes=False, title_text=""),
            zaxis=dict(showticklabels=False, showspikes=False, title_text=""),
            bgcolor="white",  # Set background color
        ),
        margin=dict(t=50, l=50, r=50, b=50),  # Adjust margins
        paper_bgcolor="rgba(0,0,0,0)",  # Transparent background outside the plot area
        plot_bgcolor="rgba(0,0,0,0)",  # Transparent background inside the plot area
    )

    # Show the plot
    fig.show()


# ---------------------------------------
# Main Execution Section
# ---------------------------------------


def main():
    """Main function to load data, infer edges, and create visualization."""
    definitions, contexts = load_graph_data()
    if definitions is None or contexts is None:
        return

    print(f"Loaded {len(definitions)} entities.")
    print(f"Loaded contexts for {len(contexts)} entities.")

    nodes = list(definitions.keys())
    edges = infer_edges_from_contexts(contexts)

    print(f"Inferring {len(edges)} edges based on co-occurrence in contexts.")
    create_3d_visualization(nodes, edges)


if __name__ == "__main__":
    main()
