import torch
import matplotlib.pyplot as plt
import numpy as np

def graph_dif (x,y1,y2):
    # Ensure x, y1, and y2 are tensors
    if x.shape[0] != y1.shape[0] or x.shape[0] != y2.shape[0]:
        raise ValueError("The x, y1, and y2 tensors must have the same length.")

    # Convert tensors to numpy for plotting
    if not isinstance(x, np.ndarray):
        x_np = x.numpy()
    else:
        x_np = x
    if not isinstance(y1, np.ndarray):
        y1_np = y1.numpy()
    else:
        y1_np = y1
    if not isinstance(y2, np.ndarray):
        y2_np = y2.numpy()
    else:
        y2_np = y2

    # Normalize both x and y1 to range [0, 1] for color mapping
    x_norm = (x_np - np.min(x_np)) / (np.max(x_np) - np.min(x_np))
    y1_norm = (y1_np - np.min(y1_np)) / (np.max(y1_np) - np.min(y1_np))

    # Combine normalized x and y1 to create RGB colors (x -> R, y1 -> G, constant -> B)
    colors_y1 = np.column_stack((x_norm, y1_norm, np.full_like(x_norm, 0.5)))  # x -> R, y1 -> G, 0.5 -> B

    # Use the same colors for y2 (corresponding index should have same color)
    colors_y2 = colors_y1  # Ensure y2 has the same colors as y1

    # Plot the graphs
    plt.figure(figsize=(10, 6))

    # Plot the first graph (quadratic curve) with RGB colors based on (x, y1)
    plt.scatter(x_np, y1_np, color=colors_y1, label='Graph y1 (Quadratic)', marker='o', s=30)

    # Plot the second graph (sine wave) with the same RGB colors (index matched with y1)
    plt.scatter(x_np, y2_np, color=colors_y2, label='Graph y2 (Sine)', marker='s', s=30)

    # Add labels and title
    plt.title('Graph Visualization with Synchronized Colors for y1 and y2')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()
    plt.grid()
