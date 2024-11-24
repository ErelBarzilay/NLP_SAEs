from graph_diff_utils import graph_dif

import torch
import matplotlib.pyplot as plt

# Create some sample data for the graphs
# You can replace these with your actual tensors
x = torch.tensor([0, 1, 2, 3, 4, 5])  # x-coordinates
y1 = torch.tensor([0, 1, 4, 9, 16, 25])  # y-coordinates for the first graph
y2 = torch.tensor([0, 1, 2, 3, 4, 5])    # y-coordinates for the second graph

graph_dif(x, y1, y2, "graph_visualization.png")
plt.savefig('graph_visualization.png', dpi=300, bbox_inches='tight')  # Save as PNG with 300 DPI