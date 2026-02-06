import torch
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx

# Load the graph
graph = torch.load("data/graph/graph_A.pt", map_location="cpu")

# Print graph info
print("Graph type:", type(graph))
print(graph)

# Convert PyTorch Geometric graph to NetworkX
G = to_networkx(graph, to_undirected=True)

# Plot the graph
plt.figure(figsize=(8, 8))
plt.axis("off")

nx.draw(
    G,
    node_size=300,
    node_color="skyblue",
    edge_color="gray",
    with_labels=True
)

plt.show()
