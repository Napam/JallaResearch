from turtle import fillcolor
import networkx as nx
import matplotlib.pyplot as plt


G = nx.DiGraph()
G.add_nodes_from([
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J"
])
G.add_edges_from([
    ["A", "B"],
    ["B", "C"], 
    ["B", "E"],
    ["D", "E"],
    ["E", "C"],
    ["F", "D"],
    ["H", "I"],
    ["I", "J"],
    ["I", "D"],
    ["J", "H"]
])
options = {
    "font_size": 20,
    "node_size": 1000,
    "node_color": "white",
    "edgecolors": "black",
    "linewidths": 2,
    "width": 2,
    "with_labels": True,
}
nx.draw(G, pos=nx.kamada_kawai_layout(G), **options)
plt.show()