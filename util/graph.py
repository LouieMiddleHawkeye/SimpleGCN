import networkx as nx

from matplotlib import pyplot as plt


def football_graph():
    """
    This is the graph for the HEI 29 joint skeletal data
    """
    G = nx.Graph()

    G.add_node(1)  # I think this should be added by the edges below, but just in case

    edges = [
        (0, 1),
        (2, 1),
        (3, 2),
        (4, 3),
        (5, 1),
        (6, 5),
        (7, 6),
        (8, 1),
        (9, 8),
        (10, 9),
        (11, 10),
        (12, 8),
        (13, 12),
        (14, 13),
        (15, 0),
        (16, 0),
        (17, 15),
        (18, 16),
        (19, 21),
        (20, 19),
        (21, 14),
        (22, 24),
        (23, 22),
        (24, 11),
        (25, 7),
        (26, 25),
        (27, 4),
        (28, 27),
    ]

    G.add_edges_from(edges)

    return G


def draw_graph(G):
    """
    Draws a networkx graphs nodes and edges
    """
    plt.figure(figsize=(12, 12))
    plt.axis("off")
    nx.draw_networkx(
        G,
        pos=nx.spring_layout(G, seed=0),
        with_labels=True,
        node_size=800,
        width=0.8,
        edge_color="grey",
        font_size=14,
    )
    plt.show()
