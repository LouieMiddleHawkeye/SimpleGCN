import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from matplotlib import animation
from torch.nn import Linear
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import KarateClub
from torch_geometric.utils import to_networkx

plt.rcParams["animation.bitrate"] = 3000


class GCN(torch.nn.Module):
    def __init__(self, dataset):
        super().__init__()
        # NOTE: If you stack another GCN, you will aggregate feature vectors from the neigbours' neigbours
        self.gcn = GCNConv(dataset.num_features, 3)  # 3 dimensional hidden layer
        self.out = Linear(3, dataset.num_classes)

    def forward(self, x, edge_index):
        h = self.gcn(x, edge_index).relu()
        z = self.out(h)

        return h, z


# Calculate accuracy
def accuracy(pred_y, y):
    return (pred_y == y).sum() / len(y)


def train(model, optimizer, criterion):
    # Data for animations
    embeddings = []
    losses = []
    accuracies = []
    outputs = []

    # Training loop
    for epoch in range(201):
        # Clear gradients
        optimizer.zero_grad()

        # Forward pass
        h, z = model(data.x, data.edge_index)

        # Calculate loss function
        loss = criterion(z, data.y)

        # Calculate accuracy
        acc = accuracy(z.argmax(dim=1), data.y)

        # Compute gradients
        loss.backward()

        # Tune parameters
        optimizer.step()

        # Store data for animations
        embeddings.append(h)
        losses.append(loss)
        accuracies.append(acc)
        outputs.append(z.argmax(dim=1))

        # Print metrics every 10 epochs
        if epoch % 10 == 0:
            print(f"Epoch {epoch:>3} | Loss: {loss:.2f} | Acc: {acc*100:.2f}%")

    return embeddings, losses, accuracies, outputs


def animate_training(i, losses, accuracies, outputs):
    G = to_networkx(data, to_undirected=True)
    nx.draw_networkx(
        G,
        pos=nx.spring_layout(G, seed=0),
        with_labels=True,
        node_size=800,
        node_color=outputs[i],
        cmap="hsv",
        vmin=-2,
        vmax=3,
        width=0.8,
        edge_color="grey",
        font_size=14,
    )
    plt.title(f"Epoch {i} | Loss: {losses[i]:.2f} | Acc: {accuracies[i]*100:.2f}%", fontsize=18, pad=20)


def animate_embeddings(i, embeddings, ax):
    embed = embeddings[i].detach().cpu().numpy()
    ax.clear()
    ax.scatter(embed[:, 0], embed[:, 1], embed[:, 2], s=200, c=data.y, cmap="hsv", vmin=-2, vmax=3)
    plt.title(f"Epoch {i} | Loss: {losses[i]:.2f} | Acc: {accuracies[i]*100:.2f}%", fontsize=18, pad=40)


# Import dataset from PyTorch Geometric
dataset = KarateClub()

# Print information
print(dataset)
print("------------")
print(f"Number of graphs: {len(dataset)}")
print(f"Number of features: {dataset.num_features}")
print(f"Number of classes: {dataset.num_classes}")

data = dataset[0]  # Node feature matrix

G = to_networkx(data, to_undirected=True)
plt.figure(figsize=(12, 12))
plt.axis("off")
nx.draw_networkx(
    G,
    pos=nx.spring_layout(G, seed=0),
    with_labels=True,
    node_size=800,
    node_color=data.y,
    cmap="hsv",
    vmin=-2,
    vmax=3,
    width=0.8,
    edge_color="grey",
    font_size=14,
)
plt.show()

model = GCN(dataset)
print(model)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.02)

embeddings, losses, accuracies, outputs = train(model, optimizer, criterion)

fig = plt.figure(figsize=(12, 12))
plt.axis("off")

anim = animation.FuncAnimation(
    fig, animate_training, np.arange(0, 200, 10), interval=500, repeat=True, fargs=(losses, accuracies, outputs)
)
plt.show()

fig = plt.figure(figsize=(12, 12))
plt.axis("off")
ax = fig.add_subplot(projection="3d")
plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

anim = animation.FuncAnimation(
    fig, animate_embeddings, np.arange(0, 200, 10), interval=800, repeat=True, fargs=(embeddings, ax)
)
plt.show()
