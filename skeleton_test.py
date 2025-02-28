import numpy as np
import torch
import torch.nn as nn
import torch_geometric.nn as geom_nn

from IPython import display
from main import accuracy, get_data, prepare_data
from matplotlib import animation
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.utils import from_networkx
from tqdm import tqdm
from util.graph import football_graph, draw_graph

plt.rcParams["animation.bitrate"] = 3000


class JustGCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.gcn1 = geom_nn.GCNConv(3, 3)

    def forward(self, x, edge_index):
        """
        This will be:
         - aggregating neighbour joint information
         - average neigbour features
         - apply a linear transformation
        """
        return self.gcn1(x, edge_index)


class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # NOTE: If you stack another GCN, you will aggregate feature vectors from the neigbours' neigbours
        self.gcn1 = geom_nn.GCNConv(3, 3)  # 3 dimensional hidden layer
        self.fc = nn.Linear(29 * 3, 2)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, edge_index, batch_size, num_frames):
        batch_frames = []  # Will store processed frames
        num_joints = x.size(0) // (batch_size * num_frames)  # Calculate joints per frame

        # Process each frame through GCN
        for i in range(batch_size * num_frames):
            start_idx = i * num_joints
            end_idx = (i + 1) * num_joints

            # Get current frame joints
            frame_joints = x[start_idx:end_idx]

            # Apply GCN layers
            h = self.relu(self.gcn1(frame_joints, edge_index))
            h = h.reshape(-1)  # Flatten to num_joints * 3

            batch_frames.append(h)

        h = torch.stack(batch_frames)
        # Reshape to (batch_size, num_frames, num_joints * hidden_dim)
        h = h.view(batch_size, num_frames, -1)

        # Mean Pooling
        h = torch.mean(h, dim=1)

        z = self.sigmoid(self.fc(h))  # Sigmoid for multi label

        return h, z


def one_forward_example():
    G = football_graph()
    draw_graph(G)

    G = from_networkx(G)

    # Example joint positions
    joint_positions = torch.tensor(
        [
            [0.09070301, 0.39912415, 0.59176034],
            [0.03346539, 0.20183754, 0.5116586],
            [0.20056057, 0.12314224, 0.4994802],
            [0.25632477, -0.01768875, 0.27096063],
            [0.28216076, 0.16908836, 0.16734141],
            [-0.14063835, 0.22788239, 0.45969182],
            [-0.22589493, 0.17983437, 0.20406264],
            [-0.17438126, 0.23991203, 0.00399679],
            [0.0, 0.0, 0.0],
            [0.11115742, -0.00127411, -0.02151579],
            [0.12312698, 0.10539246, -0.37464115],
            [0.07903004, -0.06557465, -0.7054743],
            [-0.11172009, 0.01436043, -0.00762475],
            [-0.1947174, 0.00365829, -0.370248],
            [-0.2617426, -0.13913155, -0.71045864],
            [0.12191582, 0.37132263, 0.6393663],
            [0.05795956, 0.39097786, 0.6417616],
            [0.14697742, 0.26159477, 0.6825101],
            [-0.02533627, 0.3130417, 0.6866521],
            [-0.32193565, 0.03752136, -0.81002957],
            [-0.37321854, -0.01660728, -0.7957563],
            [-0.26253128, -0.19042015, -0.78465384],
            [0.11768627, 0.10843086, -0.8193653],
            [0.17251492, 0.05667305, -0.81049865],
            [0.07127857, -0.11999893, -0.7769864],
            [-0.10579109, 0.25403595, -0.05230653],
            [-0.18121529, 0.24754524, -0.11497319],
            [0.2780838, 0.2581749, 0.17834526],
            [0.3377781, 0.2510147, 0.10060585],
        ]
    )

    model = JustGCN()
    output = model(joint_positions, G.edge_index)
    print(output)


def train(x, y, edge_index, model, optimizer, criterion):
    x = torch.FloatTensor(x)
    y = torch.FloatTensor(y)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    edge_index = edge_index.to(device)

    train_dataset = TensorDataset(x, y)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)

    # Data for animations
    embeddings = []
    losses = []
    accuracies = []
    outputs = []

    # Training loop
    for epoch in range(10):
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{10} [Train]")
        for batch_data, batch_labels in train_bar:
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)

            x, _, batch_size, num_frames = prepare_data(batch_data, edge_index)
            x = x.to(device)

            optimizer.zero_grad()

            h, z = model(x, edge_index, batch_size, num_frames)

            loss = criterion(z, batch_labels)

            acc = accuracy(z, batch_labels)

            loss.backward()

            optimizer.step()

            embeddings.append(h)
            losses.append(loss)
            accuracies.append(acc)
            outputs.append(z.argmax(dim=1))

    if epoch % 2 == 0:
        print(f"Epoch {epoch:>3} | Loss: {loss:.2f} | Acc: {acc*100:.2f}%")

    return embeddings, losses, accuracies, outputs


def animate_embeddings(i, embeddings, losses, accuracies, ax):
    embed = embeddings[i].detach().cpu().numpy()
    ax.clear()
    ax.scatter(embed[:, 0], embed[:, 1], embed[:, 2], s=200)
    plt.title(f"Epoch {i} | Loss: {losses[i]:.2f} | Acc: {accuracies[i]*100:.2f}%", fontsize=18, pad=40)


def learn_embeddings():
    G = football_graph()
    draw_graph(G)

    G = from_networkx(G)

    model = GCN()
    print(model)

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)

    train_x, train_y, _, _ = get_data()

    embeddings, losses, accuracies, _ = train(train_x, train_y, G.edge_index, model, optimizer, criterion)

    fig = plt.figure(figsize=(12, 12))
    plt.axis("off")
    ax = fig.add_subplot(projection="3d")
    plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    anim = animation.FuncAnimation(
        fig,
        animate_embeddings,
        np.arange(0, 10, 2),
        interval=800,
        repeat=True,
        fargs=(embeddings, losses, accuracies, ax),
    )
    anim.save("embeddings.mp4", fps=30)
    plt.show()


if __name__ == "__main__":
    # one_forward_example()

    learn_embeddings()
