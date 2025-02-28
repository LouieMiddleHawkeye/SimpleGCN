import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from model.skeleton_GCN import SkeletonGCN
from util.graph import football_graph, draw_graph
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.utils import from_networkx
from tqdm import tqdm


def train_model(
    model,
    train_data,
    train_labels,
    val_data,
    val_labels,
    edge_index,
    batch_size=32,
    num_epochs=100,
    learning_rate=0.001,
):
    """
    Train the skeletal GCN model
    """
    # Convert data to PyTorch tensors
    train_data = torch.FloatTensor(train_data)
    train_labels = torch.FloatTensor(train_labels)
    val_data = torch.FloatTensor(val_data)
    val_labels = torch.FloatTensor(val_labels)

    # Create data loaders
    train_dataset = TensorDataset(train_data, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)

    val_dataset = TensorDataset(val_data, val_labels)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)

    # Define loss function and optimizer
    criterion = nn.BCELoss()  # TODO: Use label smoothing loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)

    # Initialize tracking variables
    best_val_loss = float("inf")
    best_model_state = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    edge_index = edge_index.to(device)

    print(f"Training on {device}")
    print(f"Total epochs: {num_epochs}")
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print("-" * 50)

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_losses = []
        train_acc = []

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for batch_data, batch_labels in train_bar:
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)

            x, _, batch_size, num_frames = prepare_data(batch_data, edge_index)
            x = x.to(device)

            optimizer.zero_grad()
            outputs = model(x, edge_index, batch_size, num_frames)

            loss = criterion(outputs, batch_labels)

            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            acc = accuracy(outputs, batch_labels)
            train_acc.append(acc.cpu())

            train_bar.set_postfix({"loss": f"{loss.item():.4f}", "accuracy": f"{acc}"})

        # Validation phase
        model.eval()
        val_losses = []
        val_acc = []
        all_outputs = []
        all_labels = []

        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
        with torch.no_grad():
            for batch_data, batch_labels in val_bar:
                batch_data = batch_data.to(device)
                batch_labels = batch_labels.to(device)

                x, _, batch_size, num_frames = prepare_data(batch_data, edge_index)
                x = x.to(device)

                outputs = model(x, edge_index, batch_size, num_frames)
                loss = criterion(outputs, batch_labels)

                val_losses.append(loss.item())
                acc = accuracy(outputs, batch_labels)
                val_acc.append(acc.cpu())
                all_outputs.append(outputs.cpu())
                all_labels.append(batch_labels.cpu())

                val_bar.set_postfix({"loss": f"{loss.item():.4f}", "accuracy": f"{acc}"})

        # Calculate metrics
        train_loss = np.mean(train_losses)
        train_acc = np.mean(train_acc)
        val_loss = np.mean(val_losses)
        val_acc = np.mean(val_acc)

        # Update learning rate
        scheduler.step(val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()

        # Print progress
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print("--------------------")

    # Load best model
    model.load_state_dict(best_model_state)
    return model


def accuracy(output, target):
    """
    Calculate accuracy of output against target
    """
    batch_size = target.size(0)

    # Convert target to class indice
    target = target.argmax(1)

    # Get the predicted class (highest probability)
    _, pred = output.max(1)

    # Compare predictions with targets
    correct = pred.eq(target).float()

    # Calculate accuracy
    acc = correct.sum() * 100.0 / batch_size

    return acc


def prepare_data(data, edge_index):
    """
    Prepare data for the SkeletonGCN model
    :param data: in shape (batch_size, num_frames, num_joints * 3)
    :return x, edge_index, batch_size, num_frames where x is shape (batch_size * num_frames * num_joints, 3)
    """
    batch_size = data.size(0)
    num_frames = data.size(1)

    # Reshape data to (batch_size * num_frames * num_joints, 3)
    x = data.view(-1, 3)

    return x, edge_index, batch_size, num_frames


def get_data():
    with h5py.File("./data/SGN_football.h5", "r") as f:
        train_x = f["x"][:]
        train_y = f["y"][:]
        val_x = f["valid_x"][:]
        val_y = f["valid_y"][:]

    return train_x, train_y, val_x, val_y


if __name__ == "__main__":
    print("Simple GCN")

    G = football_graph()
    draw_graph(G)

    G = from_networkx(G)

    num_joints = 29
    num_classes = 2
    model = SkeletonGCN(num_joints, num_classes)

    train_x, train_y, val_x, val_y = get_data()

    # Train the model
    trained_model = train_model(
        model=model,
        train_data=train_x,
        train_labels=train_y,
        val_data=val_x,
        val_labels=val_y,
        edge_index=G.edge_index,
        batch_size=64,
        num_epochs=100,
        learning_rate=0.001,
    )

    # Save the trained model
    torch.save(trained_model.state_dict(), "skeleton_gcn_model.pth")
