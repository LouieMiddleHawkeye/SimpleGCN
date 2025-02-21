import torch
import torch.nn as nn
import torch_geometric.nn as geom_nn


class SkeletonGCN(nn.Module):
    def __init__(self, num_joints, num_classes, hidden_channels=64):
        super(SkeletonGCN, self).__init__()

        # GCN layers for each frame
        self.gcn1 = geom_nn.GCNConv(3, hidden_channels)  # 3 input features (x,y,z)
        self.gcn2 = geom_nn.GCNConv(hidden_channels, hidden_channels)

        # CNN layers for temporal modeling
        self.conv1 = nn.Conv1d(num_joints * hidden_channels, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(128, 256, kernel_size=3, padding=1)

        # Final classification layers
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(256, num_classes)

        # Activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, edge_index, batch_size, num_frames):
        """
        :param x shape: (batch_size * num_frames * num_joints, 3)
        :param edge_index: skeleton connectivity

        Input Data
        (batch_size * num_frames * num_joints, 3)
                ↓
        Process each frame through GCN
        (num_joints, hidden_channels) for each frame
                ↓
        Stack and reshape frames
        (batch_size, num_frames, num_joints * hidden_channels)
                ↓
        Permute for CNN
        (batch_size, num_joints * hidden_channels, num_frames)
                ↓
        Apply CNN layers
        (batch_size, 256, num_frames)
                ↓
        Global pooling
        (batch_size, 256)
                ↓
        Final classification
        (batch_size, num_targets)
        """
        batch_frames = []  # Will store processed frames
        num_joints = x.size(0) // (batch_size * num_frames)  # Calculate joints per frame

        # Process each frame through GCN
        for i in range(batch_size * num_frames):
            start_idx = i * num_joints
            end_idx = (i + 1) * num_joints

            # Get current frame joints
            frame_joints = x[start_idx:end_idx]

            # Apply GCN layers
            out = self.relu(self.gcn1(frame_joints, edge_index))
            out = self.relu(self.gcn2(out, edge_index))

            batch_frames.append(out)

        # Stack processed frames
        out = torch.stack(batch_frames)
        # Reshape to seperate batch and frames
        out = out.view(batch_size, num_frames, -1)  # (batch_size, num_frames, num_joints * hidden_channels)

        # Prepare for CNN (swap dimensions for Conv1d)
        out = out.permute(0, 2, 1)  # (batch_size, num_joints * hidden_channels, num_frames)

        # Apply CNN layers for temporal analysis
        out = self.relu(self.conv1(out))
        out = self.relu(self.conv2(out))

        # Global pooling and classification to aggregate temporal information
        out = self.global_pool(out).squeeze(-1)
        out = self.sigmoid(self.fc(out))  # Sigmoid for multi label

        return out
