# SimpleGCN

## Example demo

- `example.py` is a demo of a basic GCN with one GCN layer connected to a linear output layer.
- The hidden GCN layer output is 3 dimensional such that the embeddings during training can be visualised.
- This example is a _node classification_ task, training a GCN to predict which group each node belongs to.
- The training loop can be visualised to see the network being trained.

## SkeletonGCN

- This pre-defines a graph of the human skeleton (unlike SGN which learns a content adaptive graph).
- For each joint, the GCN:
    - Looks at its own position
    - Looks at connected joints' positions
    - Learns patterns in joint relationships
