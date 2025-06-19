import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool, GCNConv
from torch_geometric.data import Data


class GCN(torch.nn.Module):
    def __init__(self, num_features, hidden_dim, num_classes=2):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = torch.nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index, batch):
        # Graph convolution layers
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))

        # Pooling to graph-level embedding
        x = global_mean_pool(x, batch)

        # Classification layer
        x = self.fc(x)
        return F.log_softmax(x, dim=-1)


class SimpleEGNNLayer(MessagePassing):
    def __init__(self, node_feat_dim, edge_feat_dim, hidden_dim):
        super(SimpleEGNNLayer, self).__init__(aggr='mean')  
        self.edge_mlp = torch.nn.Sequential(
            torch.nn.Linear(2 * node_feat_dim + edge_feat_dim + 1, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim)
        )
        self.node_mlp = torch.nn.Sequential(
            torch.nn.Linear(node_feat_dim + hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, node_feat_dim)
        )

    def forward(self, x, edge_index, edge_attr, pos):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr, pos=pos)

    def message(self, x_i, x_j, edge_attr, pos_i, pos_j):
        # Distance between nodes
        dist = torch.norm(pos_i - pos_j, dim=-1, keepdim=True)
        # Combine features and distances
        msg_input = torch.cat([x_i, x_j, edge_attr, dist], dim=-1)
        msg = self.edge_mlp(msg_input)
        return msg

    def update(self, aggr_out, x):
        # Update node embeddings
        node_input = torch.cat([x, aggr_out], dim=-1)
        return self.node_mlp(node_input)


class EGNN(torch.nn.Module):
    def __init__(self, num_features, edge_feat_dim, hidden_dim, num_classes=2,dropout=0):
        super(EGNN, self).__init__()
        self.egnn1 = SimpleEGNNLayer(num_features, edge_feat_dim, hidden_dim)
        self.egnn2 = SimpleEGNNLayer(num_features, edge_feat_dim, hidden_dim)
        self.fc = torch.nn.Linear(num_features, num_classes)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr, pos, batch):
        # EGNN layers updating node embeddings (positions could be updated similarly)
        x = self.egnn1(x, edge_index, edge_attr, pos)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.egnn2(x, edge_index, edge_attr, pos)
        x = F.relu(x)
        x = self.dropout(x)


        # Pooling to graph-level embedding
        x = global_mean_pool(x, batch)

        # Classification layer
        x = self.fc(x)
        return F.log_softmax(x, dim=-1)


