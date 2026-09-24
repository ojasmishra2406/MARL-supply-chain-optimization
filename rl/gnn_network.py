import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleGCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        
    def forward(self, x, adj):
        # x: (B, N, F)
        # adj: (N, N) or (B, N, N)
        h = self.linear(x) # (B, N, out_features)
        # Message passing: out = adj @ h
        if adj.dim() == 2:
            out = torch.matmul(adj, h) # (B, N, N) x (B, N, out_features)? No, adj is (N, N)
            out = torch.einsum('vw,bwc->bvc', adj, h)
        else:
            out = torch.bmm(adj, h)
        return out

class SupplyChainGNNEncoder(nn.Module):
    def __init__(self, node_features, hidden_dim):
        super().__init__()
        # Standard line graph adjacency with self-loops
        # 0: Retailer, 1: Wholesaler, 2: Distributor, 3: Factory
        self.register_buffer('adj', torch.tensor([
            [1., 1., 0., 0.],
            [1., 1., 1., 0.],
            [0., 1., 1., 1.],
            [0., 0., 1., 1.]
        ]))
        
        # Normalize adjacency (D^-0.5 A D^-0.5)
        # For simplicity, we just divide by degree
        deg = self.adj.sum(dim=-1, keepdim=True)
        self.register_buffer('norm_adj', self.adj / deg)
        
        self.gcn1 = SimpleGCNLayer(node_features, hidden_dim)
        self.gcn2 = SimpleGCNLayer(hidden_dim, hidden_dim)
        
    def forward(self, x):
        # x expected shape: (B, N, F)
        h = self.gcn1(x, self.norm_adj)
        h = F.relu(h)
        h = self.gcn2(h, self.norm_adj)
        h = F.relu(h)
        return h

class GNN_MAPPO_Critic(nn.Module):
    def __init__(self, num_nodes, node_features, hidden_dim=64):
        super().__init__()
        self.num_nodes = num_nodes
        self.node_features = node_features
        
        self.gnn = SupplyChainGNNEncoder(node_features, hidden_dim)
        
        # Global aggregation: flatten all node embeddings
        self.fc1 = nn.Linear(num_nodes * hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        
    def forward(self, global_obs):
        # global_obs shape: (B, num_nodes * node_features)
        B = global_obs.shape[0]
        x = global_obs.reshape(B, self.num_nodes, self.node_features)
        
        node_embeds = self.gnn(x) # (B, N, hidden_dim)
        global_embed = node_embeds.reshape(B, -1) # (B, N*hidden_dim)
        
        h = F.relu(self.fc1(global_embed))
        v = self.fc2(h)
        return v
        
class GNN_MAPPO_Actor(nn.Module):
    def __init__(self, num_nodes, node_features, action_dim, hidden_dim=64, agent_index=0):
        super().__init__()
        self.num_nodes = num_nodes
        self.node_features = node_features
        self.agent_index = agent_index
        
        # CTDE principle: actor doesn't see the global state, only local state
        # But if we use GNN for actor, it implies the actor receives the local graph neighborhood.
        # Since it's decentralized execution, we just use a standard MLP for the actor,
        # OR a GNN over the local observation if it was graph-structured.
        # In our env, local obs is just a single node (5 features). 
        # A GNN on a single node is just an MLP. 
        # So we keep the standard MLP actor to strictly preserve CTDE!
        
        self.fc1 = nn.Linear(node_features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.action_head = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, local_obs):
        h = F.relu(self.fc1(local_obs))
        h = F.relu(self.fc2(h))
        logits = self.action_head(h)
        return logits
