import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
import requests

class GraphSAGEEdgeRegressor(nn.Module):
    def __init__(self, node_feat_dim, edge_feat_dim, hidden_dim=128, dropout_rate=0.3):
        super().__init__()

        # GraphSAGE layers
        self.sage1 = SAGEConv(node_feat_dim, hidden_dim)
        self.sage2 = SAGEConv(hidden_dim, hidden_dim)

        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        self.dropout = nn.Dropout(dropout_rate)

        # Encode edge features
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout_rate)
        )

        # MLP for final edge-level prediction
        self.mlp = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x, edge_index, edge_attr, edge_source, edge_target):
        # GraphSAGE encoding of nodes
        x = self.sage1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.sage2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)

        # Get source and destination node embeddings
        h_src = x[edge_source]
        h_dst = x[edge_target]

        # Encode edge features
        edge_feat_encoded = self.edge_encoder(edge_attr)

        # Concatenate and predict
        edge_input = torch.cat([h_src, h_dst, edge_feat_encoded], dim=1)
        return self.mlp(edge_input).squeeze(-1)
    
def get_average_temperature(G, api_key="4d56505373da5f07f5e24a671ee7b4e0"):
    """Get average temperature of region G using OpenWeatherMap API."""
    # Get center node
    nodes = G.nodes(data=True)
    latitudes = [data['y'] for _, data in nodes]
    longitudes = [data['x'] for _, data in nodes]
    center_lat = np.mean(latitudes)
    center_lon = np.mean(longitudes)

    url = (
        f"https://api.openweathermap.org/data/2.5/weather?"
        f"lat={center_lat}&lon={center_lon}&appid={api_key}&units=metric"
    )
    
    try:
        response = requests.get(url)
        data = response.json()
        if "main" in data and "temp" in data["main"]:
            return data["main"]["temp"]  # DegC
    except Exception as e:
        print(e)

def add_estimated_battery_and_temperature(G, model, x, node_id_to_idx, scaler):
    """Add temperature and estimated battery calculated from model (GNN) to each edge in the graph G."""
    temperature = get_average_temperature(G)
    print(f"Temperature: {temperature}")

    edge_sources = []
    edge_targets = []
    edge_attr_list = []

    for u, v, key, data in G.edges(keys=True, data=True):
        if u not in node_id_to_idx or v not in node_id_to_idx:
            continue

        G[u][v][key]['temperature'] = temperature
        edge_sources.append(node_id_to_idx[u])
        edge_targets.append(node_id_to_idx[v])

        edge_attr = [
            data.get('length', 0),
            data.get('slope', 0),
            data.get('speed', 40),
            temperature
        ]
        edge_attr_list.append(edge_attr)

    # Scale
    edge_attr_df = pd.DataFrame(edge_attr_list, columns=['length[m]', 'slope', 'speed[km/h]', 'temperature[DegC]'])
    edge_attr_array = scaler.transform(edge_attr_df)
    edge_attr_tensor = torch.tensor(edge_attr_array, dtype=torch.float)

    edge_sources_tensor = torch.tensor(edge_sources, dtype=torch.long)
    edge_targets_tensor = torch.tensor(edge_targets, dtype=torch.long)
    edge_index = torch.stack([edge_sources_tensor, edge_targets_tensor], dim=0)

    # Predict
    model.eval()
    with torch.no_grad():
        battery_pred = model(x, edge_index, edge_attr_tensor, edge_sources_tensor, edge_targets_tensor).cpu().numpy()

    # Add result to edge
    for i, (u_idx, v_idx) in enumerate(zip(edge_sources, edge_targets)):
        u = list(node_id_to_idx.keys())[u_idx]
        v = list(node_id_to_idx.keys())[v_idx]

        for key in G[u][v]:
            G[u][v][key]['battery'] = float(battery_pred[i])

    return G

def compute_total_weight(G, path, attr):
    total = 0
    slopes = []
    for u, v in zip(path[:-1], path[1:]):
        edge_data = G.get_edge_data(u, v)
        if edge_data:
            min_val = min((data.get(attr, float('inf')) for data in edge_data.values()))
            total += min_val if min_val != float('inf') else 0
            slope = min((data.get("slope", 0) for data in edge_data.values()))
            slopes.append(slope)
    return total, slopes
