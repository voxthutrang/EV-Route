from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
import networkx as nx
import pickle
import torch
import pandas as pd
import numpy as np

from services.graph_utils import compute_total_weight, add_estimated_battery_and_temperature, GraphSAGEEdgeRegressor
from services.map_generator import create_map

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")

# Load data
with open("data/road_graph.gpickle", "rb") as f:
    G = pickle.load(f)

with open("data/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Prepare node features
for node in G.nodes:
    G.nodes[node]['degree'] = G.degree(node)
    G.nodes[node]['elevation'] = G.nodes[node].get('elevation', 0)
    neighbor_slopes = [data.get('slope', 0) for _, _, data in G.edges(node, data=True)]
    G.nodes[node]['avg_neighbor_slope'] = np.mean(neighbor_slopes) if neighbor_slopes else 0.0

node_features_df = pd.DataFrame([
    {
        'node_id': node,
        'label': f"Node {node}",
        'degree': G.nodes[node]['degree'],
        'elevation': G.nodes[node]['elevation'],
        'avg_neighbor_slope': G.nodes[node]['avg_neighbor_slope']
    } for node in G.nodes
])

node_id_to_idx = {nid: i for i, nid in enumerate(node_features_df['node_id'])}
x = torch.tensor(node_features_df[['degree', 'elevation', 'avg_neighbor_slope']].values, dtype=torch.float)

model = GraphSAGEEdgeRegressor(node_feat_dim=x.shape[1], edge_feat_dim=4)
model.load_state_dict(torch.load('data/best_model.pt', map_location=torch.device('cpu')))
model.eval()

df = pd.read_csv("data/ev_energy_consumption.csv")
df.columns = ["vehicle_type", "consumption_wh_per_km"]

base_consumption = 24

@app.get("/vehicles")
def get_vehicle_types():
    vehicles = [{"label": row["vehicle_type"], "value": row["vehicle_type"]} for _, row in df.iterrows()]
    return vehicles


@app.get("/map", response_class=HTMLResponse)
def serve_map(request: Request):
    return templates.TemplateResponse("map.html", {"request": request})

@app.get("/locations")
def get_locations():
    locations = []
    for node in G.nodes:
        name = G.nodes[node].get("name", f"Node {node}")
        locations.append({"label": name, "value": str(node)})
    return locations


@app.get("/route")
def get_route(vehicle: str, start: int, end: int):
    consumption_row = df[df["vehicle_type"] == vehicle]
    consumption_rate = consumption_row.iloc[0]["consumption_wh_per_km"]

    start_node = int(start)
    end_node = int(end)

    G_updated = add_estimated_battery_and_temperature(G, model, x, node_id_to_idx, scaler)

    path_length = nx.shortest_path(G_updated, source=start_node, target=end_node, weight='length')
    path_battery = nx.shortest_path(G_updated, source=start_node, target=end_node, weight='battery')

    total_length_len, _ = compute_total_weight(G_updated, path_length, 'length')
    total_battery_len, slopes_len = compute_total_weight(G_updated, path_length, 'battery')
    total_length_bat, _ = compute_total_weight(G_updated, path_battery, 'length')
    total_battery_bat, slopes_bat = compute_total_weight(G_updated, path_battery, 'battery')

    length_type_len, length_type_bat = 'm', 'm'
    if total_length_len >= 1000:
        total_length_len = total_length_len / 1000
        length_type_len = 'km'
    if total_length_bat >= 1000:
        total_length_bat = total_length_bat / 1000
        length_type_bat = 'km'
        
    total_battery_len = total_battery_len * consumption_rate / base_consumption
    total_battery_bat = total_battery_bat * consumption_rate / base_consumption

    print("Slope in Length Path (indices 10-20):\n", slopes_len[10:20])
    print("\nSlope in Battery Path (indices 10-20):\n", slopes_bat[10:20])

    m = create_map(G_updated, path_length, path_battery, start_node, end_node)
    m.save("templates/map.html")

    return JSONResponse({
        "total_length_len": round(total_length_len, 2),
        "total_battery_len": round(total_battery_len, 2),
        "total_length_bat": round(total_length_bat, 2),
        "total_battery_bat": round(total_battery_bat, 2),
        "length_type_len": length_type_len, 
        "length_type_bat": length_type_bat
    })
