import folium

def create_map(G, path_length, path_battery, start_node, end_node):
    m = folium.Map(location=[G.nodes[start_node]['y'], G.nodes[start_node]['x']], zoom_start=14)

    def draw_path(route, color, tooltip, attr):
        for u, v in zip(route[:-1], route[1:]):
            edge_data = G.get_edge_data(u, v)
            if edge_data:
                min_key, min_data = min(edge_data.items(), key=lambda item: item[1].get(attr, float('inf')))
                if 'geometry' in min_data:
                    coords = [(lat, lon) for lon, lat in min_data['geometry'].coords]
                else:
                    coords = [(G.nodes[u]['y'], G.nodes[u]['x']), (G.nodes[v]['y'], G.nodes[v]['x'])]
                folium.PolyLine(coords, color=color, weight=5, tooltip=tooltip).add_to(m)

    draw_path(path_length, color='green', tooltip='Shortest (Length)', attr="length")
    draw_path(path_battery, color='blue', tooltip='Shortest (Battery)',  attr="battery")

    folium.Marker(location=[G.nodes[start_node]['y'], G.nodes[start_node]['x']],
                  tooltip="Start", icon=folium.Icon(color='green')).add_to(m)
    folium.Marker(location=[G.nodes[end_node]['y'], G.nodes[end_node]['x']],
                  tooltip="End", icon=folium.Icon(color='red')).add_to(m)

    return m
