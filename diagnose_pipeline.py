import sys
from unittest.mock import MagicMock
# Mock pecanpy because it is not installed and not used in prepare_data
sys.modules['pecanpy'] = MagicMock()

import pipeline
import networkx as nx
import numpy as np
import pandas as pd
import io
import sys
from contextlib import redirect_stdout

def haversine(u_lat, u_lng, v_lat, v_lng):
    EARTH_RADIUS_KM = 6371.0088
    u_lat_rad, u_lng_rad = np.radians(u_lat), np.radians(u_lng)
    v_lat_rad, v_lng_rad = np.radians(v_lat), np.radians(v_lng)
    
    dlat = v_lat_rad - u_lat_rad
    dlng = v_lng_rad - u_lng_rad
    
    a = np.sin(dlat / 2.0)**2 + np.cos(u_lat_rad) * np.cos(v_lat_rad) * np.sin(dlng / 2.0)**2
    a = np.clip(a, 0.0, 1.0)
    dist_km = EARTH_RADIUS_KM * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return dist_km

def main():
    fpath = "data/undir_trials/thresholded_networks/0_network.txt"
    
    print("Step 2: Loading graph using pipeline.load...")
    G_full = pipeline.load(fpath)
    
    print("Step 3: Calling pipeline.prepare_data...")
    f = io.StringIO()
    with redirect_stdout(f):
        # meta=True returns G, train_non_edges, test_edges, test_non_edges
        results = pipeline.prepare_data(fpath, meta=True)
        
    captured_output = f.getvalue()
    print(captured_output)
    
    if results is None:
        print("Error: pipeline.prepare_data returned None.")
        return
        
    G, train_neg, test_pos, test_neg = results
    
    # Identify training edges (positives)
    # Note: prepare_data returns sorted test_pos tuples
    all_edges = {tuple(sorted(e)) for e in G.edges()}
    test_pos_set = {tuple(sorted(e)) for e in test_pos}
    train_pos = list(all_edges - test_pos_set)
    
    print(f"Step 6: Number of 'train_neg' edges: {len(train_neg)}")
    print(f"Number of training edges in G: {len(train_pos)}")
    
    # Step 4: Compare DIST_KM
    # Real edges DIST_KM
    dist_dict = nx.get_edge_attributes(G, 'DIST_KM')
    # Use sorted tuples for lookup
    train_pos_dist = []
    for e in train_pos:
        # Check original and reversed if needed, but DIST_KM is stored by edge
        d = dist_dict.get(e)
        if d is None:
            # Maybe it's stored as (v, u)
            d = dist_dict.get((e[1], e[0]))
        if d is not None:
            train_pos_dist.append(d)
        else:
            # Fallback to coordinate distance if DIST_KM is missing for some reason
            u, v = e
            d = haversine(G.nodes[u]['latitude'], G.nodes[u]['longitude'],
                          G.nodes[v]['latitude'], G.nodes[v]['longitude'])
            train_pos_dist.append(d)

    # train_neg DIST_KM (calculated using Haversine)
    train_neg_dist = []
    for u, v in train_neg:
        d = haversine(G.nodes[u]['latitude'], G.nodes[u]['longitude'],
                      G.nodes[v]['latitude'], G.nodes[v]['longitude'])
        train_neg_dist.append(d)
        
    # Step 5: Prints mean and median
    print("\nTraining Edges (Positive) Distance Statistics:")
    print(f"Mean: {np.mean(train_pos_dist):.4f} km")
    print(f"Median: {np.median(train_pos_dist):.4f} km")
    
    print("\nSampled 'train_neg' Edges Distance Statistics:")
    print(f"Mean: {np.mean(train_neg_dist):.4f} km")
    print(f"Median: {np.median(train_neg_dist):.4f} km")
    
    # Step 7: Check if there are any warnings printed during 'prepare_data'
    warnings = [line for line in captured_output.split('\n') if 'Warning' in line or 'Error' in line]
    if warnings:
        print("\nWarnings/Errors found during prepare_data:")
        for w in warnings:
            print(f"- {w}")
    else:
        print("\nNo warnings/errors found during prepare_data.")

if __name__ == "__main__":
    main()
