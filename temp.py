import numpy as np
import geopandas as gpd
from shapely.geometry import Point


def build_feature_matrix(edges, G, features, embedding_map, operator='hadamard'):
    """
    Build a feature matrix for a list of node pairs.

    Each row corresponds to one edge (u, v). The columns are determined
    by `features`, which is a list that can contain any combination of:

        'emb'       – binary-operator output on node2vec embeddings (128-d by default)
        'geo'       – log geographic distance in km  (1-d)
        'cat'       – 
        'cbg'       - binary for same census-block group
        'comm'      - binary for same infomap community
        'ls'        - 



    Parameters
    ----------
    edges : list of (u, v) tuples
    G : nx.Graph with node attributes (latitude, longitude, poi_type, total_visits)
    features : list of str
    embedding_map : dict  (required only when 'emb' in features)
    operator : str  (which binary operator to use for embeddings)

    Returns
    -------
    X : np.ndarray of shape (n_edges, n_features)
    kept_indices : list of int – indices into `edges` that were actually kept
        (some may be dropped if embeddings are missing)
    """
    op_fn = BINARY_OPERATORS[operator]

    # 1. Pre-filter edges missing embeddings to ensure matrix shapes align later
    valid_edges = []
    kept_indices = []

    if 'emb' in features:
        for idx, (u, v) in enumerate(edges):
            if u in embedding_map and v in embedding_map:
                valid_edges.append((u, v))
                kept_indices.append(idx)
    else:
        valid_edges = edges
        kept_indices = list(range(len(edges)))

    if not valid_edges:
        return np.empty((0, 0)), []

    # Unzip the list of tuples into two parallel lists of origins (U) and destinations (V)
    U, V = zip(*valid_edges)

    feature_blocks = []

    # 2. Vectorized Embedding Operations
    if 'emb' in features:
        # Extract to 2D arrays: shape (N, 128)
        emb_u = np.array([embedding_map[u] for u in U])
        emb_v = np.array([embedding_map[v] for v in V])

        # Binary operator applies to the entire (N, 128) array simultaneously
        emb_feat = op_fn(emb_u, emb_v)
        feature_blocks.append(emb_feat)

    # 3. Vectorized Geographic Distance (Pure NumPy Haversine)
    if 'geo' in features:
        # Fast extraction using list comprehensions (dict lookups are fast, math is slow)
        lat_u = np.array([G.nodes[u].get('latitude', 0) for u in U])
        lng_u = np.array([G.nodes[u].get('longitude', 0) for u in U])
        lat_v = np.array([G.nodes[v].get('latitude', 0) for v in V])
        lng_v = np.array([G.nodes[v].get('longitude', 0) for v in V])

        # Convert all coordinates to radians at once
        lat_u_rad, lng_u_rad = np.radians(lat_u), np.radians(lng_u)
        lat_v_rad, lng_v_rad = np.radians(lat_v), np.radians(lng_v)

        dlat = lat_v_rad - lat_u_rad
        dlng = lng_v_rad - lng_u_rad

        # Calculate Haversine on the 1D arrays
        a = np.sin(dlat / 2.0)**2 + np.cos(lat_u_rad) * \
            np.cos(lat_v_rad) * np.sin(dlng / 2.0)**2

        # Clip 'a' to [0, 1] to prevent NaN errors in sqrt due to floating-point precision limits
        a = np.clip(a, 0.0, 1.0)

        dist_km = 6371.0088 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

        # Log scale and reshape to (N, 1) column vector
        geo_feat = np.log1p(dist_km).reshape(-1, 1)
        feature_blocks.append(geo_feat)

    # one-hot (new matrix for each combo i believe)
    if 'cat' in features:
        # Build vocabulary from ALL nodes in G so train/test columns always align
        all_cats = sorted({
            G.nodes[n].get('poi_type', 'Unknown')
            for n in G.nodes()
        })
        cat_to_idx = {c: i for i, c in enumerate(all_cats)}
        n_cats = len(all_cats)

        cat_u = [G.nodes[u].get('poi_type', 'Unknown') for u in U]
        cat_v = [G.nodes[v].get('poi_type', 'Unknown') for v in V]

        # Build two one-hot matrices: (N, n_cats) each
        oh_u = np.zeros((len(U), n_cats))
        oh_v = np.zeros((len(V), n_cats))

        for i, (cu, cv) in enumerate(zip(cat_u, cat_v)):
            oh_u[i, cat_to_idx.get(cu, 0)] = 1.0
            oh_v[i, cat_to_idx.get(cv, 0)] = 1.0

        # Stack side by side: (N, 2 * n_cats)
        cat_oh_feat = np.hstack([oh_u, oh_v])
        feature_blocks.append(cat_oh_feat)

    if 'cbg' in features:
        cbg_u = np.array([G.nodes[u].get('cbg', 'Unknown') for u in U])
        cbg_v = np.array([G.nodes[v].get('cbg', 'Unknown') for v in V])
        cbg_feat = (cbg_u == cbg_v).astype(float).reshape(-1, 1)
        feature_blocks.append(cbg_feat)

    if 'comm' in features:
        pass

    if 'ls' in features:
        pass

    # Assemble the Final Matrix
    # Horizontally stack all requested feature blocks into a single matrix
    X = np.hstack(feature_blocks)

    return X, kept_indices
