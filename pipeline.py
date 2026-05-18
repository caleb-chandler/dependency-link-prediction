import networkx as nx
import numpy as np
import pandas as pd
from pecanpy import pecanpy as n2v
import random
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy.io import loadmat
from tqdm.auto import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import BallTree
import geopandas as gpd
from shapely.geometry import Point
from infomap import Infomap


def load(fpath, compress=False):
    df = pd.read_csv(fpath, sep='\s+', names=['ORIGIN', 'DESTINATION', 'N_COVISITS', 'TAXONOMY_ORIGIN',
                                              'TAXONOMY_DESTINATION', 'LAT_ORIGIN', 'LNG_ORIGIN', 'LAT_DESTINATION',
                                              'LNG_DESTINATION', 'DIST_KM', 'N_UIDS_ORIGIN', 'N_VISITS_ORIGIN',
                                              'N_UIDS_DESTINATION', 'N_VISITS_DESTINATION', 'DEP'])

    # optional log-compression
    if compress:
        scale_factor = 1.0 / np.median(df['DEP'])
        df['DEP'] = df['DEP'].apply(lambda x: np.log1p(x * scale_factor))
        scale_factor = 1.0 / np.median(df['N_COVISITS'])
        df['N_COVISITS'] = df['N_COVISITS'].apply(
            lambda x: np.log1p(x * scale_factor))

    G = nx.from_pandas_edgelist(
        df,
        source='ORIGIN',
        target='DESTINATION',
        edge_attr=['N_COVISITS', 'DIST_KM', 'DEP'],
    )
    # assign node attrs
    origins = df[['ORIGIN', 'LAT_ORIGIN', 'LNG_ORIGIN',
                  'TAXONOMY_ORIGIN', 'N_UIDS_ORIGIN', 'N_VISITS_ORIGIN']].drop_duplicates()
    origins.columns = ['node_id', 'lat', 'lng',
                       'taxonomy', 'unique_visits', 'total_visits']
    destinations = df[['DESTINATION', 'LAT_DESTINATION',
                       'LNG_DESTINATION', 'TAXONOMY_DESTINATION',
                       'N_UIDS_DESTINATION', 'N_VISITS_DESTINATION']].drop_duplicates()
    destinations.columns = ['node_id', 'lat', 'lng',
                            'taxonomy', 'unique_visits', 'total_visits']

    # combine them into one master list of unique POIs
    node_data = pd.concat([origins, destinations]).drop_duplicates(
        'node_id').set_index('node_id')

    # map back to graph - convert to dict and apply
    lat_dict = node_data['lat'].to_dict()
    lng_dict = node_data['lng'].to_dict()
    tax_dict = node_data['taxonomy'].to_dict()
    uv_dict = node_data['unique_visits'].to_dict()
    tv_dict = node_data['total_visits'].to_dict()
    nx.set_node_attributes(G, lat_dict, 'latitude')
    nx.set_node_attributes(G, lng_dict, 'longitude')
    nx.set_node_attributes(G, tax_dict, 'poi_type')
    nx.set_node_attributes(G, uv_dict, 'unique_visits')
    nx.set_node_attributes(G, tv_dict, 'total_visits')

    print(f"Nodes: {G.number_of_nodes()}")
    print(f"Edges: {G.number_of_edges()}")
    return G

# ================================================================


def distribution_finder(G, bins=50):
    # --- helper: continuous data (bins ---)
    def get_binned_dist(data_dict, bins):
        if not data_dict:
            return pd.Series(dtype='float64'), {}

        # convert dict to series. index = node id or edge tuple, value = attribute
        s = pd.Series(data_dict).dropna()

        # if bins is an integer, let numpy calculate linear edges. otherwise, use custom edges.
        if isinstance(bins, int):
            _, bin_edges = np.histogram(s, bins=bins)
        else:
            bin_edges = bins

        # cut the data into bins. this returns true interval objects
        binned = pd.cut(s, bins=bin_edges, include_lowest=True)

        # The distribution is the count of edges in each Interval
        distr = binned.value_counts().sort_index()

        # group by the bin intervals and extract the ids as a set
        # this creates an interval:nodes dict
        elements_by_bin = s.groupby(binned, observed=False).apply(
            lambda x: set(x.index)).to_dict()

        return distr, elements_by_bin

    # --- helper: discrete/categorical data ---
    def get_discrete_dist(data_dict):
        if not data_dict:
            return pd.Series(dtype='float64'), {}

        s = pd.Series(data_dict)
        distr = s.value_counts().sort_index()

        # group exactly by the value
        elements_by_val = s.groupby(s).apply(lambda x: set(x.index)).to_dict()

        return distr, elements_by_val

    # --- edges: distance (continuous) ---
    dist_dict = nx.get_edge_attributes(G, 'DIST_KM')
    if dist_dict:
        max_d = max(dist_dict.values())
        if max_d > 0.01:
            # Create logarithmic bins: start with 0, then exponentially space from 0.01km (10m) to max_d
            custom_bins = np.concatenate(([0], np.geomspace(0.01, max_d, 50)))
        else:
            custom_bins = np.linspace(0, max_d + 1e-5, 50)
    else:
        custom_bins = bins

    dist_distr, dist_edges_set = get_binned_dist(dist_dict, custom_bins)

    # --- edges: covisits (continuous/discrete) ---
    cv_dict = nx.get_edge_attributes(G, 'N_COVISITS')
    cv_distr, cv_edges_set = get_binned_dist(cv_dict, bins)

    # --- nodes: categorical (poi type) ---
    type_dict = {u: data.get('poi_type', 'Unknown')
                 for u, data in G.nodes(data=True)}
    type_distr, type_nodes_set = get_discrete_dist(type_dict)

    # --- nodes: unique visits (continuous) ---
    uv_dict = {u: data.get('unique_visits', 0)
               for u, data in G.nodes(data=True)}
    uv_distr, uv_nodes_set = get_binned_dist(uv_dict, bins)

    # --- nodes: total visits (continuous) ---
    tv_dict = {u: data.get('total_visits', 0)
               for u, data in G.nodes(data=True)}
    tv_distr, tv_nodes_set = get_binned_dist(tv_dict, bins)

    # --- topology: degree (discrete) ---
    deg_dict = dict(G.degree())
    deg_distr, deg_nodes_set = get_discrete_dist(deg_dict)

    # pack the results logically so it's easy to return
    distributions = (dist_distr, cv_distr, type_distr,
                     uv_distr, tv_distr, deg_distr)
    element_sets = (dist_edges_set, cv_edges_set, type_nodes_set,
                    uv_nodes_set, tv_nodes_set, deg_nodes_set)

    return distributions, element_sets

# ===================================================================


def sample_non_edges_dist_controlled(G, distr, total_count):
    # extract coordinates and build a spatial index
    nodes = list(G.nodes())
    coords = np.radians([
        [G.nodes[n].get('latitude', 0), G.nodes[n].get('longitude', 0)]
        for n in nodes
    ])

    EARTH_RADIUS_KM = 6371.0088
    tree = BallTree(coords, metric='haversine')

    non_edges = set()

    # calculate quotas for each bin based on the target distribution
    total_edges_in_distr = distr.sum()
    if total_edges_in_distr == 0:
        raise ValueError("Error: Distribution is empty.")

    bin_quotas = {}
    for interval, count in distr.items():
        proportion = count / total_edges_in_distr
        # Round to nearest integer for the quota
        bin_quotas[interval] = int(np.round(proportion * total_count))

    # sample using targeted spatial queries
    with tqdm(total=total_count, desc='Sampling spatial non-edges', unit='edge', leave=False) as pbar:
        for interval, quota in bin_quotas.items():
            if quota <= 0:
                continue

            # Now we dynamically extract exact bounds from the IntervalIndex
            d_min_km = interval.left
            d_max_km = interval.right

            # Convert km to radians for the BallTree
            r_min = d_min_km / EARTH_RADIUS_KM
            r_max = d_max_km / EARTH_RADIUS_KM

            samples_for_bin = 0
            attempts = 0
            max_attempts = quota * 50

            while samples_for_bin < quota and attempts < max_attempts:
                attempts += 1

                # pick a random source node
                u_idx = random.randint(0, len(nodes) - 1)
                u = nodes[u_idx]
                u_coord = coords[u_idx:u_idx+1]

                # query the tree for nodes within the outer radius
                indices_within_max = tree.query_radius(u_coord, r=r_max)[0]

                # query the tree for nodes within the inner radius
                indices_within_min = tree.query_radius(u_coord, r=r_min)[0]

                # set difference gives us the nodes existing strictly in the target distance ring
                valid_indices = np.setdiff1d(
                    indices_within_max, indices_within_min)

                if len(valid_indices) == 0:
                    continue  # no nodes exist at this specific distance from node u

                # pick a random valid destination node
                v_idx = random.choice(valid_indices)
                v = nodes[v_idx]

                if u == v:
                    continue

                # enforce strict tuple ordering so (u,v) and (v,u) aren't duplicated
                edge = (u, v) if u < v else (v, u)

                # verify it's not a real edge and hasn't been sampled yet
                if not G.has_edge(*edge) and edge not in non_edges:
                    non_edges.add(edge)
                    samples_for_bin += 1
                    pbar.update(1)

            if attempts >= max_attempts:
                print(
                    f"Warning: Could not fulfill spatial quota for bin [{d_min_km:.2f}, {d_max_km:.2f}]. Got {samples_for_bin}/{quota}.")

    return list(non_edges)

# ====================================================================


def prepare_data(fpath, frac=0.5, seed=None, compress=0, weight=None):
    """
    Prepare data for link prediction pipeline.

    This function loads a graph from a file, splits it into training and testing sets,
    saves the resulting training graph in the root folder, and outputs negative training edges,
    negative test edges, and positive test edges.

    Parameters:
    fpath (str): Path to the graph file. Must be readable as an edgelist.
    frac (float, optional): Fraction of edges to use for testing. Default is 0.5.
    seed (int, optional): Seed for reproducibility.

    Returns:
    nx.Graph : original graph
    list : list of negative training samples
    list : list of positive testing samples
    list : list of negative testing samples
    """

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    def split(G, frac=frac):
        # load edges as sorted tuples for efficiency
        edges = {tuple(sorted(e)) for e in G.edges()}
        mst = {tuple(sorted(e))
               for e in nx.maximum_spanning_tree(G, weight='DEP')}

        # fast set difference
        removable_edges = list(edges - mst)
        test_count = min(int(len(edges) * frac), len(removable_edges))
        test_edges = random.sample(removable_edges, test_count)
        train_edges = list(edges - set(test_edges))

        # build training graph
        G_train = nx.Graph()
        G_train.add_nodes_from(G.nodes())
        # handle weights or not
        if weight == 'dep':
            G_train.add_weighted_edges_from(
                [(u, v, G[u][v]['DEP']) for u, v in train_edges])
        elif weight == 'cov':
            G_train.add_weighted_edges_from(
                [(u, v, G[u][v]['N_COVISITS']) for u, v in train_edges])
        else:
            G_train.add_edges_from(train_edges)

        # sample non-edges by bin to preserve distribution
        distrs, sets = distribution_finder(G)
        dist_bins = distrs[0]  # distance dict

        # getting true negatives (overlap possible but very improbable for large datasets)
        test_non_edges = sample_non_edges_dist_controlled(
            G, dist_bins, len(test_edges))
        train_non_edges = sample_non_edges_dist_controlled(
            G, dist_bins, len(train_edges))

        return G_train, test_edges, test_non_edges, train_non_edges

    G = load(fpath, compress=compress)

    # failsafe
    if G.number_of_nodes() == 0:
        print(f"Error: Graph loaded from {fpath} is entirely empty. Skipping.")
        return None

    # extracting lcc in case disconnected
    largest_cc = max(nx.connected_components(G), key=len)
    G = G.subgraph(largest_cc).copy()

    G_train, test_edges, test_non_edges, train_non_edges = split(G)

    # error handling
    if nx.is_empty(G_train):
        print("Error: Empty training graph.")
        return None

    # saving training graph
    with open('train.txt', 'w') as f:
        for u, v, d in G_train.edges(data=True):
            f.write(f"{u} {v} {d.get('weight', 1.0)}\n")
    print(
        f"Wrote training graph: {G_train.number_of_nodes()} nodes, {G_train.number_of_edges()} edges")

    return G, train_non_edges, test_edges, test_non_edges

# ====================================================================


BINARY_OPERATORS = {
    'avg': lambda a, b: np.mean([a, b], axis=0),
    'hadamard': lambda a, b: np.multiply(a, b),
    'w-l1': lambda a, b: np.abs(np.subtract(a, b)),
    'w-l2': lambda a, b: np.square(np.subtract(a, b)),
}

# ===================================================================


def assign_cbg_to_nodes(G, shapefile_path):
    """Add 'cbg' attribute to each node in G via spatial join."""
    nodes = list(G.nodes())
    lats = [G.nodes[n].get('latitude', 0) for n in nodes]
    lngs = [G.nodes[n].get('longitude', 0) for n in nodes]

    poi_gdf = gpd.GeoDataFrame(
        {'node_id': nodes},
        geometry=[Point(lng, lat) for lng, lat in zip(lngs, lats)],
        crs='EPSG:4326'
    )

    cbg_gdf = gpd.read_file(shapefile_path).to_crs('EPSG:4326')
    joined = gpd.sjoin(poi_gdf, cbg_gdf, how='left', predicate='within')

    for _, row in joined.iterrows():
        G.nodes[row['node_id']]['cbg'] = row.get('GEOID', 'Unknown')


def assign_node_to_comm(G):
    im = Infomap("--num-trials 20")
    im_to_nx = im.add_networkx_graph(G, weight='N_COVISITS')
    print("Running Infomap...")
    im.run()
    print("Done.")

    for node_id, module_id in im.modules:
        G.nodes[im_to_nx[node_id]]['community'] = module_id

    print(
        f"Assigned {len(set(nx.get_node_attributes(G, 'community').values()))} communities")


def build_feature_matrix(edges, G, features, embedding_map, operator='hadamard', cat_threshold=1):
    """
    Build a feature matrix for a list of node pairs.

    Each row corresponds to one edge (u, v). The columns are determined
    by `features`, which is a list that can contain any combination of:

        'emb'       – binary-operator output on node2vec embeddings (128-d by default)
        'geo'       – log geographic distance in km  (1-d)
        'cat'       – (N_edges, N_interactions) matrix with binary corresponding to interaction type
        'cat_same'  - simplified same/different category feature for baseline comparison
        'cbg'       - binary for same/different census-block group
        'comm'      - binary for same/different infomap community
        'ls'        - concatenated embeddings from endpoint categories constructed from word2vec on activity sequences

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

    if 'cat' in features:
        # Count each undirected type-pair across all edges in G
        pair_counts = {}
        for eu, ev in G.edges():
            cu = G.nodes[eu].get('poi_type', 'Unknown')
            cv = G.nodes[ev].get('poi_type', 'Unknown')
            pair = tuple(sorted([cu, cv]))
            pair_counts[pair] = pair_counts.get(pair, 0) + 1

        # Vocabulary: only pairs observed >= cat_threshold times, sorted for stable columns
        vocab = sorted(p for p, cnt in pair_counts.items()
                       if cnt >= cat_threshold)
        print(
            f'# kept pairs with threshold {cat_threshold}: {len(vocab)}/210 ({(len(vocab)/210)*100}%)')
        pair_to_idx = {p: i for i, p in enumerate(vocab)}

        cat_feat = np.zeros((len(U), len(vocab)))
        for i, (u, v) in enumerate(zip(U, V)):
            cu = G.nodes[u].get('poi_type', 'Unknown')
            cv = G.nodes[v].get('poi_type', 'Unknown')
            pair = tuple(sorted([cu, cv]))
            idx = pair_to_idx.get(pair)
            if idx is not None:
                cat_feat[i, idx] = 1.0

        feature_blocks.append(cat_feat)

    if 'cat_same' in features:
        cat_u = np.array([G.nodes[u].get('poi_type', '') for u in U])
        cat_v = np.array([G.nodes[v].get('poi_type', '') for v in V])

        # Boolean array comparison converted to floats: 1.0 for True, 0.0 for False
        cat_feat = (cat_u == cat_v).astype(float).reshape(-1, 1)
        feature_blocks.append(cat_feat)

    if 'cbg' in features:
        cbg_u = np.array([G.nodes[u].get('cbg', 'Unknown') for u in U])
        cbg_v = np.array([G.nodes[v].get('cbg', 'Unknown') for v in V])
        cbg_feat = ((cbg_u == cbg_v) & (cbg_u != 'Unknown')
                    ).astype(float).reshape(-1, 1)
        feature_blocks.append(cbg_feat)

    if 'comm' in features:
        comm_u = np.array([G.nodes[u].get('community', -1) for u in U])
        comm_v = np.array([G.nodes[v].get('community', -1) for v in V])
        comm_feat = ((comm_u == comm_v) & (comm_u != -1)
                     ).astype(float).reshape(-1, 1)
        feature_blocks.append(comm_feat)

    if 'ls' in features:
        pass

    # Assemble the Final Matrix
    # Horizontally stack all requested feature blocks into a single matrix
    X = np.hstack(feature_blocks)

    return X, kept_indices

# ====================================================================


def run_pipeline(trainfile, train_non_edges, test_edges, test_non_edges, G=None, features=['emb'],
                 mode='PreComp', operator='hadamard', **kwargs):
    """
    Run the link prediction pipeline with flexible feature composition. Features are controlled by the `features` list.

    Parameters
    ----------
    trainfile : str
        Path to the training graph edgelist file.
    train_non_edges : list
        Negative training edges.
    test_edges : list
        Positive testing edges.
    test_non_edges : list
        Negative testing edges.
    G : nx.Graph
        The *original* graph with node attributes (latitude, longitude,
        poi_type, total_visits). Required when features includes anything
        other than 'emb'.
    features : list of str or 'all'
        Which features to include. Default ['emb']. If 'all' then includes all features.
    mode : str
        PecanPy walk mode. Default 'PreComp'.
    operator : str
        Binary operator for embeddings. Default 'hadamard'.
    **kwargs
        Hyperparameter settings forwarded to PecanPy / Word2Vec. Also allows for seed.

    Returns
    -------
    auc : float
        AUC score for the specified feature/operator combination.
    embedding_map : dict or None
        Node embeddings (only populated when 'emb' in features).
    """
    # === unpacking kwargs ===
    # hyperparameters
    p = kwargs.get('p', 1)
    q = kwargs.get('q', 1)
    workers = kwargs.get('workers', 6)
    verbose = kwargs.get('verbose', True)
    dim = kwargs.get('dim', 128)
    num_walks = kwargs.get('num_walks', 10)
    walk_length = kwargs.get('walk_length', 80)
    window_size = kwargs.get('window_size', 10)
    epochs = kwargs.get('epochs', 1)
    cat_threshold = kwargs.get('cat_threshold', 1)
    # allow passing in precomputed embeddings
    embedding_map = kwargs.get('embedding_map', None)
    # switch for weighted/directed version
    weighted = kwargs.get('weighted', False)
    directed = kwargs.get('directed', False)

    # seed
    seed = kwargs.get('seed', None)
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    if features == 'all':
        features = ['emb', 'geo', 'cat', 'cbg', 'comm']

    # ===== Validation =====
    needs_metadata = bool({'geo', 'cat', 'cbg', 'comm'} & set(features))
    if needs_metadata and G is None:
        raise ValueError(
            "Graph G with node attributes is required when features "
            f"include {[f for f in features if f != 'emb']}"
        )

    # converting training graph to nx.Graph object
    G_train = nx.read_edgelist(trainfile, data=[('weight', float)])

    # ===== Embedding generation (only if needed) =====

    if 'emb' in features and embedding_map is not None:
        print(f"Using precomputed embeddings: {len(embedding_map)} nodes")

    elif 'emb' in features:
        def make_pecanpy_graph(chosen_mode, w_bool):
            if chosen_mode == 'PreComp':
                return n2v.PreComp(p=p, q=q, workers=workers, verbose=verbose, extend=w_bool)
            elif chosen_mode == 'SparseOTF':
                return n2v.SparseOTF(p=p, q=q, workers=workers, verbose=verbose, extend=w_bool)
            elif chosen_mode == 'DenseOTF':
                return n2v.DenseOTF(p=p, q=q, workers=workers, verbose=verbose, extend=w_bool)
            else:
                raise ValueError(f"Unknown pecanpy mode: {chosen_mode}")

        tried_modes = [mode]
        if mode != 'PreComp':
            tried_modes.append('PreComp')
        if mode not in ['SparseOTF', 'DenseOTF']:
            tried_modes.append('DenseOTF')
        # PreComp alias_indptr overflows uint32 for large weighted graphs;
        # SparseOTF computes transition probs on-the-fly and avoids this.
        if weighted and 'SparseOTF' not in tried_modes:
            tried_modes.insert(0, 'SparseOTF')

        last_exception = None
        for candidate_mode in tried_modes:
            try:
                g = make_pecanpy_graph(candidate_mode, weighted)
                g.read_edg(trainfile, weighted=weighted,
                           directed=directed, delimiter=' ')
                if candidate_mode == 'PreComp':
                    g.preprocess_transition_probs()

                embeddings = g.embed(
                    dim=dim, num_walks=num_walks,
                    walk_length=walk_length, window_size=window_size,
                    epochs=epochs, verbose=verbose,
                )

                if candidate_mode != mode:
                    print(f"Warning: fell back to '{candidate_mode}'")
                break
            except Exception as e:
                print(f"Warning: pecanpy mode '{candidate_mode}' failed: {e}")
                last_exception = e
                continue
        else:
            raise RuntimeError(
                f"Pecanpy walk generation failed for all modes: {tried_modes}"
            ) from last_exception

        embedding_map = {}
        for i, node_id in enumerate(g.nodes):
            if node_id is None:
                continue
            try:
                key = int(node_id)
            except (ValueError, TypeError):
                key = node_id
            embedding_map[key] = embeddings[i].tolist()

        print(f"Embeddings generated: {len(embedding_map)} nodes, dim={dim}")

    # ===== Assemble feature matrices =====
    train_pos_edges = [tuple(sorted(e)) for e in G_train.edges()]

    if 'cbg' in features:
        assign_cbg_to_nodes(G, 'data/cbg/tl_2025_25_bg.shp')

    if 'comm' in features:
        assign_node_to_comm(G)

    X_train_pos, _ = build_feature_matrix(
        train_pos_edges, G, features, embedding_map, operator, cat_threshold)
    X_train_neg, _ = build_feature_matrix(
        train_non_edges, G, features, embedding_map, operator, cat_threshold)

    X_train = np.vstack([X_train_pos, X_train_neg])
    y_train = np.concatenate([
        np.ones(len(X_train_pos)),
        np.zeros(len(X_train_neg))
    ])

    # shuffle
    shuffle_idx = np.random.permutation(len(y_train))
    X_train = X_train[shuffle_idx]
    y_train = y_train[shuffle_idx]

    print(
        f"Training matrix: {X_train.shape[0]} samples x {X_train.shape[1]} features")

    # ===== Train =====
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(X_train, y_train)

    # ===== Test =====
    X_test_pos, _ = build_feature_matrix(
        test_edges, G, features, embedding_map, operator, cat_threshold)
    X_test_neg, _ = build_feature_matrix(
        test_non_edges, G, features, embedding_map, operator, cat_threshold)

    X_test = np.vstack([X_test_pos, X_test_neg])
    y_test = np.concatenate([
        np.ones(len(X_test_pos)),
        np.zeros(len(X_test_neg))
    ])

    probs = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probs)

    # ===== Report =====
    feature_label = '+'.join(features)
    op_label = f" ({operator})" if 'emb' in features else ""
    print(f"[{feature_label}{op_label}]  AUC = {auc:.4f}")

    return auc, embedding_map
