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


def load(fpath):
    df = pd.read_csv(fpath, sep='\s+', names=['ORIGIN', 'DESTINATION', 'N_COVISITS', 'TAXONOMY_ORIGIN',
                                              'TAXONOMY_DESTINATION', 'LAT_ORIGIN', 'LNG_ORIGIN', 'LAT_DESTINATION',
                                              'LNG_DESTINATION', 'DIST_KM', 'N_UIDS_ORIGIN', 'N_VISITS_ORIGIN',
                                              'N_UIDS_DESTINATION', 'N_VISITS_DESTINATION', 'DEP'])
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

    # verification
    print(f"Attribute check: {random.choice(list(G.nodes(data=True)))}")

    print(f"Nodes: {G.number_of_nodes()}")
    print(f"Edges: {G.number_of_edges()}")
    return G

# ================================================================


def distribution_finder(G):
    # --- helper: continuous data (bins ---)
    def get_binned_dist(data_dict, bins=30):
        if not data_dict:
            return pd.Series(dtype='float64'), {}

        # convert dict to series. index = node id or edge tuple, value = attribute
        s = pd.Series(data_dict).dropna()

        # calculate histogram and bins
        counts, bin_edges = np.histogram(s, bins=bins)
        distr = pd.Series(counts, index=bin_edges[:-1])

        # Cut the data into the exact same bins and group them
        binned = pd.cut(s, bins=bin_edges, include_lowest=True)

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
    dist_distr, dist_edges_set = get_binned_dist(dist_dict, bins=30)

    # --- edges: covisits (continuous/discrete) ---
    cv_dict = nx.get_edge_attributes(G, 'N_COVISITS')
    cv_distr, cv_edges_set = get_binned_dist(cv_dict, bins=30)

    # --- nodes: categorical (poi type) ---
    type_dict = {u: data.get('poi_type', 'Unknown')
                 for u, data in G.nodes(data=True)}
    type_distr, type_nodes_set = get_discrete_dist(type_dict)

    # --- nodes: unique visits (continuous) ---
    uv_dict = {u: data.get('unique_visits', 0)
               for u, data in G.nodes(data=True)}
    uv_distr, uv_nodes_set = get_binned_dist(uv_dict, bins=30)

    # --- nodes: total visits (continuous) ---
    tv_dict = {u: data.get('total_visits', 0)
               for u, data in G.nodes(data=True)}
    tv_distr, tv_nodes_set = get_binned_dist(tv_dict, bins=30)

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


def sample_non_edges(G, distr, total_count):
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
    for left_edge, count in distr.items():
        proportion = count / total_edges_in_distr
        # round to nearest integer for the quota
        bin_quotas[left_edge] = int(np.round(proportion * total_count))

    # determine the uniform bin width from the pandas Series index
    index_vals = list(distr.index)
    bin_width = index_vals[1] - index_vals[0] if len(index_vals) > 1 else 1.0

    # sample using targeted spatial queries
    with tqdm(total=total_count, desc='Sampling spatial non-edges', unit='edge', leave=False) as pbar:
        for d_min_km, quota in bin_quotas.items():
            if quota <= 0:
                continue

            # reconstruct the max distance using the bin width
            d_max_km = d_min_km + bin_width

            # convert km to radians for the balltree
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


def prepare_data(fpath, ftype=None, frac=0.5):
    """
    Prepare data for link prediction pipeline.

    This function loads a graph from a file, splits it into training and testing sets,
    saves the resulting training graph in the root folder, and outputs negative training edges,
    negative test edges, and positive test edges.

    Parameters:
    fpath (str): Path to the graph file. Must be readable as an edgelist.
    frac (float, optional): Fraction of edges to use for testing. Default is 0.5.

    Returns:
    list : list of negative training samples
    list : list of positive testing samples
    list : list of negative testing samples
    """

    def split(G, frac=frac):
        # load edges as sorted tuples for efficiency
        edges = {tuple(sorted(e)) for e in G.edges()}
        mst = {tuple(sorted(e)) for e in nx.minimum_spanning_tree(G).edges()}

        # fast set difference
        removable_edges = list(edges - mst)
        test_count = min(int(len(edges) * frac), len(removable_edges))
        test_edges = random.sample(removable_edges, test_count)
        train_edges = list(edges - set(test_edges))

        # build training graph
        G_train = nx.Graph()
        G_train.add_nodes_from(G.nodes())
        G_train.add_edges_from(train_edges)

        # sample non-edges by bin to preserve distribution
        distrs, sets = distribution_finder(G)
        dist_bins = distrs[0]  # distance dict

        # getting true negatives (overlap possible but very improbable for large datasets)
        test_non_edges = sample_non_edges(G, dist_bins, len(test_edges))
        train_non_edges = sample_non_edges(G, dist_bins, len(train_edges))

        return G_train, test_edges, test_non_edges, train_non_edges

    G = load(fpath)

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
    nx.write_edgelist(G_train, 'train.txt', data=False)
    print(
        f"Wrote training graph: {G_train.number_of_nodes()} nodes, {G_train.number_of_edges()} edges")

    return train_non_edges, test_edges, test_non_edges

# ====================================================================


BINARY_OPERATORS = {
    'avg': lambda a, b: np.mean([a, b], axis=0),
    'hadamard': lambda a, b: np.multiply(a, b),
    'w-l1': lambda a, b: np.abs(np.subtract(a, b)),
    'w-l2': lambda a, b: np.square(np.subtract(a, b)),
}

# ===================================================================


def build_feature_matrix(edges, G, features, embedding_map, operator='hadamard'):
    """
    Build a feature matrix for a list of node pairs.

    Each row corresponds to one edge (u, v). The columns are determined
    by `features`, which is a list that can contain any combination of:

        'emb'       – binary-operator output on node2vec embeddings (128-d by default)
        'geo'       – log geographic distance in km  (1-d)
        'cat'       – 1 if same top-level POI category, else 0  (1-d)
        'visits'    – log total visits for u and v  (2-d)

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
    rows = []
    kept_indices = []

    for idx, (u, v) in enumerate(edges):
        vec = []

        # --- embedding features ---
        if 'emb' in features:
            emb_u = embedding_map.get(u)
            emb_v = embedding_map.get(v)
            if emb_u is None or emb_v is None:
                continue  # skip this edge entirely
            vec.append(op_fn(np.array(emb_u), np.array(emb_v)))

        # --- geographic distance ---
        if 'geo' in features:
            lat_u = G.nodes[u].get('latitude', 0)
            lng_u = G.nodes[u].get('longitude', 0)
            lat_v = G.nodes[v].get('latitude', 0)
            lng_v = G.nodes[v].get('longitude', 0)

            # haversine in km (same formula your BallTree uses)
            dlat = np.radians(lat_v - lat_u)
            dlng = np.radians(lng_v - lng_u)
            a = (np.sin(dlat / 2) ** 2 +
                 np.cos(np.radians(lat_u)) * np.cos(np.radians(lat_v)) *
                 np.sin(dlng / 2) ** 2)
            dist_km = 6371.0088 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

            # log(1 + dist) avoids log(0) and compresses scale
            vec.append(np.array([np.log1p(dist_km)]))

        # --- same cat indicator ---
        if 'cat' in features:
            cat_u = G.nodes[u].get('poi_type', '')
            cat_v = G.nodes[v].get('poi_type', '')
            vec.append(np.array([1.0 if cat_u == cat_v else 0.0]))

        # --- visit counts ---
        if 'visits' in features:
            tv_u = G.nodes[u].get('total_visits', 1)
            tv_v = G.nodes[v].get('total_visits', 1)
            vec.append(np.array([np.log1p(tv_u), np.log1p(tv_v)]))

        rows.append(np.concatenate(vec))
        kept_indices.append(idx)

    X = np.stack(rows) if rows else np.empty((0, 0))
    return X, kept_indices

# ====================================================================


def run_pipeline(trainfile, train_non_edges, test_edges, test_non_edges, G=None, features=['emb'], mode='PreComp', operator='hadamard', **kwargs):
    """
    Run the link prediction pipeline with flexible feature composition.

    Features are controlled by the `features` list, which can contain
    any combination of:
        'emb'       – node2vec embedding (binary operator applied per edge)
        'geo'       – log geographic distance between nodes
        'cat'       – same/different POI category indicator
        'visits'    – log total visits for both nodes

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
    features : list of str
        Which features to include. Default ['emb'].
    mode : str
        PecanPy walk mode. Default 'PreComp'.
    operator : str
        Binary operator for embeddings. Default 'hadamard'.
    **kwargs
        Hyperparameter settings forwarded to PecanPy / Word2Vec.

    Returns
    -------
    auc : float
        AUC score for the specified feature/operator combination.
    embedding_map : dict or None
        Node embeddings (only populated when 'emb' in features).
    """
    # unpacking kwargs
    p = kwargs.get('p', 1)
    q = kwargs.get('q', 1)
    workers = kwargs.get('workers', 4)
    verbose = kwargs.get('verbose', True)
    dim = kwargs.get('dim', 128)
    num_walks = kwargs.get('num_walks', 10)
    walk_length = kwargs.get('walk_length', 80)
    window_size = kwargs.get('window_size', 10)
    epochs = kwargs.get('epochs', 1)

    # ===== Validation =====
    needs_metadata = bool({'geo', 'cat', 'visits'} & set(features))
    if needs_metadata and G is None:
        raise ValueError(
            "Graph G with node attributes is required when features "
            f"include {[f for f in features if f != 'emb']}"
        )

    # converting training graph to nx.Graph object
    G_train = nx.read_edgelist(trainfile)

    # ===== Embedding generation (only if needed) =====
    embedding_map = None

    if 'emb' in features:
        def make_pecanpy_graph(chosen_mode):
            if chosen_mode == 'PreComp':
                return n2v.PreComp(p=p, q=q, workers=workers, verbose=verbose)
            elif chosen_mode == 'SparseOTF':
                return n2v.SparseOTF(p=p, q=q, workers=workers, verbose=verbose)
            elif chosen_mode == 'DenseOTF':
                return n2v.DenseOTF(p=p, q=q, workers=workers, verbose=verbose)
            else:
                raise ValueError(f"Unknown pecanpy mode: {chosen_mode}")

        tried_modes = [mode]
        if mode != 'PreComp':
            tried_modes.append('PreComp')
        if mode not in ['SparseOTF', 'DenseOTF']:
            tried_modes.append('DenseOTF')

        last_exception = None
        for candidate_mode in tried_modes:
            try:
                g = make_pecanpy_graph(candidate_mode)
                g.read_edg('train.txt', weighted=False,
                           directed=False, delimiter=' ')
                if candidate_mode == 'PreComp':
                    g.preprocess_transition_probs()

                embeddings = g.embed(
                    dim=dim, num_walks=num_walks,
                    walk_length=walk_length, window_size=window_size,
                    epochs=epochs, verbose=verbose
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
    train_pos_edges = list(G_train.edges())

    X_train_pos, kept_pos = build_feature_matrix(
        train_pos_edges, G, features, embedding_map, operator)
    X_train_neg, kept_neg = build_feature_matrix(
        train_non_edges, G, features, embedding_map, operator)

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
        test_edges, G, features, embedding_map, operator)
    X_test_neg, _ = build_feature_matrix(
        test_non_edges, G, features, embedding_map, operator)

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
