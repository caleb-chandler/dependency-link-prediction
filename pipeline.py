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
from collections import Counter


def load(fpath, ftype):
    if ftype == 'mat':
        edgelist = loadmat(fpath)
        adj_matrix = edgelist['network']
        G = nx.from_scipy_sparse_array(adj_matrix)
        print(f"Nodes: {G.number_of_nodes()}")
        print(f"Edges: {G.number_of_edges()}")
    else:
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
        print(f"Attribute check: {random.choice(list(G.nodes))}")

        print(f"Nodes: {G.number_of_nodes()}")
        print(f"Edges: {G.number_of_edges()}")
    return G


def distribution_finder(G):
    # --- HELPER 1: For continuous data (bins) ---
    def get_binned_dist(data_dict, bins=30):
        if not data_dict:
            return pd.Series(dtype='float64'), {}

        # Convert dict to series. Index = Node ID or Edge Tuple, Value = Attribute
        s = pd.Series(data_dict).dropna()

        # Calculate histogram and bins
        counts, edges = np.histogram(s, bins=bins)
        distr = pd.Series(counts, index=edges[:-1])

        # Cut the data into the exact same bins and group them
        binned = pd.cut(s, bins=edges, include_lowest=True)

        # Group by the bin intervals and extract the IDs as a set
        # This creates a dictionary: {Interval(...): {node1, node2, ...}}
        elements_by_bin = s.groupby(binned, observed=False).apply(
            lambda x: set(x.index)).to_dict()

        return distr, elements_by_bin

    # --- HELPER 2: For discrete/categorical data ---
    def get_discrete_dist(data_dict):
        if not data_dict:
            return pd.Series(dtype='float64'), {}

        s = pd.Series(data_dict)
        distr = s.value_counts().sort_index()

        # Group exactly by the value
        elements_by_val = s.groupby(s).apply(lambda x: set(x.index)).to_dict()

        return distr, elements_by_val

    # ---------------------------------------------------------

    # --- EDGES: Distance (Continuous) ---
    dist_dict = nx.get_edge_attributes(G, 'DIST_KM')
    dist_distr, dist_edges_set = get_binned_dist(dist_dict, bins=30)

    # --- EDGES: Covisits (Continuous/Discrete) ---
    cv_dict = nx.get_edge_attributes(G, 'N_COVISITS')
    cv_distr, cv_edges_set = get_binned_dist(cv_dict, bins=30)

    # --- NODES: Categorical (POI Type) ---
    type_dict = {u: data.get('poi_type', 'Unknown')
                 for u, data in G.nodes(data=True)}
    type_distr, type_nodes_set = get_discrete_dist(type_dict)

    # --- NODES: Unique Visits (Continuous) ---
    uv_dict = {u: data.get('unique_visits', 0)
               for u, data in G.nodes(data=True)}
    uv_distr, uv_nodes_set = get_binned_dist(uv_dict, bins=30)

    # --- NODES: Total Visits (Continuous) ---
    tv_dict = {u: data.get('total_visits', 0)
               for u, data in G.nodes(data=True)}
    tv_distr, tv_nodes_set = get_binned_dist(tv_dict, bins=30)

    # --- TOPOLOGY: Degree (Discrete) ---
    deg_dict = dict(G.degree())
    deg_distr, deg_nodes_set = get_discrete_dist(deg_dict)

    # Pack the results logically so it's easy to return
    distributions = (dist_distr, cv_distr, type_distr,
                     uv_distr, tv_distr, deg_distr)
    element_sets = (dist_edges_set, cv_edges_set, type_nodes_set,
                    uv_nodes_set, tv_nodes_set, deg_nodes_set)

    return distributions, element_sets


def prepare_data(fpath, ftype=None, frac=0.5):
    """
    Prepare data for link prediction pipeline.

    This function loads a graph from a file, splits it into training and testing sets,
    saves the resulting training graph in the root folder, and outputs negative training edges,
    negative test edges, and positive test edges.

    Parameters:
    fpath (str): Path to the graph file.
    ftype (str): Type of the file ('mat' for MATLAB format, else default to edgelist).
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
        dist_bins = sets[0]  # distance dict

        def sample_non_edges(G, bins_dict, total_count):
            non_edges = set()

            # calculate the total number of items across all bins to find proportions
            total_elements = sum(len(elements)
                                 for elements in bins_dict.values())

            with tqdm(total=total_count, desc='sampling non-edges', unit='edge', leave=False) as pbar:
                for bin, elements in bins_dict.items():
                    if len(elements) < 2:
                        continue  # need at least 2 nodes

                    # calculate how many non-edges to take from this specific bin
                    bin_proportion = len(elements) / total_elements
                    num_samples_for_this_bin = int(
                        bin_proportion * total_count)

                    # sample until this bin's quota is met
                    bin_samples = 0
                    # don't try to sample more than possible pairs in the bin
                    max_possible = (len(elements) * (len(elements) - 1)) // 2
                    quota = min(num_samples_for_this_bin, max_possible)

                    # convert set to list once for faster random.sample
                    node_list = list(elements)

                    while bin_samples < quota:
                        u, v = random.sample(node_list, 2)
                        if u > v:
                            u, v = v, u  # keep edges sorted (u, v)

                        if not G.has_edge(u, v) and (u, v) not in non_edges:
                            non_edges.add((u, v))
                            bin_samples += 1
                            pbar.update(1)

                            # stop if we hit the global count early due to rounding
                            if len(non_edges) >= total_count:
                                return list(non_edges)

            return list(non_edges)

        # getting true negatives (overlap possible but very improbable for large datasets)
        test_non_edges = sample_non_edges(G, dist_bins, len(test_edges))
        train_non_edges = sample_non_edges(G, dist_bins, len(train_edges))

        return G_train, test_edges, test_non_edges, train_non_edges

    G = load(fpath, ftype)

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


def run_pipeline(trainfile, train_non_edges, test_edges, test_non_edges, mode='PreComp', **kwargs):
    """
    Run the complete link prediction pipeline.

    This function takes a path to a training graph, a set of negative training edges, 
    and a set of positive/negative testing edges, and outputs a set of AUC scores 
    for four binary operators.

    Generates node embeddings using node2vec, trains logistic regression classifiers
    on different edge embedding operators, and evaluates them on the test set.

    Parameters:
    trainfile (str): Path to the training graph file.
    train_non_edges (list) : Negative training edges.
    test_edges (list) : Positive testing edges.
    test_non_edges (list) : Negative testing edges.
    mode (str, optional): Mode for PecanPy walk-probability computation. Default is PreComp.
    **kwargs: Hyperparameter settings (see PecanPy documentation).

    Returns:
    dict: Dictionary with AUC scores for each operator ('avg', 'hadamard', 'w-l1', 'w-l2').
    dict: Dictionary of embeddings for each node.
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
    min_count = kwargs.get('min_count', 0)
    sg = kwargs.get('sg', 1)
    epochs = kwargs.get('epochs', 1)

    # converting training graph to nx.Graph object
    G_train = nx.read_edgelist(trainfile)

    # ===== Generating Embeddings =====

    def make_pecanpy_graph(chosen_mode):
        if chosen_mode == 'PreComp':
            return n2v.PreComp(p=p, q=q, workers=workers, verbose=verbose)
        elif chosen_mode == 'SparseOTF':
            return n2v.SparseOTF(p=p, q=q, workers=workers, verbose=verbose)
        elif chosen_mode == 'DenseOTF':
            return n2v.DenseOTF(p=p, q=q, workers=workers, verbose=verbose)
        else:
            raise ValueError(f"Unknown pecanpy mode: {chosen_mode}")

    # get walk sequences + embeddings; fall back to other modes if needed
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

            # preprocess for PreComp
            if candidate_mode == 'PreComp':
                g.preprocess_transition_probs()

            # main call (combines walk simulation and embedding generation)
            embeddings = g.embed(
                dim=dim,
                num_walks=num_walks,
                walk_length=walk_length,
                window_size=window_size,
                epochs=epochs,
                verbose=verbose
            )

            if candidate_mode != mode:
                print(
                    f"Warning: requested mode '{mode}' failed, fell back to '{candidate_mode}'")
            break
        except Exception as e:
            print(f"Warning: pecanpy mode '{candidate_mode}' failed: {e}")
            last_exception = e
            continue
    else:
        raise RuntimeError(
            f"Pecanpy walk generation failed for all modes: {tried_modes}"
        ) from last_exception

    # ===== Training =====

    embedding_map = {}
    for i, node_id in enumerate(g.nodes):
        if node_id is None:
            continue
        try:
            # handle integer keys
            key = int(node_id)
        except (ValueError, TypeError):
            key = node_id

        embedding_map[key] = embeddings[i].tolist()

    print("Sample embeddings keys type:", type(
        next(iter(embedding_map.keys()))))
    print("example keys", list(embedding_map.keys())[:5])

    # function to turn node embeddings into labeled edge embeddings
    def create_training_data(edges, label, embedding_map=embedding_map):
        data = []
        missing = 0
        for u, v in tqdm(edges, desc=f'create_training_data label={label}', unit='edge', leave=False):
            emb_u = embedding_map.get(u)
            emb_v = embedding_map.get(v)
            if emb_u is None or emb_v is None:
                missing += 1
                continue
            data.append({'edge': (u, v), 'embeddings': (
                emb_u, emb_v), 'label': label})
        if missing:
            print(
                f"Warning: {missing} edge(s) skipped because embeddings missing")
        return data

    # generate data for both positive and negative samples
    pos_data = create_training_data(G_train.edges(), 1)
    neg_data = create_training_data(train_non_edges, 0)

    # combine and create final df
    training_set = pd.DataFrame(pos_data + neg_data)

    # shuffle and reset index
    training_set = training_set.sample(frac=1).reset_index(drop=True)

    def train(training_set):
        # applying binary operators
        training_set['avg'] = training_set['embeddings'].apply(
            lambda x: np.mean(x, axis=0))
        training_set['hadamard'] = training_set['embeddings'].apply(
            lambda x: np.multiply(x[0], x[1]))
        training_set['w-l1'] = training_set['embeddings'].apply(
            lambda x: np.abs(np.subtract(x[0], x[1])))
        training_set['w-l2'] = training_set['embeddings'].apply(
            lambda x: np.square(np.subtract(x[0], x[1])))

        # training classifier for each operator
        operators = ['avg', 'hadamard', 'w-l1', 'w-l2']
        models = []
        for op in tqdm(operators, desc='training models', unit='operator'):
            X = np.stack(training_set[op].values)
            y = np.stack(training_set['label'].values)

            model = make_pipeline(
                StandardScaler(), LogisticRegression(max_iter=1000))
            model.fit(X, y)

            models.append(model)

        # combined dict for lookup
        classifiers = dict(zip(operators, models))
        return classifiers

    classifiers = train(training_set)

    # ===== Testing =====

    # generate data for both positive and negative samples
    pos_data = create_training_data(test_edges, 1)
    neg_data = create_training_data(test_non_edges, 0)

    # combine and create final df
    testing_set = pd.DataFrame(pos_data + neg_data)

    # shuffle and reset index
    testing_set = testing_set.sample(frac=1).reset_index(drop=True)

    def test(classifiers, testing_set):
        # applying binary operators as before
        testing_set['avg'] = testing_set['embeddings'].apply(
            lambda x: np.mean(x, axis=0))
        testing_set['hadamard'] = testing_set['embeddings'].apply(
            lambda x: np.multiply(x[0], x[1]))
        testing_set['w-l1'] = testing_set['embeddings'].apply(
            lambda x: np.abs(np.subtract(x[0], x[1])))
        testing_set['w-l2'] = testing_set['embeddings'].apply(
            lambda x: np.square(np.subtract(x[0], x[1])))

        # getting probabilities
        prob_dict = {}
        for op, model in tqdm(classifiers.items(), desc='predicting probabilities', unit='operator'):
            X_test = np.stack(testing_set[op].values)
            probs = model.predict_proba(X_test)[:, 1]
            prob_dict[op] = probs

        # getting scores
        true_y = testing_set['label'].values
        auc_scores = {}
        for op in tqdm(prob_dict.keys(), desc='computing AUC', unit='operator'):
            prob = prob_dict[op]
            auc = roc_auc_score(true_y, prob)
            auc_scores[op] = auc

        return auc_scores

    results = test(classifiers, testing_set)

    for op, score in results.items():
        print(f"{op:10}: {score:.4f}")

    return results, embedding_map
