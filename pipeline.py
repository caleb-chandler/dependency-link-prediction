import networkx as nx
import numpy as np
import pandas as pd
from pecanpy import pecanpy as n2v
from pecanpy.graph import AdjlstGraph
import random
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy.io import loadmat
import scipy.sparse as sp
from tqdm.auto import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


def prepare_data(fpath, ftype=None, frac=0.5):
    """
    Prepare data for link prediction pipeline.

    This function loads a graph from a file, splits it into training and testing sets,
    converts node labels to integers, saves the resulting training graph in the root folder, 
    and outputs negative training edges, negative test edges, and positive test edges.

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
        edges = list(G.edges())

        # guarantee fully connected training set by protecting spanning tree from removal
        spanning_tree_edges = set(nx.minimum_spanning_tree(G).edges())
        removable_edges = [e for e in edges if e not in spanning_tree_edges]

        # sample test edges from removable edges only (make sure count isnt larger thatn removable_edges)
        test_count = min(int(len(edges) * frac), len(removable_edges))
        test_edges = list(random.sample(removable_edges, test_count))
        train_edges = [e for e in edges if e not in test_edges]

        # build training graph
        G_train = nx.Graph()
        G_train.add_nodes_from(G.nodes())
        G_train.add_edges_from(train_edges)

        # function to restrict sampled non-edges for better performance on large datasets
        def sample_non_edges(G, count):
            non_edges = set()
            nodes = list(G.nodes())
            with tqdm(total=count, desc='sampling non-edges', unit='edge', leave=False) as pbar:
                while len(non_edges) < count:
                    u, v = sorted(random.sample(nodes, 2))
                    if not G.has_edge(u, v) and (u, v) not in non_edges:
                        non_edges.add((u, v))
                        pbar.update(1)
            return list(non_edges)

        # getting true negatives (overlap possible but very improbable for large datasets)
        test_non_edges = sample_non_edges(G, len(test_edges))
        train_non_edges = sample_non_edges(G, len(train_edges))

        return G_train, test_edges, test_non_edges, train_non_edges

    def load(fpath, ftype):
        if ftype == 'mat':
            edgelist = loadmat(fpath)
            adj_matrix = edgelist['network']
            G = nx.from_scipy_sparse_array(adj_matrix)
            print(f"Nodes: {G.number_of_nodes()}")
            print(f"Edges: {G.number_of_edges()}")
        else:
            G = nx.read_edgelist(fpath)
            print(f"Nodes: {G.number_of_nodes()}")
            print(f"Edges: {G.number_of_edges()}")
        return G

    G = load(fpath, ftype)

    # failsafe
    if G.number_of_nodes() == 0:
        print(f"Error: Graph loaded from {fpath} is entirely empty. Skipping.")
        return None

    # extracting lcc in case disconnected
    largest_cc = max(nx.connected_components(G), key=len)
    G = G.subgraph(largest_cc).copy()

    G = nx.convert_node_labels_to_integers(
        G, label_attribute='orig')

    G_train, test_edges, test_non_edges, train_non_edges = split(G)

    # error handling
    if nx.is_empty(G_train):
        print("Error: Empty training graph.")
        return None

    # saving training graph
    print("About to write training graph...")
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
    nx.Graph: Graph object created from file.
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
                workers=workers,
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

    return results, embedding_map, G_train
