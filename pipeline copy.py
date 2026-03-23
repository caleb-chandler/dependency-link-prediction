import networkx as nx
import numpy as np
import pandas as pd
from pecanpy import pecanpy as n2v
from pecanpy.graph import AdjlstGraph
import random
from sklearn.linear_model import LogisticRegression
import ast
import pickle
from sklearn.metrics import roc_auc_score
from scipy.io import loadmat
import scipy.sparse as sp
from gensim.models import Word2Vec
from tqdm.auto import tqdm


def run_pipeline(fpath, ftype=None, frac=0.5, p=1, q=1, mode='PreComp'):
    """
    Run the complete link prediction pipeline.

    This function loads a graph from a file, splits it into training and testing sets,
    generates node embeddings using node2vec, trains logistic regression classifiers
    on different edge embedding operators, and evaluates them on the test set.

    Parameters:
    fpath (str): Path to the graph file.
    ftype (str): Type of the file ('mat' for MATLAB format, else default to edgelist).
    frac (float, optional): Fraction of edges to use for testing. Default is 0.5.
    p (int, optional): Node2vec parameter p. Default is 1.
    q (int, optional): Node2vec parameter q. Default is 1.
    mode (str, optional): Mode for PecanPy walk-probability computation. Default is PreComp.

    Returns:
    dict: Dictionary with AUC scores for each operator ('avg', 'hadamard', 'w-l1', 'w-l2').
    """
    def split(G, frac=frac):
        edges = list(G.edges())

        # guarantee fully connected training set by protecting spanning tree from removal
        spanning_tree_edges = set(nx.minimum_spanning_tree(G).edges())
        removable_edges = [e for e in edges if e not in spanning_tree_edges]

        # sample test edges from removable edges only
        test_count = int(len(edges) * frac)
        test_edges = set(random.sample(removable_edges, test_count))
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
                    u, v = random.sample(nodes, 2)
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
    # extracting lcc in case disconnected
    largest_cc = max(nx.connected_components(G), key=len)
    G = G.subgraph(largest_cc).copy()
    G_train, test_edges, test_non_edges, train_non_edges = split(G)

    # error handling
    if G_train.number_of_nodes() == 0:
        print("Error: Empty training graph.")
        return None

    G_train = nx.convert_node_labels_to_integers(
        G_train, label_attribute='orig')
    # saving training graph
    print("About to write training graph...")
    nx.write_edgelist(G_train, 'train.txt', data=False)
    print(
        f"Wrote training graph: {G_train.number_of_nodes()} nodes, {G_train.number_of_edges()} edges")

    # ===== Generating Embeddings =====

    if mode == 'PreComp':
        g = n2v.PreComp(p=p, q=q, workers=4, verbose=True)
    elif mode == 'SparseOTF':
        g = n2v.SparseOTF(p=p, q=q, workers=4, verbose=True)
    elif mode == 'DenseOTF':
        g = n2v.DenseOTF(p=p, q=q, workers=4, verbose=True)

    # load graph from edgelist file
    g.read_edg('train.txt',
               weighted=False, directed=False, delimiter=' ')

    # precompute and save 2nd order transition probs
    g.preprocess_transition_probs()

    # generate random walks, which could then be used to train w2v
    walks = g.simulate_walks(num_walks=10, walk_length=80)

    # train embeddings using word2vec from gensim for more control
    model = Word2Vec(walks, vector_size=128, window=10, min_count=0,
                     sg=1, workers=4, epochs=1)

    # ===== Training =====

    embedding_map = {word: model.wv[word].tolist()
                     for word in model.wv.index_to_key}

    # function to turn node embeddings into labeled edge embeddings
    def create_training_data(edges, label, embedding_map=embedding_map):
        data = []
        for u, v in tqdm(edges, desc=f'create_training_data label={label}', unit='edge', leave=False):
            # get embeddings for both nodes
            emb_u = embedding_map.get(str(u))
            emb_v = embedding_map.get(str(v))

            # store as (node_tuple, embedding_tuple, label)
            data.append({
                'edge': (u, v),
                'embeddings': (emb_u, emb_v),
                'label': label
            })
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

            model = LogisticRegression(max_iter=1000)
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

    return results
