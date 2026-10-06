import networkx as nx
import numpy as np
import pandas as pd
from pecanpy import pecanpy as n2v
import random
from sklearn.metrics import confusion_matrix, roc_auc_score
from tqdm.auto import tqdm
import geopandas as gpd
from shapely.geometry import Point
from infomap import Infomap
from scipy.spatial.distance import jensenshannon
from scipy.sparse import csr_matrix
import statsmodels.api as sm
import pickle
import psutil
import os
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from pathlib import Path
from pandas.api.types import union_categoricals


def _log_mem(label):
    """Diagnostic checkpoint: process RSS + system-wide available memory, in GiB.
    Only for tracking down where a run's memory peaks -- not used in any calculation."""
    rss = psutil.Process(os.getpid()).memory_info().rss / 1e9
    avail = psutil.virtual_memory().available / 1e9
    print(
        f"[mem] {label}: process RSS={rss:.2f} GiB, system available={avail:.2f} GiB", flush=True)


def _node_codes(cols):
    """Map node-label columns onto one shared integer index.

    Goes through the categorical codes when available so that millions of
    label strings are never materialized (this runs while X_train is resident).
    Returns (list of code arrays, n_nodes).
    """
    cats = [c.cat.categories if hasattr(c, 'cat') else pd.Index(pd.unique(c))
            for c in cols]
    nodes = cats[0]
    for c in cats[1:]:
        nodes = nodes.union(c)
    out = []
    for c in cols:
        if hasattr(c, 'cat'):
            out.append(nodes.get_indexer(c.cat.categories)
                       [c.cat.codes.to_numpy()])
        else:
            out.append(nodes.get_indexer(c))
    return out, len(nodes)


def _dyadic_cov(res, node_a, node_b, n_nodes, chunk_size=200_000):
    """Dyadic-robust sandwich covariance (Fafchamps-Gubert; Aronow-Samii-Assenova).

    Any two rows sharing an endpoint are treated as correlated. Two-way
    clustering on (NODE_A, NODE_B) can't express this: the graph is undirected,
    so a node sits in either column and cross-column pairs get counted as
    independent.

    With s_d = (y_d - p_d) x_d the score of row d, and G_i the score sum over
    rows touching node i:

        meat = sum_i G_i G_i^T - sum_d s_d s_d^T

    sum_i G_i G_i^T weights each row pair by its number of shared nodes; two
    distinct dyads share at most one, and a dyad shares two with itself, so
    subtracting the row-wise outer products removes the diagonal double count.
    Assumes one row per unordered pair.

    Chunked because the full n x k score matrix is ~7 GiB at this scale.
    """
    X, y = res.model.exog, res.model.endog
    n, k = X.shape
    p = res.predict()
    G = np.zeros((n_nodes, k))
    S2 = np.zeros((k, k))
    for lo in range(0, n, chunk_size):
        hi = min(lo + chunk_size, n)
        s = (y[lo:hi] - p[lo:hi])[:, None] * X[lo:hi]
        S2 += s.T @ s
        m = hi - lo
        rows = np.arange(m)
        # (m x n_nodes) incidence: one entry per endpoint, so each row's score
        # lands in both of its nodes' sums
        inc = csr_matrix((np.ones(2 * m),
                          (np.concatenate([rows, rows]),
                           np.concatenate([node_a[lo:hi], node_b[lo:hi]]))),
                         shape=(m, n_nodes))
        G += inc.T @ s
    bread = np.asarray(res.normalized_cov_params)
    return bread @ (G.T @ G - S2) @ bread


def _chunked_logit_hessian(self, params, chunk_size=500_000):
    """Drop-in, numerically equivalent replacement for
    statsmodels.discrete.discrete_model.Logit.hessian.

    The original does `-np.dot(L*(1-L)*X.T, X)`, which broadcasts a length-n
    vector against the (p, n) transpose of the full exog matrix, materializing
    a full new (p, n) array before the matmul. At millions of rows that's
    another full-size copy of X on top of everything else already resident --
    confirmed (via _log_mem) to be exactly what was crashing the kernel with
    an unrecoverable OOM, since statsmodels' base LikelihoodModel.fit() always
    calls .hessian() once after optimization (regardless of solver) to get the
    covariance matrix for standard errors/p-values/CIs, unless skip_hessian=True
    -- which isn't usable here since we need those stats for the regression
    tables.

    X.T @ diag(w) @ X is a sum over rows, so it can be accumulated in chunks
    with peak memory bounded by one chunk (chunk_size x p) instead of the
    full (p x n). Same result, just computed without the giant intermediate.
    """
    X = self.exog
    L = self.predict(params)
    n, p = X.shape
    _log_mem(f"entered _chunked_logit_hessian (n={n}, p={p})")
    hess = np.zeros((p, p), dtype=X.dtype)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        w = L[start:end] * (1 - L[start:end])
        Xc = X[start:end]
        hess -= (Xc * w[:, None]).T.dot(Xc)
    _log_mem("finished _chunked_logit_hessian")
    return hess


def _chunked_binary_predict(self, params, exog=None, which="mean", linear=None,
                            offset=None, chunk_size=1_000_000):
    """Drop-in replacement for BinaryModel.predict (used by Logit).

    The original does one call: `linpred = np.dot(exog, params) + offset`, a
    single BLAS matrix-vector product over the *entire* exog matrix at once.
    On this machine, at this network's scale (~7.77M x 130), that single call
    reliably kills the kernel process outright -- no Python exception, no
    MemoryError, just gone (confirmed via per-call _log_mem tracing: dies
    between "predict call start" and its own first line of work, before any
    of our own code after it ever runs). Since `predict()` is called on every
    single loglike/score evaluation during optimization, this makes the model
    entirely unfittable at full scale regardless of solver, check_rank, or
    hessian chunking.

    A plain matrix-vector product decomposes additively over row-chunks, so
    this produces numerically identical output -- it's just N smaller BLAS
    calls concatenated instead of one giant one. Whatever the underlying
    issue is (this was never fully root-caused -- isolated microbenchmarks of
    plain np.dot were themselves too flaky to pin down consistently, but the
    real pipeline consistently died at this exact call), chunking sidesteps
    it entirely.
    """
    if linear is not None:
        if linear is True:
            which = "linear"
    if offset is None and exog is None and hasattr(self, 'offset'):
        offset = self.offset
    elif offset is None:
        offset = 0.
    if exog is None:
        exog = self.exog

    n = exog.shape[0]
    linpred = np.empty(n, dtype=exog.dtype)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        linpred[start:end] = np.dot(exog[start:end], params) + offset

    if which == "mean":
        return self.cdf(linpred)
    elif which == "linear":
        return linpred
    elif which == "var":
        mu = self.cdf(linpred)
        return mu * (1 - mu)
    else:
        raise ValueError('Only `which` is "mean", "linear" or "var" are'
                         ' available.')


sm.Logit.predict = _chunked_binary_predict


sm.Logit.hessian = _chunked_logit_hessian

# ===================================================================
# EMBEDDING STORE
# ===================================================================


class EmbeddingMap:
    """Memory-efficient node embedding store.

    Takes embedding matrix and node list outputted by pecanpy and converts to
    float32 matrix plus node:row index dict. filters out None entries in the node
    list so that the matrix is smaller than the input embeddings array.

    Allows for fancy indexing without creating an unnecessary memory-intensive
    copy in float64 (which pecanpy doesn't make or need) via .rows() method.
    """

    __slots__ = ('matrix', 'idx_of')

    def __init__(self, matrix, idx_of):
        self.matrix = np.ascontiguousarray(matrix, dtype=np.float32)
        self.idx_of = idx_of

    @classmethod
    def from_pecanpy(cls, nodes, embeddings):
        """Build from pecanpy's node list and embedding matrix (skips None)."""
        idx_of = {}
        keep_rows = []
        for i, node_id in enumerate(nodes):
            if node_id is None:
                continue
            idx_of[str(node_id)] = len(keep_rows)
            keep_rows.append(i)
        matrix = np.asarray(embeddings, dtype=np.float32)[keep_rows]
        return cls(matrix, idx_of)

    def __contains__(self, key):
        return key in self.idx_of

    def __getitem__(self, key):
        return self.matrix[self.idx_of[key]]

    def __len__(self):
        return len(self.idx_of)

    def keys(self):
        return self.idx_of.keys()

    def rows(self, keys):
        """Return a (len(keys), dim) float32 array for the given node ids."""
        return self.matrix[[self.idx_of[k] for k in keys]]


# ===================================================================
# GRAPH LOADING FUNCTION
# ===================================================================


def load(fpath, compress=False):
    fpath = str(fpath)
    if fpath.endswith('.pkl'):
        with open(fpath, 'rb') as f:
            df = pickle.load(f)

    elif 'csv' in fpath:  # checks for both .csv and .csv.gz
        # downcast for better performance
        _dtypes = {
            'NODE_A': 'category', 'NODE_B': 'category',
            'N_COVISITS': 'float32', 'DIST_KM_MIN': 'float32',
            'DIST_KM_MEAN': 'float32', 'N_UIDS_A': 'float32',
            'N_POIS_A': 'float32', 'N_VISITS_A': 'float32',
            'N_UIDS_B': 'float32', 'N_POIS_B': 'float32',
            'N_VISITS_B': 'float32', 'DEP': 'float32'
        }
        df = pd.read_csv(fpath, dtype=_dtypes)

    # optional log-compression
    if compress:
        for col in ['DEP', 'N_COVISITS']:
            scale_factor = 1.0 / np.median(df[col])
            df[col] = np.log1p(df[col] * scale_factor)

    G = nx.from_pandas_edgelist(
        df,
        source='NODE_A',
        target='NODE_B',
        edge_attr=['SELF_LOOP', 'DIST_KM_MIN',
                   'DIST_KM_MEAN', 'N_COVISITS', 'DEP'],
    )

    # --- bulk assign node attrs ---

    # source
    cols_a = ['NODE_A', 'N_UIDS_A', 'N_POIS_A', 'NODE_A_COORDS', 'N_VISITS_A']
    df_a = df[cols_a].rename(columns={
        'NODE_A': 'node', 'N_UIDS_A': 'N_UIDS', 'N_POIS_A': 'N_POIS',
        'NODE_A_COORDS': 'COORDS_ARR', 'N_VISITS_A': 'N_VISITS'
    })

    # target
    cols_b = ['NODE_B', 'N_UIDS_B', 'N_POIS_B', 'NODE_B_COORDS', 'N_VISITS_B']
    df_b = df[cols_b].rename(columns={
        'NODE_B': 'node', 'N_UIDS_B': 'N_UIDS', 'N_POIS_B': 'N_POIS',
        'NODE_B_COORDS': 'COORDS_ARR', 'N_VISITS_B': 'N_VISITS'
    })

    node_df = pd.concat([df_a, df_b], ignore_index=True)
    node_df = node_df.drop_duplicates(
        subset=['node'], keep='first').set_index('node')

    node_attrs = node_df.to_dict(orient='index')
    nx.set_node_attributes(G, node_attrs)

    print(f"Nodes: {G.number_of_nodes()}")
    print(f"Edges: {G.number_of_edges()}")
    return G

# ================================================================
# DISTANCE-CONTROLLED SAMPLING
# ================================================================


def distribution_finder(G, dist_type, n_bins):
    '''
    Finds distance distribution by binning and counting number of occurrences per bin.

    Returns distribution as pd.Series indexed by bin, as well as dict mapping nodes
    to their respective intervals.
    '''
    # --- helper: binning ---
    def get_binned_dist(data_dict, bin_edges):

        # convert dict to series. index = node id or edge tuple, value = attribute
        s = pd.Series(data_dict).dropna()

        # cut the data into bins. this returns Interval objects
        binned = pd.cut(s, bins=bin_edges, include_lowest=True)

        # the distribution is the count of edges in each Interval
        distr = binned.value_counts().sort_index()

        # group by the bin intervals and extract the ids as a set
        # this creates an interval:nodes dict
        elements_by_bin = s.groupby(binned, observed=False).apply(
            lambda x: set(x.index)).to_dict()

        return distr, elements_by_bin

    # --- applying function ---

    if dist_type == 'mean':
        dist_dict = nx.get_edge_attributes(G, 'DIST_KM_MEAN')
    elif dist_type == 'min':
        dist_dict = nx.get_edge_attributes(G, 'DIST_KM_MIN')
    elif dist_type == 'poi_level':
        dist_dict = nx.get_edge_attributes(G, 'DIST_KM')
    else:
        raise SystemExit(
            'Error: invalid distance type (from distribution_finder)')

    dist_values = [v for v in dist_dict.values() if v is not None]
    if dist_values:
        max_d = max(dist_values)
        log_bins = np.concatenate(([0], np.geomspace(0.01, max_d, n_bins)))
    else:
        raise SystemExit(
            'Error: no distance values found (from distribution_finder)')

    distribution, element_set = get_binned_dist(
        {k: v for k, v in dist_dict.items() if v is not None}, log_bins)

    return distribution, element_set


def dist_controlled_sampler(G, distr, total_count, avoid=None, batch_size=2_000_000):
    def _coord(attrs, *keys):
        for k in keys:
            v = attrs.get(k)
            if v is not None:
                return float(v)
        return 0.0

    nodes = list(G.nodes())
    n = len(nodes)
    node_to_idx = {nd: i for i, nd in enumerate(nodes)}

    # agg nodes (tract+category buckets) carry a COORDS_ARR of member-POI
    # lat/lons instead of a single latitude/longitude; use the bucket
    # centroid as a representative point for candidate-distance binning
    # (the target distribution itself, DIST_KM_MEAN, is the exact
    # full-pairwise mean -- centroid distance is an approximation used
    # only to steer sampling toward the right bin, not a modeled feature)
    if not nx.get_node_attributes(G, 'latitude') and nx.get_node_attributes(G, 'COORDS_ARR'):
        lat = np.array([np.mean(G.nodes[nd]['COORDS_ARR'][:, 0])
                        if G.nodes[nd].get('COORDS_ARR') is not None and len(G.nodes[nd]['COORDS_ARR']) else 0.0
                        for nd in nodes], dtype=np.float64)
        lng = np.array([np.mean(G.nodes[nd]['COORDS_ARR'][:, 1])
                        if G.nodes[nd].get('COORDS_ARR') is not None and len(G.nodes[nd]['COORDS_ARR']) else 0.0
                        for nd in nodes], dtype=np.float64)
    else:
        lat = np.array([_coord(G.nodes[nd], 'latitude')
                       for nd in nodes], dtype=np.float64)
        lng = np.array([_coord(G.nodes[nd], 'longitude')
                       for nd in nodes], dtype=np.float64)

    # make sure you're not sampling existing edges
    # integer-keyed edge set for faster hashing than string tuples
    existing_edges_int = set()
    for u, v in G.edges():
        ui, vi = node_to_idx[u], node_to_idx[v]
        existing_edges_int.add((ui, vi) if ui < vi else (vi, ui))
    if avoid:
        for u, v in avoid:
            ui, vi = node_to_idx[u], node_to_idx[v]
            existing_edges_int.add((ui, vi) if ui < vi else (vi, ui))

    bin_intervals = list(distr.index)
    n_bins = len(bin_intervals)
    bin_edges = np.array([bin_intervals[0].left] +
                         [iv.right for iv in bin_intervals])

    total_in_distr = distr.sum()
    bin_quotas = np.array([
        int(np.round((c / total_in_distr) * total_count)) for c in distr.values
    ], dtype=int)

    bin_results = [[] for _ in range(n_bins)]
    bin_filled = np.zeros(n_bins, dtype=int)
    prev_filled = -1
    stall_rounds = 0

    with tqdm(total=total_count, desc='Sampling non-edges (fast)', unit='edge', leave=False) as pbar:
        while bin_filled.sum() < total_count:
            # bin_filled is a zero-array mirroring bin_quotas to be incremented and evaluated against it
            # this part repeats after each loop through the bins as long as the bins haven't been filled
            # to the requested amount
            still_needed = np.maximum(bin_quotas - bin_filled, 0)
            if still_needed.sum() == 0:
                break

            # counts the number of rounds with no added samples
            cur_filled = int(bin_filled.sum())
            if cur_filled == prev_filled:
                stall_rounds += 1
                if stall_rounds >= 5:
                    break
            else:
                stall_rounds = 0
            prev_filled = cur_filled

            # create candidate pairs by elementwise combination from
            # two 1d arrays containing random sequences of node indices
            # batch_size determines how large they are
            ui = np.random.randint(0, n, batch_size)
            vi = np.random.randint(0, n, batch_size)
            mask = ui != vi
            ui, vi = ui[mask], vi[mask]

            # compute vectorized haversine
            lat_u = np.radians(lat[ui])
            lat_v = np.radians(lat[vi])
            lng_u = np.radians(lng[ui])
            lng_v = np.radians(lng[vi])
            dlat = lat_v - lat_u
            dlng = lng_v - lng_u
            a = np.sin(dlat / 2) ** 2 + np.cos(lat_u) * \
                np.cos(lat_v) * np.sin(dlng / 2) ** 2
            dist = 6371.0088 * 2 * \
                np.arctan2(np.sqrt(np.clip(a, 0.0, 1.0)),
                           np.sqrt(np.clip(1.0 - a, 0.0, 1.0)))

            # array of bin indices matching distances
            # in_range filters out edges that randomly ended up with distances > n
            # (these would be mapped to bin_edges + 1)
            b_idx = np.digitize(dist, bins=bin_edges) - 1
            in_range = (b_idx < n_bins)

            for b in range(n_bins):
                need = still_needed[b]
                if need <= 0:
                    continue
                # skip if no more candidates
                candidates = np.where(in_range & (b_idx == b))[0]
                if len(candidates) == 0:
                    continue
                np.random.shuffle(candidates)
                added = 0
                for k in candidates:
                    if added >= need:
                        break
                    u_i, v_i = int(ui[k]), int(vi[k])
                    edge_int = (u_i, v_i) if u_i < v_i else (v_i, u_i)
                    if edge_int not in existing_edges_int:
                        existing_edges_int.add(edge_int)
                        bin_results[b].append((nodes[u_i], nodes[v_i]))
                        bin_filled[b] += 1
                        added += 1
                        pbar.update(1)

    for b in range(n_bins):
        if bin_filled[b] < bin_quotas[b]:
            iv = bin_intervals[b]
            print(
                f"Warning: Could not fulfill quota for bin [{iv.left:.2f}, {iv.right:.2f}]. Got {bin_filled[b]}/{bin_quotas[b]}.")

    # return the raw edges as a list (bins have served their purpose)
    return [edge for bucket in bin_results for edge in bucket]

# ====================================================================
# PREPARE_DATA
# ====================================================================


def prepare_data(
    _path, logistic=False, test_frac=0.5, seed=None, compress=True, weight=None, metadata=False, write=True,
    trainfile='data/train.txt'
):
    """
    1) splits data into train and test sets
    2) writes training graph for node2vec

    Parameters:
    _path (str): Path to the graph file. Must be readable as edgelist.
    test_frac (float, optional): Fraction of edges to use for testing. Default is 0.5.
    seed (int, optional): Seed for reproducibility.
    compress (int, optional): Option to log-transform weights when creating training graph

    If neg_path, returns:
        file : file consisting of the positive training graph as an edgelist. saved to 'trainfile'
        pd.DataFrame : negative training samples
        pd.DataFrame : positive testing samples
        pd.DataFrame : negative testing samples
    Otherwise returns the same but no negatives.
    """
    # TODO: add functionality as needed
    # - extra graph attrs
    # - logistic path if using again

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    capweight = 'DEP' if weight == 'dep' else (
        'N_COVISITS' if weight == 'cov' else None)

    # --- read in data ---

    def fast_read_csv(fpath, _dtypes):
        headers = pd.read_csv(fpath, nrows=0).columns
        _dtypes = {col: dt for col, dt in _dtypes.items() if col in headers}
        return pd.read_csv(fpath, dtype=_dtypes)

    # downcast for better performance
    _dtypes = {
        'NODE_A': 'category', 'NODE_B': 'category',
        'N_COVISITS': 'float32', 'DIST_KM_MIN': 'float32',
        'DIST_KM_MEAN': 'float32', 'DIST_KM_MEDIAN': 'float32',
        'DIST_KM_CENTROID': 'float32', 'N_UIDS_A': 'float32',
        'N_POIS_A': 'float32', 'N_VISITS_A': 'float32',
        'N_UIDS_B': 'float32', 'N_POIS_B': 'float32',
        'N_VISITS_B': 'float32', 'DEP': 'float32'
    }
    edgelist = fast_read_csv(_path, _dtypes)
    if compress:
        edgelist['LOG_'+capweight] = np.log1p(edgelist[capweight])

    print('Converting to nx.Graph for MST...')
    # add index as col
    # TODO: add to attr dict as needed
    # attrs = {
    #     ""
    # }
    edgelist = edgelist.reset_index()
    if weight:
        G = nx.from_pandas_edgelist(
            edgelist, 'NODE_A', 'NODE_B', edge_attr=['index', capweight])
    else:
        G = nx.from_pandas_edgelist(
            edgelist, 'NODE_A', 'NODE_B', edge_attr='index')

    # use nx.Graph to find mst edges, then go back to df and sample while excluding them
    edges = {tuple(sorted(e)) for e in G.edges()}
    mst_idx = [d['index'] for _, _, d in
               nx.maximum_spanning_tree(G, weight=capweight if weight else None).edges(data=True)]
    num_removable = len(edges) - len(mst_idx)
    test_num = (test_frac) * len(edgelist)
    if num_removable < test_num:
        raise SystemExit(
            f'Not enough removable edges. Test fraction is too high.\n({test_num} req / {len(num_removable)} available.)')

    if not logistic:
        # apply back to df
        edgelist_safe = edgelist.drop(index=mst_idx)
        # sample the same amount from edgelist_safe as would be needed to sample frac from original
        test = edgelist_safe.sample(
            n=int(round(test_num)), random_state=seed)
        train = edgelist.drop(test.index)

        # --- build + write training graph ---

        if write:
            # TODO: dont forget to include attr dict here as well
            if weight:
                G_train = nx.from_pandas_edgelist(
                    train, 'NODE_A', 'NODE_B', edge_attr='LOG_'+capweight if compress else capweight)
                for u, v, data in G_train.edges(data=True):
                    if compress:
                        wgt_val = data.pop('LOG_'+capweight)
                        data['weight'] = wgt_val
                    else:
                        wgt_val = data.pop(capweight)
                        data['weight'] = wgt_val
            else:
                G_train = nx.from_pandas_edgelist(train, 'NODE_A', 'NODE_B')

            if nx.is_empty(G_train):
                raise SystemExit("Error: Empty training graph.")

            # saving training graph
            trainfile = Path(trainfile)
            if not trainfile.is_file():
                with open(trainfile, 'w') as f:
                    for u, v, d in G_train.edges(data=True):
                        f.write(f"{u}\t{v}\t{d.get('weight', 1.0)}\n")
                print(
                    f"Wrote training graph: {G_train.number_of_nodes()} nodes, {G_train.number_of_edges()} edges")
            else:
                _overwrite = input("Trainfile already exists. Overwrite? Y/N")
                if (_overwrite == 'y' or _overwrite == 'Y'):
                    with open(trainfile, 'w') as f:
                        for u, v, d in G_train.edges(data=True):
                            f.write(f"{u}\t{v}\t{d.get('weight', 1.0)}\n")
                    print(
                        f"Wrote training graph: {G_train.number_of_nodes()} nodes, {G_train.number_of_edges()} edges")
                else:
                    print('Overwrite skipped; using existing training graph.')
    else:
        print("Error: Logistic branch incomplete")
        return None

    if metadata:
        print("Error: Metadata branch incomplete")
        return None

    return {'train': train, 'test': test}

    # TODO: add functionality for logistic if needed
    # return G_train, train_neg, test_pos, test_neg

# ====================================================================


def rowwise_cosine(a, b):
    """Cosine similarity of each paired row of a and b -> shape (n_pairs, 1).

    Note sklearn's cosine_similarity(a, b) would build the full (n_a, n_b)
    cross-product matrix; we only ever want its diagonal.
    """
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    num = np.einsum('ij,ij->i', a, b)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    sim = np.divide(num, den, out=np.zeros_like(num), where=den != 0)
    return sim[:, None]


BINARY_OPERATORS = {
    'avg': lambda a, b: np.mean([a, b], axis=0),
    'hadamard': lambda a, b: np.multiply(a, b),
    'w-l1': lambda a, b: np.abs(np.subtract(a, b)),
    'w-l2': lambda a, b: np.square(np.subtract(a, b)),
    # cosine similarity operates on whole vectors (not element-wise),
    # so it collapses the embedding block to a single column
    'cosine': rowwise_cosine
}


# ===================================================================
# METADATA FEATURE INCLUSION
# ===================================================================


def tract_to_area(nodes, shapefile_paths=(
        'data/geo/tl_2019_25_tract/tl_2019_25_tract.shp',
        'data/geo/tl_2019_33_tract/tl_2019_33_tract.shp')):
    """map POIs to their tracts' land area."""

    mass_file, nh_file = shapefile_paths
    mass_tracts = gpd.read_file(
        mass_file, columns=['GEOID', 'ALAND'], ignore_geometry=True)
    mass_tracts['TRACT'] = mass_tracts['GEOID'].str[:11]
    nh_tracts = gpd.read_file(
        nh_file, columns=['GEOID', 'ALAND'], ignore_geometry=True)
    nh_tracts['TRACT'] = nh_tracts['GEOID'].str[:11]
    all_tract_areas = gpd.GeoDataFrame(
        pd.concat([mass_tracts, nh_tracts]).drop(
            columns='GEOID').set_index('TRACT')
    )
    return nodes.join(all_tract_areas)


def tract_log_densities(pos_edges, agg):
    """log(POIs per km² of land) for every tract in the positive edges."""
    if agg:
        # agg nodes are tract+category buckets, so sum their POI counts per tract
        nodes = pd.concat([
            pos_edges[['NODE_A', 'N_POIS_A']].set_axis(
                ['NODE', 'N_POIS'], axis=1),
            pos_edges[['NODE_B', 'N_POIS_B']].set_axis(
                ['NODE', 'N_POIS'], axis=1),
        ], ignore_index=True).drop_duplicates(subset='NODE')
        nodes['TRACT'] = nodes['NODE'].astype(str).str.split('_', n=1).str[0]
        tract_counts = nodes.groupby('TRACT')['N_POIS'].sum().rename('count')
    else:
        nodes = edgelist_to_nodelist(pos_edges, ['ORIGIN', 'DESTINATION'])
        tract_counts = nodes['GEOID'].astype(
            str).str[:11].value_counts().rename('count')

    count_to_area = tract_to_area(tract_counts.to_frame())
    # water-only tracts have ALAND 0; tracts missing from the shapefile come back NaN
    area_km2 = count_to_area['ALAND'].replace(0, np.nan) / 1e6
    log_densities = np.log(count_to_area['count'] / area_km2)

    bad = ~np.isfinite(log_densities)
    if bad.any():
        print(f"Notice: {bad.sum()} tracts with no land area or no shapefile match "
              f"({list(log_densities.index[bad][:5])}); filled with the median log density.")
        log_densities[bad] = log_densities[~bad].median()
    return log_densities


def node_to_comm(G):
    # TODO: before you use this again have it add a step to convert df to nx.Graph
    im = Infomap("--num-trials 20")
    im_to_nx = im.add_networkx_graph(G)
    print("Running Infomap...")
    im.run()
    print("Done.")

    for node_id, module_id in im.modules:
        G.nodes[im_to_nx[node_id]]['community'] = module_id

    print(
        f"Assigned {len(set(nx.get_node_attributes(G, 'community').values()))} communities")


def add_outside_metadata(G):
    df_temporal = pd.read_csv('data/metadata/temporal_sig.csv.gz')
    df_income = pd.read_csv(
        'data/metadata/income_sig.csv', compression='gzip')

    # remove and renormalize nulls for income
    income_cols = ['1', '2', '3', '4']
    df_income[income_cols] = df_income[income_cols].div(
        df_income[income_cols].sum(axis=1), axis=0).fillna(0.25)
    df_income.drop(columns='NULL', inplace=True)

    # combine dfs
    df_temporal.set_index('POI_ID', inplace=True)
    df_income.set_index('POI_ID', inplace=True)
    df_features = df_temporal.join(df_income)

    # add to node attrs
    for poi_id in G.nodes():
        if poi_id in df_features.index:
            G.nodes[poi_id]['time_dist'] = df_features.loc[poi_id,
                                                           ['0', '6', '12', '18']].values
            G.nodes[poi_id]['inc_dist'] = df_features.loc[poi_id,
                                                          ['1', '2', '3', '4']].values


def haversine(lat_u, lng_u, lat_v, lng_v):
    # convert coordinates to radians
    lat_u_rad, lng_u_rad = np.radians(lat_u), np.radians(lng_u)
    lat_v_rad, lng_v_rad = np.radians(lat_v), np.radians(lng_v)
    # calculate haversine on the 1D arrays
    dlat = lat_v_rad - lat_u_rad
    dlng = lng_v_rad - lng_u_rad
    a = np.sin(dlat / 2.0)**2 + np.cos(lat_u_rad) * \
        np.cos(lat_v_rad) * np.sin(dlng / 2.0)**2
    # clip 'a' to [0, 1] to prevent NaN errors in sqrt from floating-point precision limits
    a = np.clip(a, 0.0, 1.0)
    return 6371.0088 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def edge_distances_km(G, edges):
    """Haversine distance between endpoints of each (u, v) edge in `edges`. For use in strength task."""
    if not edges:
        return np.array([], dtype=np.float32)
    if not nx.get_node_attributes(G, 'latitude'):
        # real graph edges so use precomputed distance
        dist_attr = 'DIST_KM_MEAN' if nx.get_edge_attributes(
            G, 'DIST_KM_MEAN') else 'DIST_KM'
        return np.array([G[u][v].get(dist_attr, 0.0) for u, v in edges], dtype=np.float32)
    lat_u = np.radians(np.array(
        [G.nodes[u].get('latitude') or 0.0 for u, _ in edges], dtype=np.float32))
    lng_u = np.radians(np.array(
        [G.nodes[u].get('longitude') or 0.0 for u, _ in edges], dtype=np.float32))
    lat_v = np.radians(np.array(
        [G.nodes[v].get('latitude') or 0.0 for _, v in edges], dtype=np.float32))
    lng_v = np.radians(np.array(
        [G.nodes[v].get('longitude') or 0.0 for _, v in edges], dtype=np.float32))
    a = np.sin((lat_v - lat_u) / 2) ** 2 + np.cos(lat_u) * \
        np.cos(lat_v) * np.sin((lng_v - lng_u) / 2) ** 2
    a = np.clip(a, 0.0, 1.0)
    return 6371.0088 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))


def edgelist_to_nodelist(df, names):
    name_1, name_2 = names
    orig_cols = [name_1] + [c for c in df.columns if c.endswith('_'+name_1)]
    dest_cols = [name_2] + [c for c in df.columns if c.endswith('_'+name_2)]
    orig = df[orig_cols]
    dest = df[dest_cols]

    def de_edgify(df, indicator):
        col_names = df.columns.tolist()
        mapper = {n: n.replace(indicator, '') for n in col_names}
        indicator2 = indicator.removeprefix('_')
        mapper |= {indicator2: 'NODE'}
        return df.rename(mapper, axis=1)

    orig = de_edgify(orig, '_'+name_1)
    dest = de_edgify(dest, '_'+name_2)

    return pd.concat([orig, dest], ignore_index=True).drop_duplicates(subset='NODE')


def build_feature_matrix(
        edges, features, embedding_map, operator='hadamard', cat_threshold=1, agg=False,
        z_score_stats=None, cats=None, log_densities=None
):
    """
    TODO: update this
    Build a feature matrix for a list of node pairs.

    Each row corresponds to one edge (u, v). The columns are determined
    by `features`, which is a list that can contain any combination of:

        'emb'       – binary-operator output on node2vec embeddings (128-d by default)
        'dist'      - log geographic distance in km. possible suffixes: _median, _mean, _min, _centroid
        'latlon'    - adds 4 features: avg lat/lon of node A + avg lat/lon of node B.
        'cat'       – (N_edges, N_interactions) matrix with binary corresponding to interaction type
        'catsame'   - simplified same/different category feature for baseline comparison
        'cbg'       - binary for same/different census-block group
        'comm'      - binary for same/different infomap community
        'ls'        - concatenated embeddings from endpoint categories constructed from word2vec on activity sequences
        'time'      - JS divergence of 6hr-window temporal distribution of visits for endpoint POIs
        'income'    - JS divergence of income-quartile distribution of endpoint POI visitors

    Parameters
    ----------
    edges : list of (u, v) tuples
    G : nx.Graph with node attributes (latitude, longitude, poi_type, total_visits)
    features : list of str
    embedding_map : dict  (required only when 'emb' in features)
    operator : str  (which binary operator to use for embeddings)
    agg : bool (flag for agg network type)

    Returns
    -------
    X : np.ndarray of shape (n_edges, n_features)
    kept_indices : list of int – indices into `edges` that were actually kept
        (some may be dropped if embeddings are missing)
    feature_names : list of str – one name per column of X, in column order
    """
    op_fn = BINARY_OPERATORS[operator]

    # unzip the edges into two parallel arrays of origins and destinations
    if agg:
        U, V = edges['NODE_A'], edges['NODE_B']
    else:
        U, V = edges['ORIGIN'], edges['DESTINATION']

    kept_indices = list(range(len(edges)))

    feature_blocks = []
    feature_names = []

    # vectorized embeddings
    if any(f in features for f in ('emb', 'cosine')):
        # extract to 2D arrays: shape (N_pairs, dim). one for each endpoint
        # fancy-index the packed matrix when available; fall back to per-key lookup when not
        if hasattr(embedding_map, 'rows'):
            emb_u = embedding_map.rows(U)
            emb_v = embedding_map.rows(V)
        else:
            emb_u = np.asarray([embedding_map[u] for u in U], dtype=np.float32)
            emb_v = np.asarray([embedding_map[v] for v in V], dtype=np.float32)

        # binary operator applies to both arrays simultaneously
        emb_feat = op_fn(emb_u, emb_v)
        if 'cosine' in features and operator != 'cosine':
            cos_feat = rowwise_cosine(emb_u, emb_v)
            feature_blocks.append(cos_feat)
            feature_names.append('emb_cosine_0')
        feature_blocks.append(emb_feat)
        feature_names.extend(
            f'emb_{operator}_{i}' for i in range(emb_feat.shape[1]))

    if agg:
        # vectorized geographic distance
        if 'dist_mean' in features:
            feature_blocks.append(np.nan_to_num(
                np.log1p(edges['DIST_KM_MEAN'].to_numpy().reshape(-1, 1))))
            feature_names.append('log_dist_mean')
        if 'dist_min' in features:
            feature_blocks.append(np.nan_to_num(
                np.log1p(edges['DIST_KM_MIN'].to_numpy().reshape(-1, 1))))
            feature_names.append('log_dist_min')
        if 'dist_median' in features:
            feature_blocks.append(
                np.nan_to_num(np.log1p(edges['DIST_KM_MEDIAN'].to_numpy().reshape(-1, 1))))
            feature_names.append('log_dist_median')
        if 'dist_centroid' in features:
            feature_blocks.append(
                np.nan_to_num(np.log1p(edges['DIST_KM_CENTROID'].to_numpy().reshape(-1, 1))))
            feature_names.append('log_dist_centroid')

        if 'latlon' in features:
            # replace coords with z-scored versions to account for the boston
            # metro being a small proportion of the whole earth
            coord_cols = edges[['LAT_A', 'LNG_A', 'LAT_B', 'LNG_B']]
            lats = coord_cols.iloc[:, [0, 2]]
            lons = coord_cols.iloc[:, [1, 3]]
            lat_means, lat_stds, lon_means, lon_stds = z_score_stats
            std_coords = coord_cols.assign(
                **{col: lambda x, c=col: (x[c]-lat_means) / lat_stds for col in lats},
                **{col: lambda x, c=col: (x[c]-lon_means) / lon_stds for col in lons},
            )
            feature_blocks.append(
                std_coords.to_numpy())
            feature_names.extend(['LAT_A', 'LNG_A', 'LAT_B', 'LNG_B'])

        if 'comm' in features:
            # TODO: fill in if using comm
            pass

        if 'time' in features:
            # TODO: probably something like add 4 cols to df for each node's distribution then add new col for js div
            pass

        if 'income' in features:
            pass

        if 'ls' in features:
            pass

        if 'cat' in features:
            # count encoding with sum-to-zero rule
            cu = pd.Categorical(
                edges['NODE_A'].astype(str).str.split('_').str[1], categories=cats).codes
            cv = pd.Categorical(
                edges['NODE_B'].astype(str).str.split('_').str[1], categories=cats).codes
            # -1 = category not in cats
            assert (cu >= 0).all() and (cv >= 0).all()

            n = len(edges)
            rows = np.arange(n)
            counts = np.zeros((n, len(cats)), dtype=np.int8)
            counts[rows, cu] += 1
            # same-category pairs end up with a 2
            counts[rows, cv] += 1

            ref = 0  # index of the column to drop
            cat_feat = np.delete(counts, ref, axis=1) - \
                counts[:, [ref]]   # (n, 19), values -2..2
            feature_blocks.append(cat_feat)
            feature_names.extend(
                f'cat_{c.lower()}' for i, c in enumerate(cats) if i != ref)

        if 'density' in features:
            tract_u = np.asarray(edges['NODE_A'].astype(str).str.split('_').str[0])
            tract_v = np.asarray(edges['NODE_B'].astype(str).str.split('_').str[0])
            density_u = log_densities.reindex(
                tract_u).to_numpy().reshape(-1, 1)
            density_v = log_densities.reindex(
                tract_v).to_numpy().reshape(-1, 1)
            assert np.isfinite(density_u).all() and np.isfinite(density_v).all(), \
                'density lookup hit a tract not in log_densities'
            feature_blocks.extend([np.minimum(density_u, density_v), np.maximum(density_u, density_v)])
            feature_names.extend(['log_density_min', 'log_density_max'])

    else:
        if 'dist' in features:
            feature_blocks.append(np.nan_to_num(
                np.log1p(edges['DIST_KM'].to_numpy().reshape(-1, 1))))
            feature_names.append('log_dist_km')

        if 'cat' in features:
            # same as above
            cu = pd.Categorical(
                edges['TAXONOMY_ORIGIN'], categories=cats).codes
            cv = pd.Categorical(
                edges['TAXONOMY_DESTINATION'], categories=cats).codes
            assert (cu >= 0).all() and (cv >= 0).all()

            n = len(edges)
            rows = np.arange(n)
            counts = np.zeros((n, len(cats)), dtype=np.int8)
            counts[rows, cu] += 1
            counts[rows, cv] += 1

            ref = 0
            cat_feat = np.delete(counts, ref, axis=1) - \
                counts[:, [ref]]
            feature_blocks.append(cat_feat)
            feature_names.extend(
                f'cat_{c.lower()}' for i, c in enumerate(cats) if i != ref)

        if 'density' in features:
            # same as above
            tract_u = np.asarray(edges['GEOID_ORIGIN'].astype(str).str[:11])
            tract_v = np.asarray(
                edges['GEOID_DESTINATION'].astype(str).str[:11])
            density_u = log_densities.reindex(
                tract_u).to_numpy().reshape(-1, 1)
            density_v = log_densities.reindex(
                tract_v).to_numpy().reshape(-1, 1)
            assert np.isfinite(density_u).all() and np.isfinite(density_v).all(), \
                'density lookup hit a tract not in log_densities'
            feature_blocks.extend([np.minimum(density_u, density_v), np.maximum(density_u, density_v)])
            feature_names.extend(['log_density_min', 'log_density_max'])

    X = np.hstack(feature_blocks).astype(np.float32)

    return X, kept_indices, feature_names


# ===================================================================
# RUN_PIPELINE
# ====================================================================

'''
not needed:
- G
- likely some of the kwargs
'''


def run_pipeline_logistic(trainfile, train_edges, train_non_edges, test_edges, test_non_edges, features=['emb'],
                          standardize=False, mode='SparseOTF', operator='hadamard', agg=False, **kwargs):
    """
    Run the link prediction pipeline with flexible feature composition. Features controlled by `features` list.

    Parameters
    ----------
    trainfile : str
        Path to the training graph edgelist file.
    train_non_edges : list
        Negative training edges.
    test_edges : list
        Positive test edges.
    test_non_edges : list
        Negative test edges.
    G : nx.Graph
        The *original* graph with node attributes (latitude, longitude,
        poi_type, total_visits). Required when features includes anything
        other than 'emb'.
    features : list of str or 'all'
        Which features to include. Default ['emb']. If 'all' then includes all features.
    mode : str
        PecanPy walk mode. Default 'SparseOTF'.
    standardize : bool
        Flag for z-score standardization of predictor variables.
    operator : str
        Binary operator for embeddings. Default 'hadamard'.
    **kwargs :
        Hyperparameter settings forwarded to PecanPy / Word2Vec. Also allows for seed.
        strength : bool, optional
            If set, additionally trains a second "strength" classifier over
            positive edges only: strong (DEP above the given quantile) vs weak.
            0.5 gives a median split. The threshold is fit on train positives.
            Reuses the same features/embeddings as the link task. Default None.
        strength_dist_control : bool, optional
            Only meaningful with strength=True. Distance-matches the strong/weak
            classes by binning positive edges by geographic distance and keeping
            min(#strong, #weak) per bin, so distance carries no marginal signal
            about strength. If a class empties out after matching,
            str_auc is nan. Default False.
        dyadic_se : bool, optional
            Swap the link model's nonrobust covariance for a dyadic-robust
            sandwich, so bse/pvalues/conf_int account for rows sharing an
            endpoint. Nonrobust SEs are badly anti-conservative here
            (~7x too small, 75% false-positive rate at nominal 5%).
            Default False.

    Returns
    -------
    link_auc : float
        AUC score for the specified feature/operator combination.
    link_model : sm.Logit()
        Trained model for later analysis.
    embedding_map : dict or None
        Node embeddings (only populated when 'emb' in features).
    str_auc : float
        AUC score for the specified feature/operator combination.
    str_model : sm.Logit()
        Trained model for later analysis.
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
    strength = kwargs.get('strength', None)
    strength_dist_control = kwargs.get('strength_dist_control', True)
    dyadic_se = kwargs.get('dyadic_se', False)

    # seed
    seed = kwargs.get('seed', None)
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)

    # TODO: update
    if not agg:
        if features == 'all' or features == ['all']:
            features = ['emb', 'dist', 'cat',
                        'comm', 'time', 'income', 'density']
    else:
        if features == 'all' or features == ['all']:
            features = ['emb', 'dist', 'comm', 'time', 'income']

    # ensure none of the other 3 sets contain nodes not in train_pos
    def node_set(df):
        if agg:
            df = df[['NODE_A', 'NODE_B']].dropna()
            return set(pd.unique(df.values.ravel()))
        else:
            df = df[['ORIGIN', 'DESTINATION']].dropna()
            return set(pd.unique(df.values.ravel()))

    train_pos_nodes = node_set(train_edges)
    missing = {
        'test_pos':  node_set(test_edges) - train_pos_nodes,
        'train_neg': node_set(train_non_edges) - train_pos_nodes,
        'test_neg':  node_set(test_non_edges) - train_pos_nodes,
    }
    if not all(not v for v in missing.values()):
        print(f'Error: positive training set is incomplete.')
        for k, v in missing.items():
            print(f'{k}: {len(v)} nodes absent from train_pos'
                  + (f' (e.g. {sorted(v)[:3]})' if v else ''))
        raise SystemExit

    # ensure training graph is fully connected
    G = nx.from_pandas_edgelist(train_edges, 'NODE_A', 'NODE_B')
    assert nx.is_connected(G), 'Error: disconnected training graph.'

    # ===== Embedding generation (only if needed) =====

    if any(f in features for f in ('emb', 'cosine')) and embedding_map is not None:
        print(f"Using precomputed embeddings: {len(embedding_map)} nodes")

    elif any(f in features for f in ('emb', 'cosine')):
        def make_pecanpy_graph(chosen_mode, w_bool):
            if chosen_mode == 'PreComp':
                return n2v.PreComp(p=p, q=q, workers=workers, verbose=verbose, extend=w_bool, random_state=seed)
            elif chosen_mode == 'SparseOTF':
                return n2v.SparseOTF(p=p, q=q, workers=workers, verbose=verbose, extend=w_bool, random_state=seed)
            elif chosen_mode == 'DenseOTF':
                return n2v.DenseOTF(p=p, q=q, workers=workers, verbose=verbose, extend=w_bool, random_state=seed)
            else:
                raise ValueError(f"Unknown pecanpy mode: {chosen_mode}")

        # set an order in which to try modes
        modes_to_try = [mode]
        if mode != 'PreComp':
            modes_to_try.append('PreComp')
        if mode not in ['SparseOTF', 'DenseOTF']:
            modes_to_try.append('DenseOTF')
        # PreComp alias_indptr overflows uint32 for large weighted graphs;
        # SparseOTF computes transition probs on-the-fly and avoids this
        # insert() puts it at the front of the queue if it isnt already
        if weighted and 'SparseOTF' not in modes_to_try:
            modes_to_try.insert(0, 'SparseOTF')

        last_exception = None
        for candidate_mode in modes_to_try:
            try:
                g = make_pecanpy_graph(candidate_mode, weighted)
                g.read_edg(trainfile, weighted=weighted,
                           directed=directed, delimiter='\t')
                if candidate_mode == 'PreComp':
                    g.preprocess_transition_probs()

                embeddings = g.embed(
                    dim=dim, num_walks=num_walks,
                    walk_length=walk_length, window_size=window_size,
                    epochs=epochs, verbose=verbose,
                )

                if candidate_mode != mode:
                    print(f"Notice: fell back to '{candidate_mode}'")
                break
            except Exception as e:
                print(f"Notice: pecanpy mode '{candidate_mode}' failed: {e}")
                last_exception = e
                continue
        else:
            raise RuntimeError(
                f"Pecanpy walk generation failed for all modes."
            ) from last_exception

        # convert to EmbeddingMap object
        embedding_map = EmbeddingMap.from_pecanpy(g.nodes, embeddings)

        print(f"Embeddings generated: {len(embedding_map)} nodes, dim={dim}")

    # ===== Assemble feature matrices =====

    if 'comm' in features:
        # TODO: insert fixed node_to_comm
        pass

    z_score_stats = None
    if 'latlon' in features:
        train_coord_cols = pd.concat([train_edges, train_non_edges])[
            ['LAT_A', 'LNG_A', 'LAT_B', 'LNG_B']]

        lats = train_coord_cols[['LAT_A', 'LAT_B']]
        lats = lats.stack().reset_index(drop=True)
        lngs = train_coord_cols[['LNG_A', 'LNG_B']]
        lngs = lngs.stack().reset_index(drop=True)

        lat_means = lats.mean()
        lat_stds = lats.std()
        lng_means = lngs.mean()
        lng_stds = lngs.std()

        z_score_stats = (lat_means, lat_stds, lng_means, lng_stds)

    cats = None
    if 'cat' in features:
        train_pairs = pd.concat([train_edges, train_non_edges])
        if agg:
            cats = sorted(
                pd.unique(pd.concat([train_pairs['NODE_A'].astype(str).str.split('_').str[1], train_pairs['NODE_B'].astype(str).str.split('_').str[1]])))
        else:
            cats = sorted(pd.unique(pd.concat(
                [train_pairs['TAXONOMY_ORIGIN'], train_pairs['TAXONOMY_DESTINATION']])))

    log_densities = None
    if 'density' in features:
        log_densities = tract_log_densities(train_edges, agg)

    # if not agg:
        # # TODO: remove and rework if using POI-level again
        # if ('cbg' in features or 'tract' in features):
        #     node_to_area(G)
        # if 'time' in features or 'income' in features:
        #     add_outside_metadata(G)

    _log_mem("before building train feature matrices")
    X_train_pos, keep_train_pos, feature_names = build_feature_matrix(
        train_edges, features, embedding_map, operator, cat_threshold, agg,
        z_score_stats, cats, log_densities)
    _log_mem("after X_train_pos built")
    X_train_neg, keep_train_neg, _ = build_feature_matrix(
        train_non_edges, features, embedding_map, operator, cat_threshold, agg,
        z_score_stats, cats, log_densities)
    _log_mem("after X_train_neg built")

    X_train = np.vstack([X_train_pos, X_train_neg])
    y_train = np.concatenate([
        np.ones(len(X_train_pos)),
        np.zeros(len(X_train_neg))
    ])

    if standardize:
        def standardizer(train_set):
            '''
            Bypasses StandardScaler float64 upcasting by z-scoring in place.
            Stats are accumulated in float64 for numerical stability, then cast back.
            '''
            # exclude dummy variables from standardization
            # (mask if vals are only in set of 0 and 1)
            dummies = np.isin(train_set, [0, 1]).all(axis=0)

            train_mean = train_set.mean(
                axis=0, dtype=np.float64).astype(np.float32)
            train_std = train_set.std(
                axis=0, dtype=np.float64).astype(np.float32)

            # identity for subtraction and division respectively
            # also ensure 0s dont enter into std dev for div by zero
            train_mean[dummies] = 0.0
            train_std[dummies] = 1.0
            train_std[train_std == 0] = 1.0
            train_set -= train_mean
            train_set /= train_std

            return train_set, train_mean, train_std
        X_train, train_mean, train_std = standardizer(X_train)

    # convert Xs to df for feature names and standardization gates
    X_train = pd.DataFrame(X_train, columns=feature_names)

    # remove originals to save memory if no longer needed
    if not strength:
        del X_train_pos, X_train_neg
    else:
        del X_train_neg

    print(
        f"Training matrix: {X_train.shape[0]} samples x {X_train.shape[1]} features")
    _log_mem("after training matrix assembled + originals freed")

    # ===== Train =====

    # add constant and fit model
    X_train = sm.add_constant(X_train)
    exog_names = ['const'] + feature_names

    X_train = X_train.to_numpy(dtype=np.float64)
    _log_mem(
        "after add_constant + consolidating to ndarray, right before Logit(...) construction")
    link_mod = sm.Logit(y_train, X_train, check_rank=False)
    link_mod.data.xnames = exog_names
    _log_mem("after Logit(...) constructed, right before .fit()")

    link_model = link_mod.fit(method='lbfgs', maxiter=200)
    _log_mem("after link_model.fit() returned")

    if dyadic_se:
        # node codes in X_train row order (train positives, then negatives)
        (a_pos, a_neg, b_pos, b_neg), n_nodes = _node_codes([
            train_edges['NODE_A'], train_non_edges['NODE_A'],
            train_edges['NODE_B'], train_non_edges['NODE_B']])
        ia = np.concatenate([a_pos[keep_train_pos], a_neg[keep_train_neg]])
        ib = np.concatenate([b_pos[keep_train_pos], b_neg[keep_train_neg]])
        tgt = getattr(link_model, '_results', link_model)
        # read the nonrobust SEs off normalized_cov_params rather than .bse:
        # .bse is cache_readonly, and touching it first would freeze the
        # nonrobust value in place and make the override below a no-op
        bse_plain = np.sqrt(np.diag(np.asarray(tgt.normalized_cov_params)))
        link_cov = _dyadic_cov(link_model, ia, ib, n_nodes)
        # cov_params() honors cov_params_default, so bse/pvalues/conf_int and
        # summary2 all pick this up; set it on _results, not the wrapper
        tgt.cov_params_default = link_cov
        tgt.cov_type = 'dyadic-robust'
        for _k in ('bse', 'tvalues', 'pvalues'):
            getattr(tgt, '_cache', {}).pop(_k, None)
        print(f"Dyadic-robust SEs over {n_nodes} node clusters: "
              f"median inflation x"
              f"{np.median(np.asarray(link_model.bse) / bse_plain):.2f}")
        _log_mem("after dyadic covariance")

    # print out description excluding embeddings but keep the header block
    # (pseudo R-squared, log-likelihood, convergence) which tables[1] alone drops
    if 'emb' in features:
        link_summary = link_model.summary2()
        emb_vec_features = [
            name for name in feature_names if name.startswith('emb_') and not 'cosine' in name]
        filt_summary = link_summary.tables[1].drop(index=emb_vec_features)
        print(link_summary.tables[0])
        print(filt_summary)
    else:
        print(link_model.summary2())

    # ===== Test =====

    X_test_pos, keep_test_pos, _ = build_feature_matrix(
        test_edges, features, embedding_map, operator, cat_threshold, agg,
        z_score_stats, cats, log_densities)
    X_test_neg, _, _ = build_feature_matrix(
        test_non_edges, features, embedding_map, operator, cat_threshold, agg,
        z_score_stats, cats, log_densities)

    X_test = np.vstack([X_test_pos, X_test_neg])
    X_test = sm.add_constant(X_test, has_constant='add')
    y_test = np.concatenate([
        np.ones(len(X_test_pos)),
        np.zeros(len(X_test_neg))
    ])

    if standardize:
        # z-score with same mean and std dev to avoid contaminating regression
        # with unaccounted-for differences
        X_test -= train_mean
        X_test /= train_std

    link_probs = link_model.predict(X_test)
    link_preds = (link_probs >= 0.5).astype(int)
    link_auc = roc_auc_score(y_test, link_probs)

    # create confusion matrix "in-house"
    link_cm = confusion_matrix(y_test, link_preds)

    # match predictions to edges
    pred_df = pd.concat([test_edges, test_non_edges],
                        join='inner', ignore_index=True)
    pred_df['LABEL'] = y_test
    pred_df['PROB'] = link_probs
    pred_df['PRED'] = link_preds
    # convert back to categories for lower memory usage
    for col in ('NODE_A', 'NODE_B'):
        pred_df[col] = union_categoricals(
            [test_edges[col], test_non_edges[col]])

    # --- report ---
    feature_label = '+'.join(features)
    op_label = f" ({operator})" if 'emb' in features else ""
    print(f"[{feature_label}{op_label}]  Link AUC = {link_auc:.4f}")

    # ===== strength head =====

    # TODO: rebuild this as well whenever we use it
    # if strength:
    #     # function to return dependencies of kept edges only
    #     def _dep(edges, keep):
    #         return np.array([G[u][v]['DEP'] for u, v in edges],
    #                         dtype=np.float64)[keep]

    #     # distance-controlled sampler (strength version)
    #     def _dist_matched_idx(dep, dist, thr):
    #         # same log bins as the link version, then keep min(#strong, #weak)
    #         strong = dep > thr
    #         if dist.max() > 0.01:
    #             edges_b = np.concatenate(
    #                 ([0], np.geomspace(0.01, dist.max(), 50)))
    #         else:
    #             edges_b = np.linspace(0, dist.max() + 1e-5, 50)
    #         # indexes the bin that each edge falls into
    #         b = np.digitize(dist, edges_b)
    #         keep = []
    #         for bin_id in np.unique(b):
    #             # strong and weak within bin (returns indices)
    #             in_bin = np.where(b == bin_id)[0]
    #             s = in_bin[strong[in_bin]]
    #             w = in_bin[~strong[in_bin]]
    #             k = min(len(s), len(w))
    #             if k == 0:
    #                 continue
    #             # select k random samples from within-bin subsets w/o replacement
    #             # (smaller one technically just gets copied but its fast enough
    #             # that the redundancy doesn't matter)
    #             keep.extend(np.random.choice(s, k, replace=False))
    #             keep.extend(np.random.choice(w, k, replace=False))
    #         return np.sort(np.array(keep, dtype=int))

    #     dep_train = _dep(train_edges, keep_train_pos)
    #     dep_test = _dep(test_edges, keep_test_pos)

    #     thr = np.quantile(dep_train, strength)
    #     y_str_train = (dep_train > thr).astype(int)
    #     y_str_test = (dep_test > thr).astype(int)

    #     if strength_dist_control:
    #         idx_tr = _dist_matched_idx(
    #             dep_train, edge_distances_km(
    #                 G, [train_edges[i] for i in keep_train_pos]), thr)
    #         idx_te = _dist_matched_idx(
    #             dep_test, edge_distances_km(
    #                 G, [test_edges[i] for i in keep_test_pos]), thr)
    #         # filter to sampled edges
    #         X_train_pos = X_train_pos[idx_tr]
    #         X_test_pos = X_test_pos[idx_te]
    #         y_str_train = y_str_train[idx_tr]
    #         y_str_test = y_str_test[idx_te]
    #         print(f"Distance-matched strength set: "
    #               f"train {len(idx_tr)}/{len(dep_train)}, "
    #               f"test {len(idx_te)}/{len(dep_test)} edges kept")
    #         if len(np.unique(y_str_test)) < 2 or len(np.unique(y_str_train)) < 2:
    #             print("Warning: a strength class vanished after distance "
    #                   "matching — cannot score. Returning link results.")
    #             return {'link_auc': link_auc,
    #                     'link_model': link_model,
    #                     'link_cm': link_cm,
    #                     'embedding_map': embedding_map}

    #     # same float32 in-place standardization as before
    #     X_train_pos = pd.DataFrame(X_train_pos, columns=feature_names)
    #     if standardize:
    #         X_train_pos, str_mean, str_std = standardizer(X_train_pos)
    #         X_test_pos -= str_mean
    #         X_test_pos /= str_std

    #     X_train_pos = sm.add_constant(X_train_pos)
    #     str_exog_names = ['const'] + feature_names
    #     X_train_pos = X_train_pos.to_numpy(dtype=np.float64)
    #     _log_mem(
    #         "strength head: after consolidating to ndarray, before Logit(...) construction")
    #     str_mod = sm.Logit(y_str_train, X_train_pos, check_rank=False)
    #     str_mod.data.xnames = str_exog_names
    #     _log_mem("strength head: after Logit(...) constructed, right before .fit()")
    #     str_model = str_mod.fit(method='lbfgs', maxiter=200)
    #     _log_mem("strength head: after str_model.fit() returned")

    #     if 'emb' in features:
    #         str_summary = str_model.summary2()
    #         filt_summary = str_summary.tables[1].drop(index=emb_vec_features)
    #         print(str_summary.tables[0])
    #         print(filt_summary)
    #     else:
    #         print(str_model.summary2())

    #     X_test_pos = sm.add_constant(X_test_pos, has_constant='add')
    #     str_probs = str_model.predict(X_test_pos)
    #     str_preds = (str_probs >= 0.5).astype(int)
    #     str_auc = roc_auc_score(y_str_test, str_probs)
    #     str_cm = confusion_matrix(y_str_test, str_preds)

    #     print(f"[{feature_label}{op_label}]"
    #           f"  Strength AUC = {str_auc:.4f}")

    #     return {'link_auc': link_auc,
    #             'link_model': link_model,
    #             'link_cm': link_cm,
    #             'embedding_map': embedding_map,
    #             'str_auc': str_auc,
    #             'str_model': str_model,
    #             'str_cm': str_cm}

    return {'auc': link_auc,
            'model': link_model,
            'cm': link_cm,
            'embedding_map': embedding_map,
            'pred_df': pred_df}


def run_pipeline_linear(trainfile, train, test, features, weight='cov', mode='SparseOTF', agg=True,
                        operator=None, embedding_map=None, standardize=False, compressed=True, **kwargs):
    '''
    1. embeddings
    2. features (y is covisit vals)
    3. train model
    4. return model + wtv else
    '''
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
    weighted = kwargs.get('weighted', False)
    directed = kwargs.get('directed', False)

    # seed
    seed = kwargs.get('seed', None)
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
    else:
        print("Notice: seed not passed.")

    capweight = 'DEP' if weight == 'dep' else (
        'N_COVISITS' if weight == 'cov' else None)

    # TODO: update
    if features == 'all' or features == ['all']:
        features = ['emb', 'dist', 'cat', 'cbg', 'comm', 'time', 'income']

    if any(f in features for f in ('emb', 'cosine')):
        assert operator, "Error: binary operator must be selected if using embeddings."

    # ensure training graph is fully connected
    G = nx.from_pandas_edgelist(train, 'NODE_A', 'NODE_B')
    assert nx.is_connected(G), 'Error: disconnected training graph.'

    # ===== Embedding generation (only if needed) =====

    if any(f in features for f in ('emb', 'cosine')) and embedding_map is not None:
        print(f"Using precomputed embeddings: {len(embedding_map)} nodes")

    elif any(f in features for f in ('emb', 'cosine')):
        def make_pecanpy_graph(chosen_mode, w_bool):
            if chosen_mode == 'PreComp':
                return n2v.PreComp(p=p, q=q, workers=workers, verbose=verbose, extend=w_bool, random_state=seed)
            elif chosen_mode == 'SparseOTF':
                return n2v.SparseOTF(p=p, q=q, workers=workers, verbose=verbose, extend=w_bool, random_state=seed)
            elif chosen_mode == 'DenseOTF':
                return n2v.DenseOTF(p=p, q=q, workers=workers, verbose=verbose, extend=w_bool, random_state=seed)
            else:
                raise ValueError(f"Unknown pecanpy mode: {chosen_mode}")

        # set an order in which to try modes
        modes_to_try = [mode]
        if mode != 'PreComp':
            modes_to_try.append('PreComp')
        if mode not in ['SparseOTF', 'DenseOTF']:
            modes_to_try.append('DenseOTF')
        # PreComp alias_indptr overflows uint32 for large weighted graphs;
        # SparseOTF computes transition probs on-the-fly and avoids this
        # insert() puts it at the front of the queue if it isnt already
        if weighted and 'SparseOTF' not in modes_to_try:
            modes_to_try.insert(0, 'SparseOTF')

        last_exception = None
        for candidate_mode in modes_to_try:
            try:
                g = make_pecanpy_graph(candidate_mode, weighted)
                g.read_edg(trainfile, weighted=weighted,
                           directed=directed, delimiter='\t')
                if candidate_mode == 'PreComp':
                    g.preprocess_transition_probs()

                embeddings = g.embed(
                    dim=dim, num_walks=num_walks,
                    walk_length=walk_length, window_size=window_size,
                    epochs=epochs, verbose=verbose,
                )

                if candidate_mode != mode:
                    print(f"Notice: fell back to '{candidate_mode}'")
                break
            except Exception as e:
                print(f"Notice: pecanpy mode '{candidate_mode}' failed: {e}")
                last_exception = e
                continue
        else:
            raise RuntimeError(
                f"Pecanpy walk generation failed for all modes."
            ) from last_exception

        # convert to EmbeddingMap object
        embedding_map = EmbeddingMap.from_pecanpy(g.nodes, embeddings)

        print(f"Embeddings generated: {len(embedding_map)} nodes, dim={dim}")

    # ===== assemble feature matrices =====

    z_score_stats = None
    if 'latlon' in features:
        train_coord_cols = train[['LAT_A', 'LNG_A', 'LAT_B', 'LNG_B']]

        lats = train_coord_cols[['LAT_A', 'LAT_B']]
        lats = lats.stack().reset_index(drop=True)
        lngs = train_coord_cols[['LNG_A', 'LNG_B']]
        lngs = lngs.stack().reset_index(drop=True)

        lat_means = lats.mean()
        lat_stds = lats.std()
        lng_means = lngs.mean()
        lng_stds = lngs.std()

        z_score_stats = (lat_means, lat_stds, lng_means, lng_stds)

    cats = None
    if 'cat' in features:
        if agg:
            cats = sorted(
                pd.unique(pd.concat([train['NODE_A'].astype(str).str.split('_').str[1], train['NODE_B'].astype(str).str.split('_').str[1]])))
        else:
            cats = sorted(pd.unique(pd.concat(
                [train['TAXONOMY_ORIGIN'], train['TAXONOMY_DESTINATION']])))

    log_densities = None
    if 'density' in features:
        log_densities = tract_log_densities(train, agg)

    X_train, _, feature_names = build_feature_matrix(
        train, features, embedding_map, operator,
        agg=agg, z_score_stats=z_score_stats, cats=cats, log_densities=log_densities)
    if compressed:
        y_train = train['LOG_'+capweight]
    else:
        y_train = train[capweight]

    X_test, _, _ = build_feature_matrix(
        test, features, embedding_map, operator,
        agg=agg, z_score_stats=z_score_stats, cats=cats, log_densities=log_densities)
    if compressed:
        y_test = test['LOG_'+capweight]
    else:
        y_test = test[capweight]

    if standardize:
        def standardizer(train_set):
            '''
            Bypasses StandardScaler float64 upcasting by z-scoring in place.
            Stats are accumulated in float64 for numerical stability, then cast back.
            '''
            # exclude dummy variables from standardization
            # (mask if vals are only in set of 0 and 1)
            dummies = np.isin(train_set, [0, 1]).all(axis=0)

            train_mean = train_set.mean(
                axis=0, dtype=np.float64).astype(np.float32)
            train_std = train_set.std(
                axis=0, dtype=np.float64).astype(np.float32)

            # identity for subtraction and division respectively
            # also ensure 0s dont enter into std dev for div by zero
            train_mean[dummies] = 0.0
            train_std[dummies] = 1.0
            train_std[train_std == 0] = 1.0
            train_set -= train_mean
            train_set /= train_std

            return train_set, train_mean, train_std
        X_train, train_mean, train_std = standardizer(X_train)
        X_test -= train_mean
        X_test /= train_std

    print(
        f"Training matrix: {X_train.shape[0]} samples x {X_train.shape[1]} features")

    # ===== Train =====

    # add constant and fit model
    X_train = sm.add_constant(X_train)
    exog_names = ['const'] + feature_names

    mod = sm.OLS(y_train, X_train)
    mod.data.xnames[:] = exog_names

    model = mod.fit(method='pinv', maxiter=200)

    # account for dependence between edges sharing nodes
    # node codes in X_train row order (build_feature_matrix keeps every row)
    (ia, ib), n_nodes = _node_codes([train['NODE_A'], train['NODE_B']])
    tgt = getattr(model, '_results', model)
    # read the nonrobust SEs off normalized_cov_params rather than .bse:
    # .bse is cache_readonly, and touching it first would freeze the
    # nonrobust value in place and make the override below a no-op.
    # unlike logit, OLS nonrobust cov is normalized_cov_params * sigma^2
    bse_plain = np.sqrt(
        np.diag(np.asarray(tgt.normalized_cov_params) * tgt.scale))
    dyad_cov = _dyadic_cov(model, ia, ib, n_nodes)
    # cov_params() honors cov_params_default, so bse/pvalues/conf_int and
    # summary2 all pick this up; set it on _results, not the wrapper.
    # sigma^2 is already in the residuals inside the meat, so no rescaling
    tgt.cov_params_default = dyad_cov
    tgt.cov_type = 'dyadic-robust'
    for _k in ('bse', 'tvalues', 'pvalues'):
        getattr(tgt, '_cache', {}).pop(_k, None)
    print(f"Dyadic-robust SEs over {n_nodes} node clusters: "
          f"median inflation x"
          f"{np.median(np.asarray(model.bse) / bse_plain):.2f}")
    _log_mem("after dyadic covariance")
    model_results = model

    # print out description excluding embeddings but keep the header block which tables[1] alone drops
    if 'emb' in features:
        summary = model_results.summary2()
        emb_vec_features = [
            name for name in feature_names if name.startswith('emb_') and not 'cosine' in name]
        filt_summary = summary.tables[1].drop(index=emb_vec_features)
        print(summary.tables[0])
        print(filt_summary)
    else:
        print(model_results.summary2())

    # === Test ===

    X_test = sm.add_constant(X_test)
    y_pred = model.predict(X_test)

    rmse = root_mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    test_r2 = r2_score(y_test, y_pred)

    print(f"Test RMSE: {rmse}")
    print(f"Test MAE: {mae}")
    print(f"Test R²:  {test_r2:.4f}")

    # match predictions and residuals to corresponding edges (drop redundant cols first)
    test = test.drop(columns=test.filter(like=capweight).columns)
    pred_df = test.assign(PRED=y_pred,
                          RESID=y_test.to_numpy() - y_pred)

    pred_results = {
        "rmse": rmse,
        "mae": mae,
        "test_r2": test_r2,
        "pred_df": pred_df
    }

    return {"model_results": model_results,
            "embedding_map": embedding_map,
            "pred_results": pred_results}
