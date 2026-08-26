import networkx as nx
import numpy as np
import pandas as pd
from pecanpy import pecanpy as n2v
import random
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm
import geopandas as gpd
from shapely.geometry import Point
from infomap import Infomap
from scipy.spatial.distance import jensenshannon
import statsmodels.api as sm


# ===================================================================
# EMBEDDING STORE
# ===================================================================


class EmbeddingMap:
    """Memory-efficient node embedding store.

    Takes node list and embedding matrix outputted by pecanpy and converts to 
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
        scale_factor = 1.0 / np.median(df['DEP'])
        df['DEP'] = df['DEP'].apply(lambda x: np.log1p(x * scale_factor))
        scale_factor = 1.0 / np.median(df['N_COVISITS'])
        df['N_COVISITS'] = df['N_COVISITS'].apply(
            lambda x: np.log1p(x * scale_factor))

    G = nx.from_pandas_edgelist(
        df,
        source='NODE_A',
        target='NODE_B',
        edge_attr=['SELF_LOOP', 'DIST_KM_MIN',
                   'DIST_KM_MEAN', 'N_COVISITS', 'DEP'],
    )

    # assign node attrs
    def node_attr(a_col, b_col):
        return pd.concat([
            df.set_index('NODE_A')[a_col],
            df.set_index('NODE_B')[b_col],
        ]).groupby(level=0).first().to_dict()

    # apply returned Series per attr
    nx.set_node_attributes(G, node_attr('N_UIDS_A', 'N_UIDS_B'), 'N_UIDS')
    nx.set_node_attributes(G, node_attr('N_POIS_A', 'N_POIS_B'), 'N_POIS')
    nx.set_node_attributes(G, node_attr(
        'N_VISITS_A', 'N_VISITS_B'), 'N_VISITS')

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


def dist_controlled_sampler(G, distr, total_count, batch_size=2_000_000):
    def _coord(attrs, *keys):
        for k in keys:
            v = attrs.get(k)
            if v is not None:
                return float(v)
        return 0.0

    nodes = list(G.nodes())
    n = len(nodes)
    node_to_idx = {nd: i for i, nd in enumerate(nodes)}

    lat = np.array([_coord(G.nodes[nd], 'latitude')
                   for nd in nodes], dtype=np.float64)
    lng = np.array([_coord(G.nodes[nd], 'longitude')
                   for nd in nodes], dtype=np.float64)

    # integer-keyed edge set for faster hashing than string tuples
    edge_set_int = set()
    for u, v in G.edges():
        ui, vi = node_to_idx[u], node_to_idx[v]
        # set so that lower idx is always first
        edge_set_int.add((ui, vi) if ui < vi else (vi, ui))

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
                    if edge_int not in edge_set_int:
                        edge_set_int.add(edge_int)
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
    fpath, test_frac=0.5, seed=None, agg=True, compress=0, weight=None, meta=None,
    trainfile='data/train.txt', controlled=True, n_bins=50, dist_type='mean'
):
    """
    Prepare data for link prediction pipeline.

    This function loads a graph from a file, splits it into training and testing sets,
    saves the resulting training graph in the root folder, and outputs negative training edges,
    negative test edges, and positive test edges.

    Parameters:
    fpath (str): Path to the graph file. Must be readable as an edgelist.
    test_frac (float, optional): Fraction of edges to use for testing. Default is 0.5.
    seed (int, optional): Seed for reproducibility.

    Returns:
    nx.Graph : original graph if metadata needed for pipeline
    list : list of negative training samples
    list : list of positive testing samples
    list : list of negative testing samples
    """

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    def split(G, frac=test_frac):
        # load edges as sorted tuples for efficiency
        edges = {tuple(sorted(e)) for e in G.edges()}
        mst = {tuple(sorted(e))
               for e in nx.maximum_spanning_tree(G, weight=weight if weight else None)}

        removable_edges = list(edges - mst)
        test_num = int(len(edges) * frac)
        if len(removable_edges) < test_num:
            print(
                f'Not enough removable edges. Test fraction is too high.\n({test_num} req / {removable_edges} available.)')

        # fast set difference
        test_count = min(test_num, len(removable_edges))
        test_edges = random.sample(removable_edges, test_count)
        train_edges = list(edges - set(test_edges))

        # build training graph
        G_train = nx.Graph()
        G_train.add_nodes_from(G.nodes())

        # handle weights or not
        if not weight:
            G_train.add_edges_from(train_edges)
        elif not agg:
            if weight == 'dep':
                G_train.add_weighted_edges_from(
                    [(u, v, G[u][v]['DEP']) for u, v in train_edges])
            elif weight == 'cov':
                G_train.add_weighted_edges_from(
                    [(u, v, G[u][v]['N_COVISITS']) for u, v in train_edges])
            else:
                # fallback for invalid weights when agg is False
                print(
                    f'Value "{weight}" not recognized when agg=False. Falling back to unweighted.')
                G_train.add_edges_from(train_edges)
        else:
            if weight == 'cov':
                G_train.add_weighted_edges_from(
                    [(u, v, G[u][v]['N_COVISITS']) for u, v in train_edges])
            elif weight == 'dep':
                G_train.add_weighted_edges_from(
                    [(u, v, G[u][v]['DEP']) for u, v in train_edges])
            else:
                # fallback for invalid weights when agg is True
                print(
                    f'Value "{weight}" not recognized when agg=True. Falling back to unweighted.')
                G_train.add_edges_from(train_edges)

        # sampling
        if controlled:
            dist_bins, _ = distribution_finder(
                G, dist_type=dist_type, n_bins=n_bins)
            test_non_edges = dist_controlled_sampler(
                G, dist_bins, len(test_edges))
            train_non_edges = dist_controlled_sampler(
                G, dist_bins, len(train_edges))
        else:
            # function to sample non-edges randomly
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

            test_non_edges = sample_non_edges(G, len(test_edges))
            train_non_edges = sample_non_edges(G, len(train_edges))

        return G_train, test_edges, test_non_edges, train_non_edges

    # load in graph
    G = load(fpath, compress=compress)
    if G.number_of_nodes() == 0:
        raise SystemExit(
            f"Error: Graph loaded from {fpath} is entirely empty.")

    # extracting lcc in case disconnected
    largest_cc = max(nx.connected_components(G), key=len)
    G = G.subgraph(largest_cc).copy()

    G_train, test_edges, test_non_edges, train_non_edges = split(G)

    if nx.is_empty(G_train):
        raise SystemExit("Error: Empty training graph.")

    # saving training graph
    with open(trainfile, 'w') as f:
        for u, v, d in G_train.edges(data=True):
            f.write(f"{u}\t{v}\t{d.get('weight', 1.0)}\n")
    print(
        f"Wrote training graph: {G_train.number_of_nodes()} nodes, {G_train.number_of_edges()} edges")

    if meta:
        return G, train_non_edges, test_edges, test_non_edges
    else:
        return train_non_edges, test_edges, test_non_edges

# ====================================================================


BINARY_OPERATORS = {
    'avg': lambda a, b: np.mean([a, b], axis=0),
    'hadamard': lambda a, b: np.multiply(a, b),
    'w-l1': lambda a, b: np.abs(np.subtract(a, b)),
    'w-l2': lambda a, b: np.square(np.subtract(a, b)),
}


# ===================================================================
# METADATA FEATURE INCLUSION
# ===================================================================


def node_to_area(G, shapefile_path='data/geo/tl_2025_25_bg.shp'):
    """Add 'cbg' + 'tract' attribute to each node in G via spatial join."""
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
        geoid = row.get('GEOID')
        geoid = geoid if pd.notna(geoid) else None
        G.nodes[row['node_id']]['cbg'] = geoid if geoid else 'Unknown'
        G.nodes[row['node_id']]['tract'] = geoid[:11] if geoid else 'Unknown'


def node_to_comm(G):
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


def edge_distances_km(G, edges):
    """Haversine distance (km) between endpoints of each (u, v) edge in `edges`."""
    if not edges:
        return np.array([], dtype=np.float64)
    lat_u = np.radians(np.array(
        [G.nodes[u].get('latitude') or 0.0 for u, _ in edges], dtype=np.float64))
    lng_u = np.radians(np.array(
        [G.nodes[u].get('longitude') or 0.0 for u, _ in edges], dtype=np.float64))
    lat_v = np.radians(np.array(
        [G.nodes[v].get('latitude') or 0.0 for _, v in edges], dtype=np.float64))
    lng_v = np.radians(np.array(
        [G.nodes[v].get('longitude') or 0.0 for _, v in edges], dtype=np.float64))
    a = np.sin((lat_v - lat_u) / 2) ** 2 + np.cos(lat_u) * \
        np.cos(lat_v) * np.sin((lng_v - lng_u) / 2) ** 2
    a = np.clip(a, 0.0, 1.0)
    return 6371.0088 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))


def build_feature_matrix(
        edges, G, features, embedding_map, operator='hadamard', cat_threshold=1, agg=False
):
    """
    Build a feature matrix for a list of node pairs.

    Each row corresponds to one edge (u, v). The columns are determined
    by `features`, which is a list that can contain any combination of:

        'emb'       – binary-operator output on node2vec embeddings (128-d by default)
        'dist'      - log geographic distance in km  (1-d)
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

    # pre-filter edges missing embeddings to ensure matrix shapes align later
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
        return np.empty((0, 0)), [], []

    # unzip the list of tuples into two parallel lists of origins (U) and destinations (V)
    U, V = zip(*valid_edges)

    feature_blocks = []
    feature_names = []

    # vectorized embeddings
    if 'emb' in features:
        # extract to 2D arrays: shape (N, dim). fancy-index the packed matrix
        # when available; fall back to per-key lookup when not.
        if hasattr(embedding_map, 'rows'):
            emb_u = embedding_map.rows(U)
            emb_v = embedding_map.rows(V)
        else:
            emb_u = np.asarray([embedding_map[u] for u in U], dtype=np.float32)
            emb_v = np.asarray([embedding_map[v] for v in V], dtype=np.float32)

        # binary operator applies to whole array simultaneously
        emb_feat = op_fn(emb_u, emb_v)
        feature_blocks.append(emb_feat)
        feature_names.extend(
            f'emb_{operator}_{i}' for i in range(emb_feat.shape[1]))

    if not agg and any(x in features for x in ('cat', 'catsame', 'cbg')):
        if 'cat' in features:
            # count each undirected type-pair across all edges in G
            pair_counts = {}
            for eu, ev in G.edges():
                cu = G.nodes[eu].get('poi_type', 'Unknown')
                cv = G.nodes[ev].get('poi_type', 'Unknown')
                pair = tuple(sorted([cu, cv]))
                pair_counts[pair] = pair_counts.get(pair, 0) + 1

            # only pairs observed >= cat_threshold times, sorted for stable columns
            vocab = sorted(p for p, cnt in pair_counts.items()
                           if cnt >= cat_threshold)
            print(
                f'Number of kept pairs with threshold {cat_threshold}: {len(vocab)}/210 ({((len(vocab)/210)*100):.4f}%)')
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
            feature_names.extend(f'cat_{a}||{b}' for a, b in vocab)

        if 'catsame' in features:
            cat_u = np.array([G.nodes[u].get('poi_type', '') for u in U])
            cat_v = np.array([G.nodes[v].get('poi_type', '') for v in V])

            # boolean array comparison converted to floats: 1.0 for True, 0.0 for False
            cat_feat = (cat_u == cat_v).astype(float).reshape(-1, 1)
            feature_blocks.append(cat_feat)
            feature_names.append('catsame')

        if 'cbg' in features:
            cbg_u = np.array([G.nodes[u].get('cbg', 'Unknown') for u in U])
            cbg_v = np.array([G.nodes[v].get('cbg', 'Unknown') for v in V])
            cbg_feat = ((cbg_u == cbg_v) & (cbg_u != 'Unknown')
                        ).astype(float).reshape(-1, 1)
            feature_blocks.append(cbg_feat)
            feature_names.append('cbg_same')
    elif not agg:
        print(
            'Category and census-based features invalid for aggregated network. Skipping.')

    # vectorized geographic distance
    if 'dist' in features:
        if not agg:
            # convert NaNs to 0.0 while obtaining arrays of lat/lon for both u and v
            lat_u = np.nan_to_num(np.array(
                [G.nodes[u].get('latitude') for u in U], dtype=np.float64), nan=0.0)
            lng_u = np.nan_to_num(np.array(
                [G.nodes[u].get('longitude') for u in U], dtype=np.float64), nan=0.0)
            lat_v = np.nan_to_num(np.array(
                [G.nodes[v].get('latitude') for v in V], dtype=np.float64), nan=0.0)
            lng_v = np.nan_to_num(np.array(
                [G.nodes[v].get('longitude') for v in V], dtype=np.float64), nan=0.0)

            # convert all coordinates to radians at once
            lat_u_rad, lng_u_rad = np.radians(lat_u), np.radians(lng_u)
            lat_v_rad, lng_v_rad = np.radians(lat_v), np.radians(lng_v)

            # calculate haversine on the 1D arrays
            dlat = lat_v_rad - lat_u_rad
            dlng = lng_v_rad - lng_u_rad
            a = np.sin(dlat / 2.0)**2 + np.cos(lat_u_rad) * \
                np.cos(lat_v_rad) * np.sin(dlng / 2.0)**2

            # clip 'a' to [0, 1] to prevent NaN errors in sqrt from floating-point precision limits
            a = np.clip(a, 0.0, 1.0)

            dist_km = 6371.0088 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

            # log scale and reshape to (n_pairs, 1) column vector
            geo_feat = np.log1p(dist_km).reshape(-1, 1)
            feature_blocks.append(geo_feat)
            feature_names.append('log_dist_km')

    if 'comm' in features:
        comm_u = np.array([G.nodes[u].get('community', -1) for u in U])
        comm_v = np.array([G.nodes[v].get('community', -1) for v in V])
        comm_feat = ((comm_u == comm_v) & (comm_u != -1)
                     ).astype(float).reshape(-1, 1)
        feature_blocks.append(comm_feat)
        feature_names.append('comm_same')

    if 'time' in features:
        # uniform distribution for fallback
        default_distr = np.array([0.25, 0.25, 0.25, 0.25])
        time_u = np.array(
            [G.nodes[u].get('time_dist', default_distr) for u in U])
        time_v = np.array(
            [G.nodes[v].get('time_dist', default_distr) for v in V])
        js_dist = jensenshannon(time_u, time_v, axis=1)
        time_feat = (js_dist ** 2).reshape(-1, 1)
        feature_blocks.append(time_feat)
        feature_names.append('time_js_div')

    if 'income' in features:
        # uniform distribution for fallback
        default_distr = np.array([0.25, 0.25, 0.25, 0.25])
        inc_u = np.array(
            [G.nodes[u].get('inc_dist', default_distr) for u in U])
        inc_v = np.array(
            [G.nodes[v].get('inc_dist', default_distr) for v in V])
        js_dist = jensenshannon(inc_u, inc_v, axis=1)
        inc_feat = (js_dist ** 2).reshape(-1, 1)
        feature_blocks.append(inc_feat)
        feature_names.append('income_js_div')

    if 'ls' in features:
        pass

    X = np.hstack(feature_blocks).astype(np.float32)

    return X, kept_indices, feature_names


# ===================================================================
# RUN_PIPELINE
# ====================================================================


def run_pipeline(trainfile, train_non_edges, test_edges, test_non_edges, G=None, features=['emb'],
                 mode='PreComp', operator='hadamard', agg=False, **kwargs):
    """
    Run the link prediction pipeline with flexible feature composition. Features are controlled by the `features` list.

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
        PecanPy walk mode. Default 'PreComp'.
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
            strength_auc is nan. Default False.
        diagnostics : bool, optional
            If True, additionally returns a dict for building a regression
            table + error analysis: fitted model(s), feature names, test
            labels/predicted probabilities, and per-test-edge geographic
            distance (km). Keys: 'feature_names', 'link' (always), and
            'strength' (only when `strength` is set) — each of the latter two
            maps to {'model', 'y_test', 'probs', 'dist_km', 'edges'}
            (edges are the (u, v) node-id pairs aligned with y_test/probs/dist_km,
            for downstream analyses keyed on node attributes like category). 
            Default False.

    Returns
    -------
    auc : float
        AUC score for the specified feature/operator combination.
    embedding_map : dict or None
        Node embeddings (only populated when 'emb' in features).
    strength_auc : float
        Only returned when `strength` is set — AUC of the strong-vs-weak
        classifier on the positive test edges.
    diagnostics : dict
        Only returned when `diagnostics=True` (always the last return value).
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
    # allow passing in precomputed embeddings
    cat_threshold = kwargs.get('cat_threshold', 1)
    embedding_map = kwargs.get('embedding_map', None)
    # switch for weighted/directed version
    weighted = kwargs.get('weighted', False)
    directed = kwargs.get('directed', False)
    strength = kwargs.get('strength', None)
    strength_dist_control = kwargs.get('strength_dist_control', True)
    diagnostics = kwargs.get('diagnostics', False)

    # seed
    seed = kwargs.get('seed', None)
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    if not agg:
        if features == 'all' or features == ['all']:
            features = ['emb', 'dist', 'cat', 'cbg', 'comm', 'time', 'income']
    else:
        if features == 'all' or features == ['all']:
            features = ['emb', 'dist', 'comm', 'time', 'income']

    # ===== Validation =====
    needs_metadata = bool({'dist', 'cat', 'cbg', 'comm',
                          'time', 'income'} & set(features))
    if needs_metadata and G is None:
        raise ValueError(
            "Graph G with node attributes is required when features "
            f"include {[f for f in features if f != 'emb']}"
        )

    # converting training graph to nx.Graph object
    G_train = nx.read_edgelist(
        trainfile, data=[('weight', float)], delimiter='\t')

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

    # convert to list and sort for consistent indexing
    train_pos_edges = [tuple(sorted(e)) for e in G_train.edges()]

    if 'comm' in features:
        node_to_comm(G)

    if not agg:
        # TODO: remove and rework if using POI-level again
        if ('cbg' in features or 'tract' in features):
            node_to_area(G)
        if 'time' in features or 'income' in features:
            add_outside_metadata(G)

    X_train_pos, keep_train_pos, feature_names = build_feature_matrix(
        train_pos_edges, G, features, embedding_map, operator, cat_threshold, agg)
    X_train_neg, _, _ = build_feature_matrix(
        train_non_edges, G, features, embedding_map, operator, cat_threshold, agg)

    X_train = np.vstack([X_train_pos, X_train_neg])
    y_train = np.concatenate([
        np.ones(len(X_train_pos)),
        np.zeros(len(X_train_neg))
    ])

    # remove originals to save memory if no longer needed
    if not strength:
        del X_train_pos, X_train_neg
    else:
        del X_train_neg

    print(
        f"Training matrix: {X_train.shape[0]} samples x {X_train.shape[1]} features")

    # ===== Train =====
    # bypass StandardScaler float64 upcasting by z-scoring in place. stats are
    # accumulated in float64 for numerical stability, then cast back
    train_mean = X_train.mean(axis=0, dtype=np.float64).astype(np.float32)
    train_std = X_train.std(axis=0, dtype=np.float64).astype(np.float32)
    train_std[train_std == 0] = 1.0  # stand-in to avoid div by 0
    X_train -= train_mean
    X_train /= train_std

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # ===== Test =====
    X_test_pos, keep_test_pos, _ = build_feature_matrix(
        test_edges, G, features, embedding_map, operator, cat_threshold, agg)
    X_test_neg, keep_test_neg, _ = build_feature_matrix(
        test_non_edges, G, features, embedding_map, operator, cat_threshold, agg)

    X_test = np.vstack([X_test_pos, X_test_neg])
    y_test = np.concatenate([
        np.ones(len(X_test_pos)),
        np.zeros(len(X_test_neg))
    ])

    # apply the training z-score to the test matrix in place (same reason as above)
    X_test -= train_mean
    X_test /= train_std

    probs = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probs)

    diag = None
    if diagnostics:
        # the edge_indices returns are integer positions; this finds the actual edges from them
        test_pos_kept_edges = [test_edges[i] for i in keep_test_pos]
        test_neg_kept_edges = [test_non_edges[i] for i in keep_test_neg]
        if G is not None:
            test_dist_km = np.concatenate([
                edge_distances_km(G, test_pos_kept_edges),
                edge_distances_km(G, test_neg_kept_edges),
            ])
        else:
            test_dist_km = np.full(len(y_test), np.nan)
        diag = {
            'feature_names': feature_names,
            'link': {
                'model': model,
                'y_test': y_test,
                'probs': probs,
                'dist_km': test_dist_km,
                'edges': test_pos_kept_edges + test_neg_kept_edges,
            },
        }

    # ===== report =====
    feature_label = '+'.join(features)
    op_label = f" ({operator})" if 'emb' in features else ""
    print(f"[{feature_label}{op_label}]  AUC = {auc:.4f}")

    # ===== strength head =====
    if strength:
        # function to return dependencies of kept edges only
        def _dep(edges, keep):
            return np.array([G[u][v]['DEP'] for u, v in edges],
                            dtype=np.float64)[keep]

        # distance-controlled sampler (strength version)
        def _dist_matched_idx(dep, dist, thr):
            # same log bins as the link version, then keep min(#strong, #weak)
            strong = dep > thr
            if dist.max() > 0.01:
                edges_b = np.concatenate(
                    ([0], np.geomspace(0.01, dist.max(), 50)))
            else:
                edges_b = np.linspace(0, dist.max() + 1e-5, 50)
            # indexes the bin that each edge falls into
            b = np.digitize(dist, edges_b)
            keep = []
            for bin_id in np.unique(b):
                # strong and weak within bin (returns indices)
                in_bin = np.where(b == bin_id)[0]
                s = in_bin[strong[in_bin]]
                w = in_bin[~strong[in_bin]]
                k = min(len(s), len(w))
                if k == 0:
                    continue
                # select k random samples from within-bin subsets w/o replacement
                # (smaller one technically just gets copied but its fast enough
                # that the redundancy doesn't matter)
                keep.extend(np.random.choice(s, k, replace=False))
                keep.extend(np.random.choice(w, k, replace=False))
            return np.sort(np.array(keep, dtype=int))

        dep_train = _dep(train_pos_edges, keep_train_pos)
        dep_test = _dep(test_edges, keep_test_pos)

        thr = np.quantile(dep_train, strength)
        y_str_train = (dep_train > thr).astype(int)
        y_str_test = (dep_test > thr).astype(int)

        if strength_dist_control:
            idx_tr = _dist_matched_idx(
                dep_train, edge_distances_km(
                    G, [train_pos_edges[i] for i in keep_train_pos]), thr)
            idx_te = _dist_matched_idx(
                dep_test, edge_distances_km(
                    G, [test_edges[i] for i in keep_test_pos]), thr)
            # filter to sampled edges
            X_train_pos = X_train_pos[idx_tr]
            X_test_pos = X_test_pos[idx_te]
            y_str_train = y_str_train[idx_tr]
            y_str_test = y_str_test[idx_te]
            print(f"Distance-matched strength set: "
                  f"train {len(idx_tr)}/{len(dep_train)}, "
                  f"test {len(idx_te)}/{len(dep_test)} edges kept")
            if len(np.unique(y_str_test)) < 2 or len(np.unique(y_str_train)) < 2:
                print("Warning: a strength class vanished after distance "
                      "matching — cannot score. Returning nan.")
                return (auc, embedding_map, float('nan'), diag) if diagnostics \
                    else (auc, embedding_map, float('nan'))

        # same float32 in-place standardization as before
        str_mean = X_train_pos.mean(
            axis=0, dtype=np.float64).astype(np.float32)
        str_std = X_train_pos.std(axis=0, dtype=np.float64).astype(np.float32)
        str_std[str_std == 0] = 1.0
        X_train_pos -= str_mean
        X_train_pos /= str_std
        X_test_pos -= str_mean
        X_test_pos /= str_std

        str_model = LogisticRegression(max_iter=1000, class_weight='balanced')
        str_model.fit(X_train_pos, y_str_train)
        str_probs = str_model.predict_proba(X_test_pos)[:, 1]
        strength_auc = roc_auc_score(y_str_test, str_probs)

        print(f"[{feature_label}{op_label}]"
              f"  strength AUC = {strength_auc:.4f}")

        if diagnostics:
            if strength_dist_control:
                str_test_edges = [test_edges[keep_test_pos[i]] for i in idx_te]
            else:
                str_test_edges = [test_edges[i] for i in keep_test_pos]
            diag['strength'] = {
                'model': str_model,
                'y_test': y_str_test,
                'probs': str_probs,
                'dist_km': edge_distances_km(G, str_test_edges) if G is not None
                else np.full(len(y_str_test), np.nan),
                'edges': str_test_edges,
            }
            return auc, embedding_map, strength_auc, diag

        return auc, embedding_map, strength_auc

    return (auc, embedding_map, diag) if diagnostics else (auc, embedding_map)
