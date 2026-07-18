import networkx as nx
import numpy as np
import pandas as pd
from pecanpy import pecanpy as n2v
import random
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import geopandas as gpd
from shapely.geometry import Point
from infomap import Infomap
from scipy.spatial.distance import jensenshannon

# ===================================================================
# EMBEDDING STORE
# ===================================================================


class EmbeddingMap:
    """Memory-efficient node embedding store.

    Keeps embeddings as a single contiguous float32 matrix plus a
    ``node_id -> row_index`` dict, instead of a dict of Python lists. This is
    ~8x smaller in RAM (float32 packed vs. boxed Python floats + list
    pointers) and lets feature construction fancy-index rows without a
    float64 detour. Storing float32 is lossless: pecanpy/gensim emit float32
    and the feature matrix is cast to float32 downstream anyway.

    Exposes a dict-like interface (``in``, ``[]``, ``len``, ``keys``) so
    existing call sites — including the precomputed path and the returned
    value — keep working unchanged. Plain dicts saved by older runs still
    work everywhere too (build_feature_matrix falls back to per-key lookup).
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
    _cols = ['ORIGIN', 'DESTINATION', 'N_COVISITS', 'TAXONOMY_ORIGIN',
             'TAXONOMY_DESTINATION', 'LAT_ORIGIN', 'LNG_ORIGIN', 'LAT_DESTINATION',
             'LNG_DESTINATION', 'DIST_KM', 'N_UIDS_ORIGIN', 'N_VISITS_ORIGIN',
             'N_UIDS_DESTINATION', 'N_VISITS_DESTINATION', 'DEP']
    _dtypes = {
        'ORIGIN': 'category', 'DESTINATION': 'category',
        'TAXONOMY_ORIGIN': 'category', 'TAXONOMY_DESTINATION': 'category',
        'N_COVISITS': 'float32', 'LAT_ORIGIN': 'float32', 'LNG_ORIGIN': 'float32',
        'LAT_DESTINATION': 'float32', 'LNG_DESTINATION': 'float32',
        'DIST_KM': 'float32', 'N_UIDS_ORIGIN': 'float32', 'N_VISITS_ORIGIN': 'float32',
        'N_UIDS_DESTINATION': 'float32', 'N_VISITS_DESTINATION': 'float32',
        'DEP': 'float32',
    }
    df = pd.read_csv(fpath, sep='\s+', names=_cols, dtype=_dtypes)
    df = df.dropna(subset=['ORIGIN', 'DESTINATION',
                   'TAXONOMY_ORIGIN', 'TAXONOMY_DESTINATION'])

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
        edge_attr=['DIST_KM', 'DEP', 'N_COVISITS'],
    )

    # assign node attrs
    origins = df[['ORIGIN', 'LAT_ORIGIN', 'LNG_ORIGIN',
                  'TAXONOMY_ORIGIN', 'N_UIDS_ORIGIN', 'N_VISITS_ORIGIN']].drop_duplicates()
    # FIX: Align columns directly to match expected attributes ('latitude', 'longitude', 'poi_type')
    origins.columns = ['node_id', 'latitude', 'longitude',
                       'poi_type', 'unique_visits', 'total_visits']

    destinations = df[['DESTINATION', 'LAT_DESTINATION',
                       'LNG_DESTINATION', 'TAXONOMY_DESTINATION',
                       'N_UIDS_DESTINATION', 'N_VISITS_DESTINATION']].drop_duplicates()
    # FIX: Align columns directly to match expected attributes
    destinations.columns = ['node_id', 'latitude', 'longitude',
                            'poi_type', 'unique_visits', 'total_visits']

    # combine them into one master list of unique POIs
    node_data = pd.concat([origins, destinations]).drop_duplicates(
        'node_id').set_index('node_id')

    # map back to graph - convert to dict and apply
    nx.set_node_attributes(G, node_data['latitude'].to_dict(), 'latitude')
    nx.set_node_attributes(G, node_data['longitude'].to_dict(), 'longitude')
    nx.set_node_attributes(G, node_data['poi_type'].to_dict(), 'poi_type')
    nx.set_node_attributes(
        G, node_data['unique_visits'].to_dict(), 'unique_visits')
    nx.set_node_attributes(
        G, node_data['total_visits'].to_dict(), 'total_visits')

    print(f"Nodes: {G.number_of_nodes()}")
    print(f"Edges: {G.number_of_edges()}")
    return G


# ================================================================
# DISTANCE-CONTROLLED SAMPLING
# ================================================================


def distribution_finder(G, bins=50):
    # --- helper: continuous data (bins) ---
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
    dist_values = [v for v in dist_dict.values() if v is not None]
    if dist_values:
        max_d = max(dist_values)
        if max_d > 0.01:
            # Create logarithmic bins: start with 0, then exponentially space from 0.01km (10m) to max_d
            custom_bins = np.concatenate(([0], np.geomspace(0.01, max_d, 50)))
        else:
            custom_bins = np.linspace(0, max_d + 1e-5, 50)
    else:
        custom_bins = bins

    dist_distr, dist_edges_set = get_binned_dist(
        {k: v for k, v in dist_dict.items() if v is not None}, custom_bins)

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
    distributions = (dist_distr, type_distr,
                     uv_distr, tv_distr, deg_distr)
    element_sets = (dist_edges_set, type_nodes_set,
                    uv_nodes_set, tv_nodes_set, deg_nodes_set)

    return distributions, element_sets


def sample_non_edges_dist_controlled(G, distr, total_count, batch_size=2_000_000):
    def _coord(attrs, *keys):
        for k in keys:
            v = attrs.get(k)
            if v is not None:
                return float(v)
        return 0.0

    nodes = list(G.nodes())
    n = len(nodes)
    node_to_idx = {nd: i for i, nd in enumerate(nodes)}

    lat = np.array([_coord(G.nodes[nd], 'latitude', 'lat')
                   for nd in nodes], dtype=np.float64)
    lng = np.array([_coord(G.nodes[nd], 'longitude', 'lon')
                   for nd in nodes], dtype=np.float64)

    # Integer-keyed edge set for faster hashing than string tuples
    edge_set_int = set()
    for u, v in G.edges():
        ui, vi = node_to_idx[u], node_to_idx[v]
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
            still_needed = np.maximum(bin_quotas - bin_filled, 0)
            if still_needed.sum() == 0:
                break

            cur_filled = int(bin_filled.sum())
            if cur_filled == prev_filled:
                stall_rounds += 1
                if stall_rounds >= 5:
                    break
            else:
                stall_rounds = 0
            prev_filled = cur_filled

            ui = np.random.randint(0, n, batch_size)
            vi = np.random.randint(0, n, batch_size)
            mask = ui != vi
            ui, vi = ui[mask], vi[mask]

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

            b_idx = np.digitize(dist, bins=bin_edges) - 1
            in_range = (b_idx >= 0) & (b_idx < n_bins)

            for b in range(n_bins):
                need = still_needed[b]
                if need <= 0:
                    continue
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

    return [edge for bucket in bin_results for edge in bucket]


def sample_non_edges_agg_stratified(G, total_count):
    """
    Non-edge sampler for aggregated (cat||tract) networks.

    Same-tract pairs share a centroid (DIST_KM=0), so spatial sampling can't
    fill the 0-distance bins — the within-tract graph is too dense. This sampler
    enumerates same-tract non-edges directly, then uses distance-controlled bulk
    sampling for cross-tract pairs, with the quota split proportional to the
    same-tract / cross-tract ratio in the actual edge set.
    """
    node_tract = {}
    for node in G.nodes():
        parts = str(node).split('||')
        node_tract[node] = parts[1] if len(parts) == 2 else None

    same_count, cross_count = 0, 0
    for u, v in G.edges():
        tu, tv = node_tract.get(u), node_tract.get(v)
        if tu is not None and tu == tv:
            same_count += 1
        else:
            cross_count += 1

    total_edges = same_count + cross_count
    if total_edges == 0:
        return []

    same_quota = int(round((same_count / total_edges) * total_count))
    cross_quota = total_count - same_quota

    # same-tract: enumerate all missing category pairs per tract directly
    tract_to_nodes = {}
    for node in G.nodes():
        t = node_tract.get(node)
        if t is not None:
            tract_to_nodes.setdefault(t, []).append(node)

    pool = []
    for nodes in tract_to_nodes.values():
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                u, v = nodes[i], nodes[j]
                if not G.has_edge(u, v):
                    pool.append((u, v))

    if len(pool) <= same_quota:
        same_sample = pool
        shortfall = same_quota - len(pool)
        if shortfall > 0:
            print(f"Warning: only {len(pool)} same-tract non-edges available "
                  f"(needed {same_quota}). Redistributing {shortfall} to cross-tract.")
            cross_quota += shortfall
    else:
        same_sample = random.sample(pool, same_quota)

    # cross-tract: build a distance distribution from cross-tract edges only,
    # then delegate to the standard distance-controlled sampler
    cross_non_edges = []
    if cross_quota > 0:
        cross_dist_vals = [
            attrs['DIST_KM']
            for u, v, attrs in G.edges(data=True)
            if node_tract.get(u) != node_tract.get(v) and attrs.get('DIST_KM') is not None
        ]

        if cross_dist_vals:
            max_d = max(cross_dist_vals)
            bin_edges = (
                np.concatenate(([0], np.geomspace(0.01, max_d, 50)))
                if max_d > 0.01 else np.linspace(0, max_d + 1e-5, 50)
            )
            cross_dist_distr = (
                pd.cut(pd.Series(cross_dist_vals),
                       bins=bin_edges, include_lowest=True)
                .value_counts().sort_index()
            )
            cross_non_edges = sample_non_edges_dist_controlled(
                G, cross_dist_distr, cross_quota)
            # drop any same-tract pairs (distance=0) that landed in the first bin
            cross_non_edges = [
                (u, v) for u, v in cross_non_edges
                if node_tract.get(u) != node_tract.get(v)
            ]

    return same_sample + cross_non_edges


# ====================================================================
# PREPARE_DATA
# ====================================================================


def prepare_data(
    fpath, frac=0.5, seed=None, agg=False, compress=0, weight=None, meta=None, trainfile='train.txt', controlled=True
):
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
    nx.Graph : original graph if metadata needed for pipeline
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
            if weight == 'num_occ':
                G_train.add_weighted_edges_from(
                    [(u, v, G[u][v]['weight']) for u, v in train_edges])
            elif weight == 'cov':
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

        if controlled:
            if agg:
                test_non_edges = sample_non_edges_agg_stratified(
                    G, len(test_edges))
                train_non_edges = sample_non_edges_agg_stratified(
                    G, len(train_edges))
            else:
                distrs, _ = distribution_finder(G)
                dist_bins = distrs[0]
                test_non_edges = sample_non_edges_dist_controlled(
                    G, dist_bins, len(test_edges))
                train_non_edges = sample_non_edges_dist_controlled(
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

    if agg:
        import pickle
        with open(fpath, 'rb') as f:
            G = pickle.load(f)
    else:
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
    im_to_nx = im.add_networkx_graph(G, weight='N_COVISITS')
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


def build_feature_matrix(
        edges, G, features, embedding_map, operator='hadamard', cat_threshold=1, agg=False
):
    """
    Build a feature matrix for a list of node pairs.

    Each row corresponds to one edge (u, v). The columns are determined
    by `features`, which is a list that can contain any combination of:

        'emb'       – binary-operator output on node2vec embeddings (128-d by default)
        'geo'       – log geographic distance in km  (1-d)
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
    """
    op_fn = BINARY_OPERATORS[operator]

    # Pre-filter edges missing embeddings to ensure matrix shapes align later
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

    # Vectorized Embedding Operations
    if 'emb' in features:
        # Extract to 2D arrays: shape (N, dim). Fancy-index the packed matrix
        # when available; fall back to per-key lookup for plain-dict maps.
        if hasattr(embedding_map, 'rows'):
            emb_u = embedding_map.rows(U)
            emb_v = embedding_map.rows(V)
        else:
            emb_u = np.asarray([embedding_map[u] for u in U], dtype=np.float32)
            emb_v = np.asarray([embedding_map[v] for v in V], dtype=np.float32)

        # Binary operator applies to the entire (N, 128) array simultaneously
        emb_feat = op_fn(emb_u, emb_v)
        feature_blocks.append(emb_feat)

    if not agg and any(x in features for x in ('cat', 'catsame', 'cbg')):
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

        if 'catsame' in features:
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
    elif not agg:
        print(
            'Category and census-based features invalid for aggregated network. Skipping.')

    # vectorized geographic distance
    if 'geo' in features:
        # Fast extraction using list comprehensions (dict lookups are fast, math is slow)
        lat_u = np.array([G.nodes[u].get('latitude')
                         or 0.0 for u in U], dtype=np.float64)
        lng_u = np.array([G.nodes[u].get('longitude')
                         or 0.0 for u in U], dtype=np.float64)
        lat_v = np.array([G.nodes[v].get('latitude')
                         or 0.0 for v in V], dtype=np.float64)
        lng_v = np.array([G.nodes[v].get('longitude')
                         or 0.0 for v in V], dtype=np.float64)

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

    if 'comm' in features:
        comm_u = np.array([G.nodes[u].get('community', -1) for u in U])
        comm_v = np.array([G.nodes[v].get('community', -1) for v in V])
        comm_feat = ((comm_u == comm_v) & (comm_u != -1)
                     ).astype(float).reshape(-1, 1)
        feature_blocks.append(comm_feat)

    if 'time' in features:
        # uniform distribution for fallback
        default_distr = np.array([0.25, 0.25, 0.25, 0.25])
        # add only time
        time_u = np.array(
            [G.nodes[u].get('time_dist', default_distr) for u in U])
        time_v = np.array(
            [G.nodes[v].get('time_dist', default_distr) for v in V])
        js_dist = jensenshannon(time_u, time_v, axis=1)  # ty: ignore
        time_feat = (js_dist ** 2).reshape(-1, 1)
        feature_blocks.append(time_feat)

    if 'income' in features:
        # uniform distribution for fallback
        default_distr = np.array([0.25, 0.25, 0.25, 0.25])
        # add only income
        inc_u = np.array(
            [G.nodes[u].get('inc_dist', default_distr) for u in U])
        inc_v = np.array(
            [G.nodes[v].get('inc_dist', default_distr) for v in V])
        js_dist = jensenshannon(inc_u, inc_v, axis=1)  # ty: ignore
        inc_feat = (js_dist ** 2).reshape(-1, 1)
        feature_blocks.append(inc_feat)

    if 'ls' in features:
        pass

    X = np.hstack(feature_blocks).astype(np.float32)

    return X, kept_indices


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

    if not agg:
        if features == 'all' or features == ['all']:
            features = ['emb', 'geo', 'cat', 'cbg', 'comm', 'time', 'income']
    else:
        if features == 'all' or features == ['all']:
            features = ['emb', 'geo', 'comm', 'time', 'income']

    # ===== Validation =====
    needs_metadata = bool({'geo', 'cat', 'cbg', 'comm',
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
                           directed=directed, delimiter='\t')
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

        # Store as a packed float32 matrix + index (str keys stay consistent
        # with graph node IDs) instead of a dict of Python lists — ~8x less RAM.
        embedding_map = EmbeddingMap.from_pecanpy(g.nodes, embeddings)

        print(f"Embeddings generated: {len(embedding_map)} nodes, dim={dim}")

    # ===== Assemble feature matrices =====
    train_pos_edges = [tuple(sorted(e)) for e in G_train.edges()]

    if ('cbg' in features or 'tract' in features) and not agg:
        node_to_area(G)

    if 'comm' in features:
        node_to_comm(G)

    if not agg:
        if 'time' in features or 'income' in features:
            add_outside_metadata(G)

    X_train_pos, _ = build_feature_matrix(
        train_pos_edges, G, features, embedding_map, operator, cat_threshold, agg)
    X_train_neg, _ = build_feature_matrix(
        train_non_edges, G, features, embedding_map, operator, cat_threshold, agg)

    X_train = np.vstack([X_train_pos, X_train_neg])
    y_train = np.concatenate([
        np.ones(len(X_train_pos)),
        np.zeros(len(X_train_neg))
    ])
    del X_train_pos, X_train_neg

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
        test_edges, G, features, embedding_map, operator, cat_threshold, agg)
    X_test_neg, _ = build_feature_matrix(
        test_non_edges, G, features, embedding_map, operator, cat_threshold, agg)

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
