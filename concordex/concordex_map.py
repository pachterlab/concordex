import warnings
import numpy as np
import pandas as pd
from scipy.io import mmread
# from sklearn.neighbors import kneighbors_graph
from scipy.sparse import csr_matrix

def setup_map_args(parser):
    parser_map = parser.add_parser(
        "map",
        description="",
        help="",
    )
    parser_map.add_argument("-knn_file", help="Path to KNN graph")
    parser_map.add_argument(
        "-a",
        metavar="Labels",
        help=("Labels"),
        type=str,
        default=None,
    )
    parser_map.add_argument(
        "-o",
        metavar="OUT",
        help=("Path to output file"),
        type=str,
        default=None,
    )
    parser_map.add_argument(
        "-k",
        metavar="Neighbors",
        help=("Number of neighbors to expect for each observation; defaults to 20"),
        type=int,
        default=20,
    )
    # default value is false
    return parser_map

def check_labels(labels, expected=None):
    uniq = list(set(labels))
    n_uniq = len(uniq)

    if n_uniq < 2:
        warnings.warn("Must have at least 2 distinct labels")
        warnings.warn(f"There {'is' if n_uniq == 1 else 'are'} {n_uniq} class label{'s' if n_uniq != 1 else ''}")
        return False

    if expected is not None:
        n_labels = len(labels)

        if n_labels != expected:
            message = (
                "Too few labels supplied"
                if n_labels < expected
                else "Too many labels supplied"
            )
            warnings.warn(message)
            warnings.warn(f"{expected} label{'s' if expected != 1 else ''} are required")
            warnings.warn(f"You supplied {n_labels} label{'s' if n_labels != 1 else ''}")
            return False

    return True

# need to check
def check_graph(graph, neighbors):
    orientation = check_matrix_dims(graph, neighbors, return_dims=False)
    graph = reorient_matrix(graph, neighbors, how=orientation)
    
    diag_sum = np.trace(graph)
    if diag_sum != 0:
        warnings.warn("Some nodes in the graph are self-referential. There should not be an edge between a node and itself.")
    
    return graph

def check_matrix_dims(x, k):
    dims = np.array(np.shape(x))
    # print(f'dims {dims}')
    
    def guess_orientation(x, k, dims):
        if np.diff(dims) == 0:
            if np.all(np.sum(x, axis=1) / k) == 1:
                return 1
            if np.all(np.sum(x, axis=0) / k) == 1:
                return 2
        else:
            axis = np.where(dims == k)[0] # axis = which(dims == k)
            if len(axis) == 0:
                return None
            if 0 in axis:
                return 3
            if 1 in axis:
                return 4

        return None
    
    pattern = guess_orientation(x, k=k, dims=dims)

    if pattern is None:
        raise ValueError("Cannot determine whether neighbors are oriented on the rows or columns")

    return dims, {
        1: "none",
        2: "transpose",
        3: "expand_row",
        4: "expand_col"
    }[pattern]

def reorient_matrix(x, k, how):
    dims, _ = check_matrix_dims(x, k) # look closely at this
    r, c = dims

    if how == "none":
        return x
    elif how == "transpose":
        return np.transpose(x)
    elif how == "expand_row":
        i = np.sort(np.repeat(np.arange(c), k))
        j = np.ravel(x)
        data = np.ones(c * k)
        return csr_matrix((data, (i, j)), shape=(c, c))
    elif how == "expand_col":
        i = np.repeat(np.arange(r), k)
        j = np.ravel(x)
        data = np.ones(r * k)
        print(f"sizes: {np.shape(i)}, {np.shape(j)}, {np.shape(data)}")
        return csr_matrix((data, (i, j)), shape=(r, r))
    else:
        raise ValueError("Invalid 'how' parameter")
    

def validate_map_args(parser, args):
    knn_fname = args.knn_file
    labels_fname = args.a
    output = args.o
    neighbors = args.k
    run_map(knn_fname, labels_fname, neighbors, output)
    return


def run_map(knn_fname, labels_fname, k, output):
    knn = mmread(knn_fname)
    labels = pd.read_csv(labels_fname, header=None)
    labels.columns = ["label"]
    labels = labels["label"].values
    map_knn = concordex_map(knn, labels, k)
    map_knn.to_csv(output, sep="\t")
    return


def concordex_map(x, labels, k):
    """
    @param x: A numeric matrix specifying the neighborhood structure of observations. Typically an adjacency matrix produced by a k-Nearest Neighbor algorithm. It can also be a matrix whose rows correspond to each observation and columns correspond to neighbor indices, i.e. matrix form of an adjacency list which can be a matrix due to fixed number of neighbors.
    
    @param labels: A numeric or character vector containing the label or class corresponding to each observation. For example, a cell type or cluster ID.
    """
    x = check_graph(x, k)
    df = pd.DataFrame(x, index=labels, columns=labels)
    t = df.T.groupby(df.columns).sum().T.groupby(df.columns).sum()
    map_mtx = t.div(t.sum(1), axis=0)
    return map_mtx
