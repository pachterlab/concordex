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
    _, orientation = check_matrix_dims(graph, neighbors)
    graph = reorient_matrix(graph, neighbors, how=orientation)
    
    print(graph)
    diag_sum = np.trace(graph)
    if diag_sum != 0:
        warnings.warn("Some nodes in the graph are self-referential. There should not be an edge between a node and itself.")
    
    return graph

def check_matrix_dims(x, k):
    """
    Check the dimensions of a matrix and determine the orientation of its neighbors.

    Parameters:
    x (numpy.ndarray): The input matrix.
    k (int): The number of neighbors.

    Returns:
    tuple: A tuple containing the dimensions of the matrix and a pattern indicating the orientation of the neighbors.

    Raises:
    ValueError: If the orientation of the neighbors cannot be determined.

    """
    dims = np.array(np.shape(x))
    
    def guess_orientation(x, k, dims):
        """
        Guess the orientation of the neighbors based on the dimensions of the matrix.

        Parameters:
        x (numpy.ndarray): The input matrix.
        k (int): The number of neighbors.
        dims (numpy.ndarray): The dimensions of the matrix.

        Returns:
        int: A pattern indicating the orientation of the neighbors.

        """
        if np.diff(dims) == 0:
            if np.all(np.sum(x, axis=1) / k) == 1:
                return 1
            if np.all(np.sum(x, axis=0) / k) == 1:
                return 2
        else:
            axis = np.where(dims == k)[0]
            if len(axis) == 0:
                return None
            if axis[0] == 0:
                return 3
            if axis[0] == 1:
                return 4

        return None
    
    pattern = guess_orientation(x, k=k, dims=dims)

    if pattern is None:
        raise ValueError("Cannot determine whether neighbors are oriented on the rows or columns")

    return dims, {
        1: "no_reorient",
        2: "transpose",
        3: "expand_row",
        4: "expand_col"
    }[pattern]


def reorient_matrix(x, k, how):
    """
    Reorients the given matrix based on the specified orientation pattern.

    Parameters:
    x (numpy.ndarray or scipy.sparse.csr_matrix): The matrix to be reoriented.
    k (int): The number of neighbors.
    how (str): The orientation pattern. Can be one of the following:
        - "no_reorient": No reorientation needed.
        - "transpose": Transpose the matrix.
        - "expand_row": Expand the matrix by repeating rows.
        - "expand_col": Expand the matrix by repeating columns.

    Returns: 
    Reoriented matrix

    Raises:
    ValueError: If the 'how' parameter is invalid
    """
    dims, _ = check_matrix_dims(x, k)
    r, c = dims
    
    if how == "no_reorient":
        return x
    elif how == "transpose":
        return np.transpose(x)
    elif how == "expand_row":
        i = np.sort(np.repeat(np.arange(c), k))
        j = np.ravel(x)
        data = np.ones(c * k)
        return csr_matrix((data, (i, j)), shape=(c, c))
    elif how == "expand_col":
        i = np.sort(np.repeat(np.arange(r), k))
        j = np.ravel(x)
        data = np.ones(r * k)
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
    """
    Runs concordex_map calculation on given k-nearest neighbors (knn) matrix.

    Parameters:
    knn_fname (str): The file path to the k-nearest neighbors matrix file.
    labels_fname (str): The file path to the labels file.
    k (int): The number of nearest neighbors to consider for each data point.
    output (str): The file path to save the MAP results.

    Returns:
    None
    """
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
