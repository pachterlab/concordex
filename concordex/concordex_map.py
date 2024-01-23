import pandas as pd
from scipy.io import mmread
from sklearn.neighbors import kneighbors_graph


def setup_map_args(parser):
    parser_map = parser.add_parser(
        "map",
        description="",
        help="",
    )
    parser_map.add_argument("knn_file", help="Path to KNN graph")
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
        "-n",
        metavar="Neighbors",
        help=("Number of neighbors to expect for each observation; defaults to 20"),
        type=int,
        default=20,
    )
    # default value is false
    return parser_map


def validate_map_args(parser, args):
    knn_fname = args.knn_file
    labels_fname = args.a
    output = args.o
    neighbors = args.n
    run_map(knn_fname, labels_fname, neighbors, output)
    return


def run_map(knn_fname, labels_fname, neighbors, output):
    knn = mmread(knn_fname)
    labels = pd.read_csv(labels_fname, header=None)
    labels.columns = ["label"]
    labels = labels["label"].values
    map_knn = concordex_map(knn, labels)
    map_knn.to_csv(output, sep="\t")
    return


def concordex_map(x, labels):
    """
    @param x: A numeric matrix specifying the neighborhood structure of observations. Typically an adjacency matrix produced by a k-Nearest Neighbor algorithm. It can also be a matrix whose rows correspond to each observation and columns correspond to neighbor indices, i.e. matrix form of an adjacency list which can be a matrix due to fixed number of neighbors.
    
    @param labels: A numeric or character vector containing the label or class corresponding to each observation. For example, a cell type or cluster ID.
    """
    df = pd.DataFrame(x, index=labels, columns=labels)
    t = df.T.groupby(df.columns).sum().T.groupby(df.columns).sum()
    map_mtx = t.div(t.sum(1), axis=0)
    return map_mtx
