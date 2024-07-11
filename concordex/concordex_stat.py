import pandas as pd
import numpy as np
from scipy.io import mmread
import random

from concordex_map import concordex_map 


def setup_stat_args(parser):
    parser_stat = parser.add_parser(
        "stat",
        description="",
        help="",
    )
    parser_stat.add_argument("knn_file", help="KNN matrix")
    parser_stat.add_argument(
        "-a",
        metavar="labels",
        help=("labels"),
        type=str,
        default=None,
    )
    parser_stat.add_argument(
        "-o",
        metavar="OUT",
        help=("Path to output file"),
        type=str,
        default=None,
    )
    parser_stat.add_argument(
        "-k",
        metavar="Neighbors",
        help=("Number of neighbors to expect for each observation; defaults to 20"),
        type=int,
        default=20,
    )
    # default value is false
    return parser_stat


def validate_stat_args(parser, args):
    knn_fname = args.knn_file
    labels_fname = args.a
    output = args.o
    neighbors = args.k
    run_stat(knn_fname, labels_fname, neighbors, output)
    return


def run_stat(knn_fname, labels_fname, output):
    """
    Calculates concordex_stat.

    Args:
        knn_fname (str): The file path of the knn data.
        labels_fname (str): The file path of the labels data.
        output (str): The file path to save the output.

    Returns:
        None
    """
    knn = mmread(knn_fname)
    labels = pd.read_csv(labels_fname, header=None)
    labels.columns = ["label"]
    labels = labels["label"].values
    trace, corrected_trace, random_trace = calculate_concordex(knn, labels)
    # Missing format for output file containing trace values
    print(f"Concordex: {trace}")
    print(f"Corrected concordex: {corrected_trace}")
    print(f"Mean random concordex: {random_trace}")
    return


def calculate_concordex(knn, labels):
    """
    Computes concordex.

    Parameters:
    - knn: The k-nearest neighbors matrix.
    - labels: The labels corresponding to the data points.

    Returns:
    - concordex (float): The trace of the mapped matrix divided by its size.
    - mean random concordex (float): The average trace of randomly permuted mapped matrices divided by their size.
    - corrected concordex (float): The ratio of the trace to the average random trace.

    """
    # mapped matrix with normal labels
    map_mtx = concordex_map(knn, labels)

    # mapped matrix with permuted labels
    n_iters = 15
    random_map_matrices = []
    for i in range(n_iters):
        random.shuffle(labels)
        random_map_matrices.append(concordex_map(knn, labels))

    # compute trace and random
    trace = np.trace(map_mtx) / map_mtx.shape[0]
    random_trace = np.mean(
        [
            np.trace(random_map_mtx) / random_map_mtx.shape[0]
            for random_map_mtx in random_map_matrices
        ]
    )
    corrected_trace = trace / random_trace
    return trace, corrected_trace, random_trace
