import pandas as pd
import numpy as np
from scipy.io import mmread
import random

from concordex_map import concordex_map # why was there a dot


def setup_stat_args(parser):
    parser_stat = parser.add_parser(
        "stat",
        description="",
        help="",
    )
    parser_stat.add_argument("knn_file", help="KNN graph")
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
    knn = mmread(knn_fname)
    labels = pd.read_csv(labels_fname, header=None)
    labels.columns = ["label"]
    labels = labels["label"].values
    trace, random_trace, corrected_trace = concordex_stat(knn, labels)
    # Missing format for output file containing trace values
    print(f"Trace: {trace}")
    print(f"Average random trace: {random_trace}")
    print(f"Corrected trace: {corrected_trace}")
    return


def concordex_stat(knn, labels):
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
    return trace, random_trace, corrected_trace
