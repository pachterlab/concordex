import pandas as pd
import numpy as np
from scipy.io import mmread
import random

from .concordex_map import concordex_map


def setup_stat_args(parser):
    parser_stat = parser.add_parser(
        "stat",
        description="",
        help="",
    )
    parser_stat.add_argument("knn_file", help="KNN graph")
    parser_stat.add_argument(
        "-a",
        metavar="Assignments",
        help=("Assignments"),
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
    assignments_fname = args.a
    output = args.o
    neighbors = args.k
    run_stat(knn_fname, assignments_fname, neighbors, output)
    return


def run_stat(knn_fname, assignments_fname, output):
    knn = mmread(knn_fname)
    assignments = pd.read_csv(assignments_fname, header=None)
    assignments.columns = ["label"]
    assignments = assignments["label"].values
    trace, random_trace, corrected_trace = concordex_stat(knn, assignments)
    # Missing format for output file containing trace values
    print(f"Trace: {trace}")
    print(f"Average random trace: {random_trace}")
    print(f"Corrected trace: {corrected_trace}")
    return


def concordex_stat(knn, assignments):
    # mapped matrix with normal assignments
    map_mtx = concordex_map(knn, assignments)

    # mapped matrix with permuted assignments
    n_iters = 15
    random_map_matrices = []
    for i in range(n_iters):
        random.shuffle(assignments)
        random_map_matrices.append(concordex_map(knn, assignments))

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
