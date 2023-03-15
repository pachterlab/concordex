import pandas as pd
import numpy as np
from scipy.io import mmread
import random

from .nomap_map import nomap_map


def setup_trace_args(parser):
    parser_trace = parser.add_parser(
        "trace",
        description="",
        help="",
    )
    parser_trace.add_argument("mtx_file", help="Matrix file")
    parser_trace.add_argument(
        "-a",
        metavar="Assignments",
        help=("Assignments"),
        type=str,
        default=None,
    )
    parser_trace.add_argument(
        "-o",
        metavar="OUT",
        help=("Path to output file"),
        type=str,
        default=None,
    )
    # default value is false
    return parser_trace


def validate_trace_args(parser, args):
    mtx_fname = args.mtx_file
    assignments_fname = args.a
    output = args.o
    run_trace(mtx_fname, assignments_fname, output)
    return


def run_trace(mtx_fname, assignments_fname, output):
    mtx = mmread(mtx_fname)
    assignments = pd.read_csv(assignments_fname, header = None)
    assignments.columns = ['label']
    assignments = assignments['label'].values
    trace, random_trace, corrected_trace = nomap_trace(mtx, assignments)
    # Missing format for output file containing trace values
    print(f'Trace: {trace}')
    print(f'Average random trace: {random_trace}')
    print(f'Corrected trace: {corrected_trace}')
    return


def nomap_trace(mtx, assignments):
    # mapped matrix with normal assignments
    map_mtx = nomap_map(mtx, assignments)

    # mapped matrix with permuted assignments
    n_iters = 15  
    random_map_matrices = []
    for i in range(n_iters):
      random.shuffle(assignments)
      random_map_matrices.append(nomap_map(mtx, assignments))

    # compute trace and random
    trace = np.trace(map_mtx)/map_mtx.shape[0]
    random_trace = np.mean([np.trace(random_map_mtx)/random_map_mtx.shape[0] for random_map_mtx in random_map_matrices])
    corrected_trace = trace / random_trace
    return trace, random_trace, corrected_trace
