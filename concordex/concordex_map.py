import pandas as pd
from scipy.io import mmread
from sklearn.neighbors import kneighbors_graph


def setup_map_args(parser):
    parser_map = parser.add_parser(
        "map",
        description="",
        help="",
    )
    parser_map.add_argument("mtx_file", help="Matrix file")
    parser_map.add_argument(
        "-a",
        metavar="Assignments",
        help=("Assignments"),
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
    # default value is false
    return parser_map


def validate_map_args(parser, args):
    mtx_fname = args.mtx_file
    assignments_fname = args.a
    output = args.o
    run_map(mtx_fname, assignments_fname, output)
    return


def run_map(mtx_fname, assignments_fname, output):
    mtx = mmread(mtx_fname)
    assignments = pd.read_csv(assignments_fname, header=None)
    assignments.columns = ["label"]
    assignments = assignments["label"].values
    map_mtx = concordex_map(mtx, assignments)
    map_mtx.to_csv(output, sep="\t")
    return


def concordex_map(mtx, assignments):
    n_neighbors = 20
    conn = kneighbors_graph(mtx, n_neighbors, mode="connectivity", include_self=False)
    df = pd.DataFrame(conn.A, index=assignments, columns=assignments)
    t = df.T.groupby(df.columns).sum().T.groupby(df.columns).sum()
    map_mtx = t.div(t.sum(1), axis=0)
    return map_mtx
