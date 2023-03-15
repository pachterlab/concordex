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
    run_trace()
    return


def run_trace():
    nomap_trace()
    return


def nomap_trace():
    # mapped matrix with normal assignments
    map_mtx = nomap_map()

    # mapped matrix with permuted assignments

    # compute trace and random
    return
