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
    run_map()
    return


def run_map():
    nomap_map()
    return


def nomap_map():
    return
