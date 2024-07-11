from . import __version__
import argparse
import sys
from .concordex_map import setup_map_args, validate_map_args
from .concordex_stat import setup_stat_args, validate_stat_args

def main():
    # setup parsers
    parser = argparse.ArgumentParser(
        description=f"concordex {__version__}: A clustering diagnostic to replace UMAP"
    )
    subparsers = parser.add_subparsers(
        dest="command",
        metavar="<CMD>",
    )

    # Setup the arguments for all subcommands
    command_to_parser = {
        "map": setup_map_args(subparsers),
        "stat": setup_stat_args(subparsers),
    }

    # Show help when no arguments are given
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    if len(sys.argv) == 2:
        if sys.argv[1] in command_to_parser:
            command_to_parser[sys.argv[1]].print_help(sys.stderr)
        else:
            parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    # Setup validator and runner for all subcommands (validate and run if valid)
    COMMAND_TO_FUNCTION = {
        "map": validate_map_args,
        "stat": validate_stat_args,
    }
    COMMAND_TO_FUNCTION[sys.argv[1]](parser, args)


if __name__ == "__main__":
    main()
