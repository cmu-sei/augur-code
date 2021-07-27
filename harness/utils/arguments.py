import argparse
import sys


def get_parsed_arguments():
    """Sets up the parser, and parses current arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", help="run in test mode", action="store_true")
    parser.add_argument("--config", type=str, help="config file name in current folder")
    parser.add_argument("--exp_config", type=str, help="path to experiment config file, from base config folder")
    parser.add_argument("--exp_user", help="indicates that user should select from available config files", action="store_true")

    if len(sys.argv) < 2:
        print("No command line arguments.")

    args = parser.parse_args()
    return args
