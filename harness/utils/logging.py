import logging
import os


def setup_logging(logfile):
    if os.path.exists(logfile):
        os.remove(logfile)
    logging.basicConfig(filename=logfile, format='%(asctime)s %(message)s', level=logging.DEBUG)


def print_and_log(message):
    """Utility function."""
    print(message, flush=True)
    logging.info(message)
