import logging


def setup_logging(logfile):
    logging.basicConfig(filename=logfile, format='%(asctime)s %(message)s', level=logging.DEBUG)


def print_and_log(message):
    """Utility function."""
    print(message, flush=True)
    logging.info(message)
