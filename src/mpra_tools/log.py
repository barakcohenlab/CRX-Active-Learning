"""Helper functions to make and get a console logger."""
import logging


def get_logger(name=None):
    logger = logging.getLogger(name)
    if len(logger.handlers) == 0:
        logger = _make_logger(name=name)
    return logger


def _make_logger(name=None):
    console = logging.StreamHandler()
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.addHandler(console)
    log_format = "[%(asctime)s][%(levelname)-7s] %(message)s"
    log_formatter = logging.Formatter(log_format, "%Y-%m-%d %H:%M:%S")
    console.setFormatter(log_formatter)
    return logger
