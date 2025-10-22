import logging
import logging.handlers
from . import constants as c

def setup_logger(log_level=logging.INFO):
    log_format = logging.Formatter(
        '%(levelname)s   | %(asctime)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    logger = logging.getLogger(c.LOGGER_NAME)
    logger.setLevel(log_level)
    logger.propagate = False

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)

    if not logger.handlers:
        logger.addHandler(console_handler)

    return logger
