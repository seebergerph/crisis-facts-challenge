import sys
import logging

LOG_FORMAT = "%(message)s"

logging.basicConfig(
    format=LOG_FORMAT,
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("crisis-facts-challenge")