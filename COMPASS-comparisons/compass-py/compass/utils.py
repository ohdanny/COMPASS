import logging
from pathlib import Path
import sys

def setup_logger(name: str,log_file:Path,clear:bool=True):

    if clear:
        log_file.write_text("")  # clear existing log file
        
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False  # prevent duplicate logs if root logger is also configured
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    fh = logging.FileHandler(log_file)

    fh.setFormatter(fmt)
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger
