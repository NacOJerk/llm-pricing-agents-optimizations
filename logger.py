from datetime import datetime
import logging
from pathlib import Path

_logger = None

def init_logger(dir_path: Path):
    global _logger
    assert _logger == None, 'Logger should only be init once'

    LOG_NAME = datetime.now().strftime('llm_attempt_%H_%M_%d_%m_%Y.log')

    logger = logging.getLogger('LLMPricer')
    logger.setLevel(logging.DEBUG)

    dir_path.mkdir(parents=True, exist_ok=True)
    log_path = dir_path / LOG_NAME

    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    _logger = logger

def get_logger():
    global _logger
    assert _logger != None, 'Logger wasn\'t init'
    return _logger
    