from datetime import datetime
import logging

LOG_NAME = datetime.now().strftime('llm_attempt_%H_%M_%d_%m_%Y.log')

logger = logging.getLogger('LLMPricer')
logger.setLevel(logging.DEBUG)

fh = logging.FileHandler(LOG_NAME)
fh.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)

logger.addHandler(fh)
logger.addHandler(ch)