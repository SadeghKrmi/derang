import logging

def logger():
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s  [%(levelname)8s]: %(message)s')
    return  logging.getLogger(__file__)


_LOGGER = logger()