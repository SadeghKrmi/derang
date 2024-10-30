import logging

def logger():
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s  [%(levelname)8s]: %(message)s')
    return  logging.getLogger(__file__)

logger = logger()

if __name__ == '__main__':
    logger.info(f'Running the logger.py function with level {logger} active!')