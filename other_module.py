import logging

logger = logging.getLogger(__name__)

def fcn():
    logger.info('hi from other module')
    print('In the other fcn')