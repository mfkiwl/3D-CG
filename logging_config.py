import logging

def initialize_logger(root_name):
    # create logger
    logger = logging.getLogger('root')
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s %(levelname)s [%(module)s]: %(message)s')

    # create file handler and set level to debug
    file_handler = logging.FileHandler(root_name+'.log')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger