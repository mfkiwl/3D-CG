import logging

def initialize_logger(log_fname):
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s %(levelname)s [%(module)s]: %(message)s')

    # Create file handler and set level to debug
    file_handler = logging.FileHandler('{}.log'.format(log_fname))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger