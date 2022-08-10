import logging

def init_log(name):

    logger = logging.getLogger()

    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s %(levelname)s [%(module)s]: %(message)s')

    file_handler = logging.FileHandler(name)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger