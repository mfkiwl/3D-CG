import logging

logger = logging.getLogger(__name__)

def log():
    for i in range(10):
        logger.info(i)
        print(i)


# def log():
#     logger.info("An INFO message from " + __name__)
#     for i in range(10):
#         logger.info(i)
#         print(i)
