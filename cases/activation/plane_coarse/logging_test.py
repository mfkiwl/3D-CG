import logging
import sublog       # Have to import the whole module for the logging to work

# def main():
#     # create logger
#     global logger
#     logger = logging.getLogger(__name__)
#     logger.setLevel(logging.DEBUG)

#     # create console handler and set level to debug
#     fh = logging.FileHandler('test_log.log')
#     fh.setLevel(logging.DEBUG)

#     # create formatter
#     formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

#     # add formatter to ch
#     fh.setFormatter(formatter)

#     # add ch to logger
#     logger.addHandler(fh)

#     # 'application' code
#     # logger.debug('debug message')
#     logger.info('initializing')
#     # logger.warning('warn message')
#     # logger.error('error message')
#     # logger.critical('critical message')

#     test_fcn()
#     logger.info('done')

global logger
logger = logging.getLogger()

formatter = logging.Formatter('%(asctime)s %(levelname)s [%(module)s]: %(message)s')
logging_out = logging.FileHandler('test.log')
logging_out.setFormatter(formatter)
logging_out.setLevel(logging.DEBUG)

# root logger, no __name__ as in submodules further down the hierarchy
logger.addHandler(logging_out)
logger.setLevel(logging.DEBUG)

logger.info("An INFO message from " + __name__)
sublog.log()
