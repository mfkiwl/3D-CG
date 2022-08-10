import logging
import log_start

logger = log_start.init_log('test2.log')


import other_module
other_module.fcn()