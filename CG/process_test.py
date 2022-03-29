from datetime import datetime
import time
import logging

logging.basicConfig(filename='test.log', encoding='utf-8', level=logging.DEBUG)

while True:
    logging.debug(datetime.now())    
    time.sleep(2)
