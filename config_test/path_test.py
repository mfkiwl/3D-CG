import logging.config
import logging.handlers

logger = logging.getLogger()
logger.setLevel(logging.INFO)
smtp_handler = logging.handlers.SMTPHandler(mailhost=('outgoing.mit.edu', 465),
                            fromaddr='austinsp@mit.edu',
                            toaddrs=['austinsp@mit.edu'],
                            subject='Sample Log Mail',
                            credentials=('austinsp','gwm2653hj'),
                            secure=None)

logger.addHandler(smtp_handler)
logger.info("logger configured")