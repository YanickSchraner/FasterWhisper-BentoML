import logging


def configure_logging():

    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s -%(funcName)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        )
