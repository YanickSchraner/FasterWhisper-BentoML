import logging

import ctranslate2


def configure_logging():

    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d %(levelname)s %(module)s -%(funcName)s: %(message)s')
    ch.setFormatter(formatter)

    bentoml_logger = logging.getLogger("bentoml")
    bentoml_logger.addHandler(ch)
    bentoml_logger.setLevel(logging.DEBUG)

    logging.getLogger("faster_whisper").setLevel(logging.DEBUG)
    ctranslate2.set_log_level(logging.INFO)
