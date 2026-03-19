import logging
from accelerate.logging import get_logger as acc_get_logger

def get_logger(name):
    logger = acc_get_logger(name)
    return logger