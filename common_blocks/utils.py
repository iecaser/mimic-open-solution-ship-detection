import sys
import yaml
import logging
from attrdict import AttrDict


def read_yaml(filepath):
    with open(filepath) as f:
        content = yaml.load(f)
    return AttrDict(content)


def get_logger(name):
    logger = logging.getLogger(name)
    if logger.hasHandlers():
        return logger
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setLevel(logging.INFO)
    handler.setFormatter(fmt=logging.Formatter(fmt='%(asctime)s %(name)s [%(levelname)s] >>> %(message)s',
                                               datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(handler)
    return logger
