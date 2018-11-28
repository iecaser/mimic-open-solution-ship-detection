import sys
import yaml
import logging
from attrdict import AttrDict
from sklearn.model_selection import GroupShuffleSplit
import pandas as pd


def add_big_image_id(df):
    big_image_ids = pd.read_csv('../open-solution-ship-detection/big-images-ids_v2.csv')
    df['ImageId'] = df['id']+'.jpg'
    df = pd.merge(df, big_image_ids)
    return df


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


def train_test_split_with_empty_fraction_with_groups(df,
                                                     test_size,
                                                     test_empty_fraction,
                                                     random_state):
    cv = GroupShuffleSplit(n_splits=2, test_size=test_size, random_state=random_state)
    for train_inds, test_inds in cv.split(df, groups=df.BigImageId):
        train, test = df.iloc[train_inds], df.iloc[test_inds]
        break
    empty_test, non_empty_test = test.query('is_not_empty==0'), test.query('is_not_empty==1')
    resample_empty_test = empty_test.sample(n=int(test_size * test_empty_fraction),
                                            random_state=random_state)
    resample_non_empty_test = non_empty_test.sample(n=test_size - resample_empty_test.shape[0],
                                                    random_state=random_state)
    train = train.sample(frac=1, random_state=random_state)
    test = pd.concat([resample_empty_test, resample_non_empty_test],
                     axis=0).sample(frac=1, random_state=random_state)
    return train, test
