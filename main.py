import pandas as pd
import numpy as np
from attrdict import AttrDict
from common_blocks import utils
import ipdb
from sklearn.model_selection import GroupShuffleSplit

SEED = 1234
DEV_MODE = True
PARAMS = utils.read_yaml('config.yaml').parameters
CONFIG = AttrDict({

})


def train():
    pass


def train_ship_no_ship():
    metadata = pd.read_csv(PARAMS.metadata_filepath)
    metadata_train = metadata.query('is_train==1')
    metadata_train = add_big_image_id(metadata_train)
    meta_train_split, meta_valid_split = train_test_split_with_empty_fraction_with_groups(df=metadata_train,
                                                                                          test_size=PARAMS.evaluation_size,
                                                                                          test_empty_fraction=PARAMS.evaluation_empty_fraction,
                                                                                          random_state=SEED)


def train_test_split_with_empty_fraction_with_groups(df, test_size, test_empty_fraction, random_state):
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
    if DEV_MODE:
        train = train.sample(PARAMS.dev_mode_size, random_state=SEED)
        test = test.sample(PARAMS.dev_mode_size//2, random_state=SEED)
    return train, test


def add_big_image_id(df):
    big_image_ids = pd.read_csv('../open-solution-ship-detection/big-images-ids_v2.csv')
    df['ImageId'] = df['id']+'.jpg'
    df = pd.merge(df, big_image_ids)
    return df


def preprocess_binary_train():
    pass


train_ship_no_ship()
