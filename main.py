import pandas as pd
from attrdict import AttrDict
from common_blocks import utils, pipelines

logger = utils.get_logger('ship-detection')
SEED = 1234
DEV_MODE = True
MEAN = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]
PARAMS = utils.read_yaml('config.yaml').parameters
CONFIG = AttrDict({
    'loader_params': {'train': {'batch_size': 32,
                                'shuffle': False,
                                'num_workers': 4,
                                'pin_memory': True,
                                },
                      'inference': {'batch_size': PARAMS.batch_size,
                                    'shuffle': False,
                                    'num_workers': PARAMS.num_workers,
                                    'pin_memory': PARAMS.pin_memory,
                                    },
                      },
    'dataset_params': {'mean': MEAN,
                       'std': STD,
                       'sample_size': 7500,
                       'empty_fraction': PARAMS.evaluation_empty_fraction},
    'execution': {'experiment_dir': PARAMS.experiment_dir,
                  }
})


def train():
    pass


def train_ship_no_ship():
    metadata = pd.read_csv(PARAMS.metadata_filepath)
    metadata_train = metadata.query('is_train==1')
    metadata_train = utils.add_big_image_id(metadata_train)
    meta_train_split, meta_valid_split = utils.train_test_split_with_empty_fraction_with_groups(
        df=metadata_train,
        test_size=PARAMS.evaluation_size,
        test_empty_fraction=PARAMS.evaluation_empty_fraction,
        random_state=SEED)
    if DEV_MODE:
        meta_train_split = meta_train_split.sample(PARAMS.dev_mode_size, random_state=SEED)
        meta_valid_split = meta_valid_split.sample(PARAMS.dev_mode_size//2, random_state=SEED)
    data = {'input': {'meta': meta_train_split},
            'callback_input': {'meta_valid': meta_valid_split},
            }
    pipeline = ship_no_ship_pipeline(config=CONFIG)
    a, b = pipeline.fit_transform(data)


def ship_no_ship_pipeline(config):
    preprocessing = pipelines.preprocess_binary_train(config=CONFIG,
                                                      suffix='_ship_no_ship',
                                                      )
    preprocessing.set_parameters_upstream({'experiment_directory': config.execution.experiment_dir,
                                           'is_fittable': False,
                                           })
    pipeline = preprocessing
    return pipeline


train_ship_no_ship()
