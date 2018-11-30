from steppy.base import Step
from steppy.adapter import Adapter, E
from common_blocks import loaders, models


def preprocess_binary_train(config, suffix):
    reader_train = Step(transformer=loaders.MetaReader(),
                        name='xy_train{}'.format(suffix),
                        input_data=['input'],
                        adapter=Adapter({'meta': E('input', 'meta')}),
                        )
    reader_inference = Step(transformer=loaders.MetaReader(),
                            name='xy_inference{}'.format(suffix),
                            input_data=['callback_input'],
                            adapter=Adapter({'meta': E('callback_input', 'meta_valid')}),
                            )
    binary_loader = Step(transformer=loaders.OneClassClassificationLoader(dataset_params=config.dataset_params,
                                                                          loader_params=config.loader_params),
                         name='loader{}'.format(suffix),
                         input_steps=[reader_train, reader_inference],
                         adapter=Adapter({'X': E(reader_train.name, 'X'),
                                          'y': E(reader_train.name, 'y'),
                                          'X_valid': E(reader_inference.name, 'X'),
                                          'y_valid': E(reader_inference.name, 'y')}),
                         )
    return binary_loader
