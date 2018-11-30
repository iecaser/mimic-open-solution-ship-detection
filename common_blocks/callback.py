from common_blocks import utils
import numpy as np


class Averager(object):
    def __init__(self):
        self.sequence = []

    @property
    def value(self):
        return np.mean(self.sequence)

    def send(self, num):
        self.sequence.append(num)

    def reset(self):
        self.sequence = []


class Callback(object):
    def __init__(self):
        self.logger = utils.get_logger('callback')
        self.batch_id = None
        self.epoch_id = None
        self.model = None
        self.optimizer = None
        self.criterion = None

    def on_train_begin(self, *args, **kwargs):
        self.batch_id = 0
        self.epoch_id = 0
        self.logger.info('train begin')

    def on_epoch_begin(self, *args, **kwargs):
        self.epoch_id += 1
        self.batch_id = 0
        self.logger.info('epoch {}'.format(self.epoch_id))

    def on_batch_begin(self, *args, **kwargs):
        self.batch_id += 1

    def on_batch_end(self, *args, **kwargs):
        pass

    def on_epoch_end(self, *args, **kwargs):
        pass

    def on_train_end(self, *args, **kwargs):
        self.logger.info('train end')


class CallbackContainer(Callback):
    def __init__(self, callbacks):
        super().__init__()
        if isinstance(callbacks, Callback):
            self.callbacks = [callbacks]
        else:
            self.callbacks = callbacks

    def on_train_begin(self, *args, **kwargs):
        for callback in self.callbacks:
            callback.on_train_begin(*args, **kwargs)

    def on_epoch_begin(self, *args, **kwargs):
        for callback in self.callbacks:
            callback.on_epoch_begin(*args, **kwargs)

    def on_batch_begin(self, *args, **kwargs):
        for callback in self.callbacks:
            callback.on_batch_begin(*args, **kwargs)

    def on_batch_end(self, *args, **kwargs):
        for callback in self.callbacks:
            callback.on_batch_end(*args, **kwargs)

    def on_epoch_end(self, *args, **kwargs):
        for callback in self.callbacks:
            callback.on_epoch_end(*args, **kwargs)

    def on_train_end(self, *args, **kwargs):
        for callback in self.callbacks:
            callback.on_train_end(*args, **kwargs)


class TrainMonitor(Callback):
    def __init__(self):
        super().__init__()
        self.averager = Averager()

    def on_batch_end(self, loss):
        self.averager.send(loss.item())

    def on_epoch_end(self):
        self.logger.info('loss: {}'.format(self.averager.value))

    def on_train_end(self):
        pass


class ValidMonitor(Callback):
    def __init__(self):
        super().__init__()

    def on_epoch_end(self, datagen_valid, device, model):
        for data in datagen_valid:
            X, y = data[0].to(device), data[1].to(device)
