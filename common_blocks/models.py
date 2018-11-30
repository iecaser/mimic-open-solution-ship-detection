from steppy.base import BaseTransformer
from torchvision.models import densenet201, resnet18
from torch import nn
import torch
from common_blocks import callback
import ipdb
from tqdm import tqdm


class MyDenseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.densenet = densenet201(pretrained=True)
        print(self.densenet)
        self.fc = nn.Linear(in_features=1000, out_features=2)
        self.softmax = nn.Softmax()

    def forward(self, input):
        x = self.densenet(input)
        x = self.fc(x)
        x = self.softmax(x)
        return x


class MyResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = resnet18(pretrained=True)
        self.fc = nn.Linear(in_features=1000, out_features=2)
        self.softmax = nn.Softmax()

    def forward(self, input):
        x = self.resnet(input)
        x = self.fc(x)
        x = self.softmax(x)
        return x


class BinaryModel(BaseTransformer):
    def __init__(self, epochs=999999):
        super().__init__()
        self.epochs = epochs
        self.device = torch.device('cuda:1')
        self.model = None
        self.criterion = None
        self.callbacks = None
        self.init()

    def init(self):
        self._set_callbacks()
        self._set_model()
        self._set_criterion()
        self._set_optimizer()

    def fit(self, datagen, datagen_valid):
        self.callbacks.on_train_begin()
        self.model = self.model.to(self.device)
        for epoch in range(self.epochs):
            self.callbacks.on_epoch_begin()
            for data in tqdm(datagen):
                self.callbacks.on_batch_begin()
                loss = self._fit_batch(data)
                self.callbacks.on_batch_end(loss)
            self.callbacks.on_epoch_end()
        self.callbacks.on_train_end()

    def _fit_batch(self, data):
        X, y = data[0].to(self.device), data[1].to(self.device)
        output = self.model(X)
        loss = self.criterion(output, y)
        _, y_ = torch.max(output, 1)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def transform(self, datagen):
        for data in datagen:
            pass

    def _transform(self, datagen):
        pass

    def _set_callbacks(self):
        logger_callback = callback.TrainMonitor()
        self.callbacks = callback.CallbackContainer(logger_callback)

    def _set_optimizer(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=1e-4,
                                          )

    def _set_model(self):
        # self.model = MyDenseNet()
        # self.model = MyResNet()
        model = resnet18(pretrained=True)
        model.fc = nn.Linear(in_features=512, out_features=2)
        self.model = model

    def _set_criterion(self):
        self.criterion = nn.CrossEntropyLoss()
