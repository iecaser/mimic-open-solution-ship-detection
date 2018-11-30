import numpy as np
import cv2
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
from torchvision import transforms
from steppy.base import BaseTransformer
import warnings


class MetaReader(BaseTransformer):
    def __init__(self):
        super().__init__()
        pass

    def transform(self, meta):
        X = meta.file_path_image.values
        y = meta.is_not_empty.values
        return {'X': X,
                'y': y}


class OneClassClassificationDataset(Dataset):
    def __init__(self, X, y, dataset_params):
        super().__init__()
        self.X = X
        self.y = y
        self.transform = transforms.Compose([transforms.Resize((224, 224)),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=dataset_params.mean,
                                                                  std=dataset_params.std),
                                             ])

    def __getitem__(self, i):
        # Xi = cv2.imread(self.X[i], 1)
        # Xi = cv2.cvtColor(Xi, cv2.COLOR_BGR2RGB)
        Xi = Image.open(self.X[i])
        Xi = self.transform(Xi)
        yi = self.y[i]
        return Xi, int(yi)

    def __len__(self):
        return self.y.shape[0]


class BalancedSampler(Sampler):
    def __init__(self, data_source, sample_size, empty_fraction):
        super().__init__(data_source)
        self.sample_size = sample_size
        self.data_size = len(data_source)
        self._check_size()
        self.data_source = data_source
        self.empty_size = int(empty_fraction * self.sample_size)
        self.non_empty_size = self.sample_size - self.empty_size

    def __iter__(self):
        return iter(self._get_samples())

    def __len__(self):
        return self.sample_size

    def _get_samples(self):
        empty_indices = np.where(self.data_source == 0)[0]
        non_empty_indices = np.where(self.data_source == 1)[0]
        resampled_empty_indices = np.random.choice(empty_indices, self.empty_size)
        resampled_non_empty_indices = np.random.choice(non_empty_indices, self.non_empty_size)
        samples = np.r_[resampled_empty_indices, resampled_non_empty_indices]
        np.random.shuffle(samples)
        return samples

    def _check_size(self):
        """Check size for dev mode because valid data doesn't use sample."""
        if self.sample_size > self.data_size:
            self.sample_size = self.data_size // 2
            warnings.warn('Sample size is bigger than Data size, using ``sample_size=data_size/2``.')


class OneClassClassificationLoader(BaseTransformer):
    def __init__(self, dataset_params, loader_params):
        super().__init__()
        self.loader_params = loader_params
        self.dataset_params = dataset_params

    def transform(self, X, y, X_valid=None, y_valid=None):
        datagen = self.get_datagen(X, y, True, self.loader_params.train)
        if X_valid is not None and y_valid is not None:
            datagen_valid = self.get_datagen(X_valid, y_valid, False, self.loader_params.inference)
        else:
            datagen_valid = None
        return {'datagen': datagen,
                'datagen_valid': datagen_valid}

    def get_datagen(self, X, y, train_mode, loader_params):
        if train_mode:
            sampler = BalancedSampler(data_source=y,
                                      sample_size=self.dataset_params.sample_size,
                                      empty_fraction=self.dataset_params.empty_fraction,
                                      )
            sampler = None
            flow = DataLoader(dataset=OneClassClassificationDataset(X, y, self.dataset_params),
                              sampler=sampler,
                              **loader_params,
                              )
        else:
            flow = DataLoader(dataset=OneClassClassificationDataset(X, y, self.dataset_params),
                              **loader_params,
                              )
        return flow
