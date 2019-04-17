import gzip
import numpy as np
import torch
from torch.utils.data import Dataset


class Split_MNIST(Dataset):

    def __init__(self, images_path, labels_path):
        self.images = self.load_images(images_path)
        self.labels = self.load_labels(labels_path)
        self.eye = torch.eye(10)

    def load_images(self, fp):
        with gzip.open(fp, 'rb') as f:
            data = f.read()
        num = int.from_bytes(data[4:8], byteorder='big', signed=True)
        row = int.from_bytes(data[8:12], byteorder='big', signed=True)
        col = int.from_bytes(data[12:16], byteorder='big', signed=True)
        images = np.fromstring(data[16:], dtype=np.uint8).reshape(num, row, col)
        return images

    def load_labels(self, fp):
        with gzip.open(fp, 'rb') as f:
            data = f.read()
        num = int.from_bytes(data[4:8], byteorder='big', signed=True)
        labels = np.fromstring(data[8:], dtype=np.uint8)
        return labels

    def __getitem__(self, index):
        image = self.images[index]
        image = torch.from_numpy(image).to(torch.float32).unsqueeze(0) / 255.
        label = self.labels[index]
        label = self.eye[label]
        return image, label

    def __len__(self):
        return self.images.shape[0]


class MNIST(object):

    def __init__(self, data_dir):
        self.train_set = Split_MNIST(
            data_dir+'train-images-idx3-ubyte.gz',
            data_dir+'train-labels-idx1-ubyte.gz'
        )
        self.test_set = Split_MNIST(
            data_dir+'t10k-images-idx3-ubyte.gz',
            data_dir+'t10k-labels-idx1-ubyte.gz'
        )

    def train(self):
        return self.train_set

    def test(self):
        return self.test_set
