import numpy as np

import torch
import torch.utils.data as data


class GridworldData(data.Dataset):
    def __init__(self,
                 file,
                 imsize,
                 train=True,
                 transform=None,
                 target_transform=None):
        assert file.endswith('.npz')  # Must be .npz format
        self.file = file
        self.imsize = imsize
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # Training set or test set

        self.images, self.S1, self.S2, self.labels =  \
                                self._process(file, self.train)

    def __getitem__(self, index):
        img = self.images[index]
        s1 = self.S1[index]
        s2 = self.S2[index]
        label = self.labels[index]
        # Apply transform if we have one
        if self.transform is not None:
            img = self.transform(img)
        else:  # Internal default transform: Just to Tensor
            img = torch.from_numpy(img)
        # Apply target transform if we have one
        if self.target_transform is not None:
            label = self.target_transform(label)
        return img, int(s1), int(s2), int(label)

    def __len__(self):
        return self.images.shape[0]

    def _process(self, file, train):
        """Data format: A list, [train data, test data]
        Each data sample: label, S1, S2, Images, in this order.
        """
        with np.load(file, mmap_mode='r') as f:
            if train:
                images = f['arr_0']
                S1 = f['arr_1']
                S2 = f['arr_2']
                labels = f['arr_3']
            else:
                images = f['arr_4']
                S1 = f['arr_5']
                S2 = f['arr_6']
                labels = f['arr_7']
        # Set proper datatypes
        images = images.astype(np.float32)
        S1 = S1.astype(int)  # (S1, S2) location are integers
        S2 = S2.astype(int)
        labels = labels.astype(int)  # Labels are integers
        # Print number of samples
        if train:
            print("Number of Train Samples: {0}".format(images.shape[0]))
        else:
            print("Number of Test Samples: {0}".format(images.shape[0]))
        return images, S1, S2, labels
