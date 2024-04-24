# stdlib
import os
import re

# third party
import numpy as np
import torch
from torch.utils.data import Dataset

from .utils import preprocess_2d_img


class MedSamDataset(Dataset):
    """
    Dataset for training MedSAM on AE
    """

    def __init__(
        self, root: str, transform=None, target_size=256, include_3d=False, subset=None
    ):
        self.root = root
        self.transform = transform

        self.files = os.listdir(root)
        if not include_3d:
            self.files = [f for f in self.files if not f.startswith('3D')]
        self.target_size = target_size
        self.class_pattern = re.compile(r'^(?:\dDBox_)?(.*)_.*\.npz')
        self.classes = set([self.class_pattern.match(f).group(1) for f in self.files])  # type: ignore  # noqa
        self.classes_dict = {c: i for i, c in enumerate(self.classes)}
        self.classes_dict_rev = {i: c for i, c in enumerate(self.classes)}
        self.subset = subset

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(os.path.join(self.root, self.files[idx]))
        H, W = data['imgs'].shape[-2:]
        preproc = preprocess_2d_img(data['imgs'], self.target_size)
        newh, neww = preproc.shape[-2:]
        box = data['boxes']
        box = box / np.array([W, H, W, H]) * self.target_size
        c = self.class_pattern.match(self.files[idx]).group(1)
        ret = {
            'raw': data['imgs'],
            'image': preproc[0],
            'box': torch.Tensor(box),
            'original_size': (H, W),
            'new_size': (newh, neww),
            'filename': self.files[idx],
            'class': self.classes_dict[c],
        }
        if self.subset:
            ret = {k: ret[k] for k in self.subset}
        return ret
