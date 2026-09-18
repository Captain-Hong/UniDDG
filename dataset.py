import os

import numpy as np
import torch
from torch.utils import data

from utils.default_aug import apply_augmentations
from utils.tools import make_dataset, mask_to_onehot


class NpySegmentationDataset(data.Dataset):
    def __init__(self, root_list, file_list, aug=False, num_classes=3):
        self.imgs = make_dataset(root_list, file_list)
        self.aug = bool(aug)
        self.num_classes = int(num_classes)

        if not self.imgs:
            raise RuntimeError("No images found, please check the dataset")

    def __getitem__(self, index):
        img_path, mask_path = self.imgs[index]
        img = np.load(img_path).astype(np.float32) / 255.0
        mask = np.expand_dims(np.load(mask_path), axis=2)
        palette = [[class_index] for class_index in range(self.num_classes)]
        mask = mask_to_onehot(mask, palette=palette)

        if self.aug:
            img, mask = apply_augmentations(img, mask)

        img = np.clip(img, 0.0, 1.0).astype(np.float32)
        mask = np.clip(mask, 0.0, 1.0)
        mask = (mask > 0.5).astype(np.float32)

        img = torch.from_numpy(img.transpose(2, 0, 1).astype(np.float32))
        mask = torch.from_numpy(mask.transpose(2, 0, 1).astype(np.float32))

        return img, mask

    def __len__(self):
        return len(self.imgs)
