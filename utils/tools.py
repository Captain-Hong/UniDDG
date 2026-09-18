import os

import numpy as np
import torch


def mask_to_onehot(mask, palette=None):
    if palette is None:
        palette = [[0], [1], [2]]
    height, width, _ = mask.shape
    onehot_mask = np.zeros((height, width, len(palette)), dtype=np.float32)
    for index, color in enumerate(palette):
        onehot_mask[:, :, index] = np.all(mask == color, axis=-1).astype(np.float32)
    return onehot_mask


def make_dataset(root_list, file_list):
    items = []
    for root, filename in zip(root_list, file_list):
        image_dir = os.path.join(root, "image")
        mask_dir = os.path.join(root, "mask")
        with open(os.path.join(root, filename), encoding="utf-8") as file_obj:
            names = [line.strip() for line in file_obj if line.strip()]
        items.extend((os.path.join(image_dir, name), os.path.join(mask_dir, name)) for name in names)
    return items


def postprocessing(prediction, threshold=0.75, dataset=None):
    del dataset
    prediction = torch.sigmoid(prediction).detach().cpu().numpy()
    processed = np.copy(prediction)
    processed[2] = (prediction[2] > threshold).astype(np.uint8)
    processed[1] = (prediction[1] > threshold).astype(np.uint8)
    return processed
