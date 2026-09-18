import random
import numpy as np
import cv2
import scipy.ndimage as nd


def random_flip(image, label):
    if random.random() > 0.5:
        image, label = np.flip(image, axis=1), np.flip(label, axis=1)
    if random.random() > 0.5:
        image, label = np.flip(image, axis=0), np.flip(label, axis=0)
    return image, label


def random_rotate(image, label):
    k = random.randint(0, 3)
    return np.rot90(image, k=k, axes=(0, 1)), np.rot90(label, k=k, axes=(0, 1))


def random_scale(image, label):
    scale = random.uniform(0.8, 1.2)
    h, w = image.shape[:2]
    new_h, new_w = int(h * scale), int(w * scale)

    image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    label = cv2.resize(label, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    if label.ndim == 2:
        label = np.expand_dims(label, axis=-1)

    if scale < 1:
        pad_top, pad_bottom = (h - new_h) // 2, h - new_h - (h - new_h) // 2
        pad_left, pad_right = (w - new_w) // 2, w - new_w - (w - new_w) // 2
        image = np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='constant')
        label = np.pad(label, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='constant')
    else:
        center_h, center_w = new_h // 2, new_w // 2
        offset_h, offset_w = h // 2, w // 2
        image = image[center_h - offset_h:center_h + offset_h, center_w - offset_w:center_w + offset_w]
        label = label[center_h - offset_h:center_h + offset_h, center_w - offset_w:center_w + offset_w]
    return image, label


def random_shift(image, label):
    shift = random.randint(-50, 50)
    shifted_image = nd.shift(image, (0, shift, 0), mode='constant', order=1)
    shifted_label = nd.shift(label, (0, shift, 0), mode='constant', order=0)
    return shifted_image, shifted_label


def random_noise(image):
    noise = np.random.normal(0, random.uniform(0, 0.1), image.shape).astype(image.dtype)
    return np.clip(image + noise, 0, 1)


def random_brightness(image):
    return np.clip(image * random.uniform(0.5, 1.9), 0, 1)


def random_channel_swap(image):
    return np.take(image, np.random.permutation(3), axis=-1)


def apply_augmentations(image, label):
    augmentations = [
        lambda x, y: (x, y),
        random_flip,
        random_rotate,
        random_scale,
        random_shift,
        lambda x, y: (random_noise(x), y),
        lambda x, y: (random_brightness(x), y),
        lambda x, y: (random_channel_swap(x), y),
    ]
    aug_fn = random.choice(augmentations)
    image, label = aug_fn(image, label)

    image = np.clip(image, 0.0, 1.0).astype(np.float32)
    label = np.clip(label, 0.0, 1.0)
    label = (label > 0.5).astype(np.float32)
    return image, label
