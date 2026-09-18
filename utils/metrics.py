import numpy as np


def dice_coefficient_numpy(prediction, target):
    prediction = np.asarray(prediction, dtype=np.bool_)
    target = np.asarray(target, dtype=np.bool_)
    intersection = float(np.logical_and(prediction, target).sum())
    return (2.0 * intersection + 1.0) / (float(prediction.sum()) + float(target.sum()) + 1.0)


def dice_coeff_2label(prediction, target):
    prediction = np.asarray(prediction)
    if hasattr(target, "detach"):
        target = target.detach().cpu().numpy()
    else:
        target = np.asarray(target)

    if prediction.ndim == 3:
        return (
            dice_coefficient_numpy(prediction[1], target[1]),
            dice_coefficient_numpy(prediction[2], target[2]),
        )

    cup_scores = []
    disc_scores = []
    for index in range(prediction.shape[0]):
        cup_scores.append(dice_coefficient_numpy(prediction[index, 1], target[index, 1]))
        disc_scores.append(dice_coefficient_numpy(prediction[index, 2], target[index, 2]))
    return float(np.mean(cup_scores)), float(np.mean(disc_scores))
