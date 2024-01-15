from functools import lru_cache
from typing import Dict, Tuple

import numpy as np
import tensorflow as tf
from keras.src.engine.functional import Functional
from numpy.typing import NDArray

from scripts.prepare_cityscapes_data import get_all_labels, get_color_to_id_mapping


def predict_img(img: NDArray, model: Functional, apply_argmax: bool = True):
    raw_pred = model(tf.expand_dims(img, axis=0))
    return tf.argmax(raw_pred, axis=-1) if apply_argmax else raw_pred


@lru_cache(1)
def get_index_to_color_mapping() -> Dict[int, Tuple[int, ...]]:
    color_to_id_mapping = get_color_to_id_mapping(get_all_labels())

    return {idx: color for idx, (color, _) in enumerate(color_to_id_mapping.items())}


def convert_idx_img_to_color(img: NDArray) -> NDArray:
    idx_to_color_mapping = get_index_to_color_mapping()
    shape = img.shape

    if shape[0] == 1:  # argmax prediction
        res_img = np.zeros((*shape[1:], 3), dtype=int)
        for idx, color in idx_to_color_mapping.items():
            res_img[img[0, :, :] == idx, :] = color
    else:  # mask from the dataset
        res_img = np.zeros((*shape[:-1], 3), dtype=int)
        for idx, color in idx_to_color_mapping.items():
            res_img[img[:, :, 0] == idx, :] = color

    return res_img.astype(int)
