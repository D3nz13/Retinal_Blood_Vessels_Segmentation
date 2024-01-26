import dataclasses as dc
from collections import defaultdict
from functools import lru_cache
from json import dump
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf
from keras.src.engine.functional import Functional
from numpy.typing import NDArray

from scripts.constants import EVALUATION_DIR, NUM_CLASSES
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
        # TODO: requires argmax
        res_img = np.zeros((*shape[:-1], 3), dtype=int)
        for idx, color in idx_to_color_mapping.items():
            res_img[img[:, :, 0] == idx, :] = color

    return res_img.astype(int)


@lru_cache(1)
def get_dataset_generators(
    data_dir: Path = Path("../cityscapes_data_preprocessed/"),
    batch_size: int = 4,
    datagen_seed: int = 24,
    num_classes: int = 29,
    shape: Tuple[int, int] = (256, 256),
):
    train_dir, val_dir = data_dir / "train", data_dir / "val"

    img_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)
    mask_datagen = tf.keras.preprocessing.image.ImageDataGenerator()

    train_image_datagen = img_datagen.flow_from_directory(
        train_dir / "img/", class_mode=None, batch_size=batch_size, seed=datagen_seed
    )
    train_mask_datagen = mask_datagen.flow_from_directory(
        train_dir / "mask/",
        class_mode=None,
        batch_size=batch_size,
        seed=datagen_seed,
        color_mode="grayscale",
    )

    train_mask_generator = tf.data.Dataset.from_generator(
        lambda: train_mask_datagen, output_types=tf.float32, output_shapes=(batch_size, *shape, 1)
    ).map(
        lambda x: tf.reshape(
            tf.one_hot(tf.cast(x, tf.uint8), depth=num_classes), (batch_size, *shape, num_classes)
        )
    )

    train_set = zip(train_image_datagen, train_mask_generator)

    val_image_datagen = img_datagen.flow_from_directory(
        val_dir / "img/", class_mode=None, batch_size=batch_size, seed=datagen_seed
    )
    val_mask_datagen = mask_datagen.flow_from_directory(
        val_dir / "mask/",
        class_mode=None,
        batch_size=batch_size,
        seed=datagen_seed,
        color_mode="grayscale",
    )
    val_mask_generator = tf.data.Dataset.from_generator(
        lambda: val_mask_datagen, output_types=tf.float32, output_shapes=(batch_size, *shape, 1)
    ).map(
        lambda x: tf.reshape(
            tf.one_hot(tf.cast(x, tf.uint8), depth=num_classes), (batch_size, *shape, num_classes)
        )
    )

    val_set = zip(val_image_datagen, val_mask_generator)

    return train_set, val_set


def calculate_tp_tn_fp_fn_for_class(y_pred_argmax, y_true_argmax, class_idx):
    X_pred_argmax_class = y_pred_argmax == class_idx
    y_true_argmax_class = y_true_argmax == class_idx

    tp = np.sum(X_pred_argmax_class & y_true_argmax_class)
    tn = np.sum(~X_pred_argmax_class & ~y_true_argmax_class)
    fp = np.sum(X_pred_argmax_class & ~y_true_argmax_class)
    fn = np.sum(~X_pred_argmax_class & y_true_argmax_class)

    return np.array([tp, tn, fp, fn])


@dc.dataclass
class TpTnFpFnResults:
    results: Dict[str, List[int]]


def evaluate_tp_tn_fp_fn(model, val_set, name):
    """Val set's batch size is assumed to be 1."""
    res = defaultdict(lambda: np.array([0, 0, 0, 0], dtype="int"))

    for i, (X, y_true) in enumerate(val_set, start=1):
        if i == 501:
            break
        y_pred = model.predict_on_batch(X)[0]
        y_true = y_true[0]
        y_pred_argmax = tf.argmax(y_pred, axis=-1)
        y_true_argmax = tf.argmax(y_true, axis=-1)

        for idx in range(NUM_CLASSES + 1):
            res[str(idx)] += calculate_tp_tn_fp_fn_for_class(y_pred_argmax, y_true_argmax, idx)

    res = {k: list(v) for (k, v) in res.items()}
    res = TpTnFpFnResults(results=res)

    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    with open(EVALUATION_DIR / f"{name}_tp_tn_fp_fn.json", "w") as f:
        dump(dc.asdict(res), f, default=str)

    return res
