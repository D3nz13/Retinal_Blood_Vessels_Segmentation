from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from scripts.constants import HEIGHT, WIDTH, Label, get_all_labels


def get_color_to_label_mapping(labels: List[Label]) -> Dict[Tuple[int, ...], str]:
    res = {label.color: label.name for label in labels}
    return res


def get_color_to_id_mapping(labels: List[Label]) -> Dict[Tuple[int, ...], int]:
    res = {label.color: label.id for label in labels}
    return res


def read_img(path: Path) -> Tuple[NDArray, NDArray]:
    raw_img = np.array(Image.open(path))
    img, mask = raw_img[:, :WIDTH, :], raw_img[:, WIDTH:, :]

    return img, mask


def read_data(data_dir: Path) -> Tuple[List[NDArray], List[NDArray]]:
    all_img_names = data_dir.iterdir()
    img_and_masks = (read_img(f_path) for f_path in all_img_names)
    imgs, masks = list(zip(*img_and_masks))

    return imgs, masks


def encode_mask(mask: NDArray, mapping: Dict[Tuple[int, ...], int]) -> NDArray:
    distances = np.zeros((HEIGHT, WIDTH, len(mapping)))

    for i, color in enumerate(mapping.keys()):
        distances[:, :, i] = np.sum((mask - color) ** 2, axis=-1)

    return np.argmin(distances, axis=2)


if __name__ == "__main__":
    from multiprocessing import Pool
    from time import time

    starting_time = time()
    images, masks = read_data(Path("./cityscapes_data/train/dummy_folder/"))
    print(time() - starting_time)

    mapping = get_color_to_id_mapping(get_all_labels())

    arguments = ((mask, mapping) for mask in masks)
    starting_time = time()
    with Pool() as p:
        encoded_masks = p.starmap(encode_mask, arguments)
    print(time() - starting_time)
