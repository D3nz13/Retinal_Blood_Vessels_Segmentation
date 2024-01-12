from multiprocessing import Pool
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
    imgs, masks = zip(*img_and_masks)

    return imgs, masks


def encode_mask(mask: NDArray, mapping: Dict[Tuple[int, ...], int]) -> NDArray:
    distances = np.zeros((HEIGHT, WIDTH, len(mapping)))

    for i, color in enumerate(mapping.keys()):
        distances[:, :, i] = np.sum((mask - color) ** 2, axis=-1)

    return np.argmin(distances, axis=2)


def save_image(img: NDArray, saving_path: Path, grayscale: bool = False) -> None:
    if grayscale:
        img = img.astype(np.uint8)
    img_pil = Image.fromarray(img)
    img_pil.save(saving_path)


def preprocess_and_save_data(source_dir: Path, saving_dir: Path) -> None:
    saving_dir.mkdir(parents=True, exist_ok=True)

    all_images, all_masks = read_data(source_dir)

    mapping = get_color_to_id_mapping(get_all_labels())
    encoding_arguments = ((mask, mapping) for mask in all_masks)

    with Pool() as p:
        encoded_masks = p.starmap(encode_mask, encoding_arguments)

    (saving_dir / "img").mkdir(exist_ok=True)
    (saving_dir / "mask").mkdir(exist_ok=True)

    image_saving_arguments = (
        (img, saving_dir / f"img/{i}.png") for i, img in enumerate(all_images, start=1)
    )
    mask_saving_arguments = (
        (mask, saving_dir / f"mask/{i}.png", True) for i, mask in enumerate(encoded_masks, start=1)
    )

    with Pool() as p:
        p.starmap(save_image, image_saving_arguments)
        p.starmap(save_image, mask_saving_arguments)


if __name__ == "__main__":
    from time import time

    starting_time = time()
    preprocess_and_save_data(
        Path("./cityscapes_data/train/dummy_folder/"), Path("./cityscapes_data_preprocessed/train/")
    )
    print(time() - starting_time)

    starting_time = time()
    preprocess_and_save_data(
        Path("./cityscapes_data/val/dummy_folder/"), Path("./cityscapes_data_preprocessed/val/")
    )
    print(time() - starting_time)
