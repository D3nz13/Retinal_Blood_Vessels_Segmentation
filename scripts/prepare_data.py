import os

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import cv2
import matplotlib.pyplot as plt


def read_images(dir: str) -> list:
    all_image_names = os.listdir(dir)
    all_images = [cv2.imread(f'{dir}/{img_name}') for img_name in all_image_names]

    return all_images


def read_labels(dir: str) -> list:
    all_label_names = os.listdir(dir)
    all_labels = [cv2.imread(f'{dir}/{label_name}', cv2.IMREAD_GRAYSCALE) for label_name in all_label_names]

    return all_labels


def resize_images(images: list, shape: tuple) -> list:
    try:
        width, height = shape

        return [cv2.resize(image, (width, height)) for image in images]
    except:
        raise Exception('Incorrect shape/input images.')


def convert_images_to_rgb(images: list) -> np.ndarray:
    rgb_images = [cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB) for bgr_image in images]

    return np.array(rgb_images)


def select_channel(images: np.ndarray, channel: int) -> np.ndarray:
    try:
        return images[:, :, :, channel]
    except:
        raise Exception('Incorrect channel number.')


def create_sliding_window(image: np.ndarray, window_shape: tuple, pad=True, padding=(2, 2)) -> np.ndarray:
    if pad:
        image = np.pad(image, pad_width=padding, mode='constant', constant_values=(0, 0))
    
    return sliding_window_view(image, window_shape=window_shape).reshape(-1, *window_shape)
    

def create_dataset(windows_and_labels):
    X = []
    y = []

    for img_windows, img_labels in windows_and_labels:
        img_labels = img_labels.flatten()
        for img_window, img_label in zip(img_windows, img_labels):
            #  potentially add window transformation here
            window_moments = cv2.moments(img_window, binaryImage=False)
            hu_moments = cv2.HuMoments(window_moments)
            variable = np.hstack((img_window.flatten(), hu_moments.flatten()))

            X.append(variable)
            y.append(img_label)
    
    return np.array(X), np.array(y)


def create_dataset_from_directory(dir: str, channel=1, shape=None, window_shape=(5, 5), pad=True, padding=(2, 2)) -> tuple:
    try:
        images = read_images(f'{dir}/img')
        masks = read_labels(f'{dir}/mask')
        n_images = len(images)
    except:
        raise Exception('Incorrect directory !')
    
    if shape:
        if isinstance(shape, tuple):
            images = resize_images(images, shape)
        else:
            raise Exception('Shape must be a tuple of ints.')

    images = convert_images_to_rgb(images)

    one_channel_images = select_channel(images, channel)

    X, y = [], []

    for i, (img, mask) in enumerate(zip(one_channel_images, masks), start=1):
        print(f"Image {i}/{n_images}")
        mask = mask.flatten()
        img_windows = create_sliding_window(img, window_shape, pad, padding)
        for img_window, window_mask in zip(img_windows, mask):
            window_moments = cv2.moments(img_window, binaryImage=False)
            hu_moments = cv2.HuMoments(window_moments)
            variable = np.hstack((img_window.flatten(), hu_moments.flatten()))

            X.append(variable)
            y.append(window_mask)
    
    return np.array(X), np.array(y)


def crop_images(source_dir: str, destination_dir: str) -> None:
    """Crops images to (960, 960) shape

    Args:
        source_dir (str): directory with images
        destination_dir (str): directory where the cropped images will be saved
    """
    image_names = os.listdir(source_dir)

    for img_name in image_names:
        img = cv2.imread(f"{source_dir}/{img_name}")
        cropped = img[:, 20:-19]
        cv2.imwrite(f"{destination_dir}/{img_name}", cropped)

        print(f"Image {img_name} successfully cropped and saved.")

if __name__ == '__main__':
    # X_train, y_train = create_dataset_from_directory(dir="../images/train", channel=1, shape=(246, 256), window_shape=(5, 5), pad=True, padding=(2, 2))

    # print(X_train.shape, y_train.shape)

    crop_images(source_dir="../images/train/img", destination_dir="../cropped_images/train/img")
    crop_images(source_dir="../images/train/mask", destination_dir="../cropped_images/train/mask")
    crop_images(source_dir="../images/test/img", destination_dir="../cropped_images/test/img")
    crop_images(source_dir="../images/test/mask", destination_dir="../cropped_images/test/mask")
