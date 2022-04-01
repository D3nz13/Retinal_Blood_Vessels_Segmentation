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
            X.append(img_window.flatten())
            y.append(img_label)
    
    return np.array(X), np.array(y)


def create_dataset_from_directory(dir: str, channel=1, window_shape=(5, 5), pad=True, padding=(2, 2)) -> tuple:
    try:
        images = read_images(f'{dir}/img')
        labels = read_labels(f'{dir}/labels')
    except:
        raise Exception('Incorrect directory !')
    
    images = convert_images_to_rgb(images)

    one_channel_images = select_channel(images, channel)
    images_windows = np.array([create_sliding_window(img, window_shape, pad, padding) for img in one_channel_images])

    windows_and_labels = zip(images_windows, labels)

    return create_dataset(windows_and_labels)


if __name__ == '__main__':
    X, y = create_dataset_from_directory(dir="../images/CHASE", channel=1, window_shape=(5, 5), pad=True, padding=(2, 2))

    print(X.shape, y.shape)
