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
            X.append(img_window)
            y.append(img_label)
    
    return np.array(X), np.array(y)


if __name__ == '__main__':
    images = read_images('../images/CHASE/img')
    rgb_images = convert_images_to_rgb(images)

    green_images = select_channel(rgb_images, 1)

    labels = read_labels('../images/CHASE/labels')

    windowed_images = np.array([create_sliding_window(img, (5, 5)) for img in green_images])
    
    windows_and_labels = zip(windowed_images, labels)

    X, y = create_dataset(windows_and_labels)

    print(X.shape, y.shape)
