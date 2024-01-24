import tensorflow.keras.backend as K
from tensorflow.keras.models import load_model
import numpy as np
import os

import zipfile
import pandas as pd
import random
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet_v2 import preprocess_input
from tensorflow.keras import backend as K

from scripts.prepare_cityscapes_data import get_all_labels,  get_color_to_label_mapping, get_color_to_id_mapping
from utils.predictions import predict_img, convert_idx_img_to_color, get_dataset_generators
import numpy as np
from PIL import Image


class GradCAM:
    def __init__(self, model, layerName=None):
        """
        model: pre-softmax layer (logit layer)
        """
        self.model = model
        self.layerName = layerName
            
        if self.layerName == None:
            self.layerName = self.find_target_layer()
    
    def find_target_layer(self):
        for layer in reversed(self.model.layers):
            if len(layer.output_shape) == 4:
                return layer.name
        raise ValueError("Could not find 4D layer. Cannot apply GradCAM")
            
    def compute_heatmap(self, image, class_idx, upsample_size, eps=1e-5):
        gradModel = Model(
            inputs = [self.model.inputs],
            outputs = [self.model.get_layer(self.layerName).output, self.model.output]
        )
        
        with tf.GradientTape() as tape:
            inputs = tf.cast(image, tf.float32)
            (convOuts, preds) = gradModel(inputs) # preds after softmax
            loss = preds[:,:,:,class_idx] #tf.math.reduce_max(preds,axis=3)# preds[:,classIdx]
        
        # compute gradients with automatic differentiation
        grads = tape.gradient(loss, convOuts)
        # discard batch
        convOuts = convOuts[0]
        grads = grads[0]
        norm_grads = tf.divide(grads, tf.reduce_mean(tf.square(grads)) + tf.constant(eps))
        
        # compute weights
        weights = tf.reduce_mean(norm_grads, axis=(0,1))
        cam = tf.reduce_sum(tf.multiply(weights, convOuts), axis=-1)
        
        # Apply reLU
        cam = np.maximum(cam, 0)
        cam = cam/np.max(cam)
        cam = cv2.resize(cam, upsample_size,interpolation=cv2.INTER_LINEAR)
        
        # convert to 3D
        cam3 = np.expand_dims(cam, axis=2)
        cam3 = np.tile(cam3, [1,1,3])
        
        return cam3
    
def overlay_gradCAM(img, cam3):
    cam3 = np.uint8(255*cam3)
    cam3 = cv2.applyColorMap(cam3, cv2.COLORMAP_JET)
    
    new_img = 0.3*cam3 + 0.5*img
    # new_img = 0.9*cam3 + 0.1*img
    
    return (new_img*255.0/new_img.max()).astype("uint8")


def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.25
    x = x.copy()
    x -= x.mean()
    x /= (x.std() + K.epsilon())
    x *= 0.25

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_data_format() == 'channels_first':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def show_gradCAMs(models, gradCAMs, models_names, image_dir,mask_dir, width=256, height=256, n=3):
    im_ls = os.listdir(image_dir)
    m_ls = os.listdir(mask_dir)
    """
    model: softmax layer
    """
    

    img = cv2.imread(os.path.join(image_dir,im_ls[n]))
    upsample_size = (img.shape[1],img.shape[0])
    color2label = get_color_to_label_mapping(get_all_labels())
    color2idx = get_color_to_id_mapping(get_all_labels())
    mask = Image.open(os.path.join(mask_dir,m_ls[n]))
    mask = np.unique(mask)
    converted_mask = convert_idx_img_to_color(tf.expand_dims(tf.expand_dims(mask, axis=0), axis=0))

    plt.subplots(figsize=(60, 10*len(mask) * 2))
    k=1

    for class_idx, color in zip(mask, converted_mask[0]):
        label = color2label[tuple(color)]
        print(class_idx, color, label)
            
        # Show original image
        plt.subplot(len(mask) * 2, 6, k)
        plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
        plt.title("Filename: {}, class: {}".format(im_ls[n], label), fontsize=20)
        plt.axis("off")
        # Show overlayed grad
        im = img_to_array(load_img(os.path.join(image_dir,im_ls[n]), target_size=(width,height)))
        x = np.expand_dims(im, axis=0)
        # x = preprocess_input(x)
        x = x/255.0
        
        for i, gradCAM in enumerate(gradCAMs):
            plt.subplot(len(mask)* 2, 6, k + i + 1)
            cam3 = gradCAM.compute_heatmap(image=x, class_idx=class_idx, upsample_size=upsample_size)

            new_img = overlay_gradCAM(img, cam3)
            new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)
            plt.imshow(new_img)
            plt.title(f"{models_names[i]}", fontsize=20)
            plt.axis("off")
       
        k += 6
        ### maski i predyckje 
        _mask = Image.open(os.path.join(mask_dir,m_ls[n]))
        _converted_mask = convert_idx_img_to_color(tf.expand_dims(tf.expand_dims(_mask, axis=0), axis=0))
        plt.subplot(len(mask) * 2, 6, k)
        plt.imshow(_converted_mask[0])
        plt.title("Original mask")
        plt.axis("off")

        for i, unet in enumerate(models):
            plt.subplot(len(mask)* 2, 6, k + i + 1)
            pred_img = predict_img(x[0], unet)
            colored_img = convert_idx_img_to_color(pred_img)
            plt.imshow(colored_img)
            plt.title(f"Prediction {models_names[i]}", fontsize=20)
            plt.axis("off")
        k += 6

    plt.show()
