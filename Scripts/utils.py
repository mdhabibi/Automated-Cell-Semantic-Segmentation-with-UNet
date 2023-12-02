"""
utils.py

This script includes utility functions for the Cell-Segmentation-UNet project. 
These functions provide general-purpose utilities that can be used in various parts of the project, 
such as data visualization, performance evaluation, etc.

Functions:
- display_random_sample: Displays a random sample from the dataset, including the original image and its mask.

Author: Mahdi Habibi
"""

import numpy as np
import matplotlib.pyplot as plt
import random

def display_random_sample(X_train, Y_train):
    """
    Display a random sample from the dataset, including the original image and its corresponding mask.
    
    Parameters:
    - X_train (ndarray): Array of training images.
    - Y_train (ndarray): Array of corresponding mask images.
    """
    idx = random.randint(0, len(X_train) - 1)
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    ax[0].imshow(X_train[idx])
    ax[0].set_title('Original Image')

    ax[1].imshow(np.squeeze(Y_train[idx]))
    ax[1].set_title('Mask Image')

    plt.tight_layout()
    plt.show()

