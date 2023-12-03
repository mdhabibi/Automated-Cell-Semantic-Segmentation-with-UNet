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

def display_sample_grid(X_train, Y_train, num_samples=3):
    """Display a grid of microscopy images and their associated masks."""
    fig, axs = plt.subplots(2, num_samples, figsize=(12, 8))  

    for i in range(num_samples):
        idx = random.randint(0, len(X_train) - 1)
        
        # Display original image
        axs[0, i].imshow(X_train[idx])
        axs[0, i].set_title(f'Original Image {i+1}')
        axs[0, i].axis('off')  # Turn off axis

        # Display corresponding mask
        axs[1, i].imshow(np.squeeze(Y_train[idx]))
        axs[1, i].set_title(f'Mask Image {i+1}')
        axs[1, i].axis('off')  # Turn off axis

    plt.tight_layout()
    plt.show()


