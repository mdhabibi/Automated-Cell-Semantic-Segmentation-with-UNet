"""
data_loader.py

This script contains functions for handling various data processing tasks for the 
Cell-Segmentation-UNet project. It includes functions for unzipping datasets, retrieving data paths, 
preprocessing images and masks, and loading datasets into a suitable format for training and testing.

Functions:
- unzip_to_original_subfolders: Unzips all files within a directory to their respective subfolders.
- get_data_paths: Retrieves paths to folders named by unique IDs within a specified directory.
- preprocess_image: Processes an image file to fit the input requirements of the model.
- preprocess_mask: Processes mask files associated with each image for segmentation tasks.
- load_dataset: Loads and preprocesses the dataset from a specified directory for model training.

Author: Mahdi Habibi
"""

import os
import numpy as np
from skimage.io import imread
from skimage.transform import resize

def unzip_to_original_subfolders(root_directory):
    """
    Unzip all files within a directory to their respective subfolders.
    
    Parameters:
    - root_directory (str): Path to the root directory containing zip files.
    """
    # Walking through the root directory to find all files
    for dirpath, _, filenames in os.walk(root_directory):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            # Extract the file name without the extension to create a subfolder
            folder_name = os.path.splitext(filename)[0]
            destination_folder = os.path.join(dirpath, folder_name)
            # Unzipping to the created subfolder
            os.system(f'unzip -q {file_path} -d {destination_folder}')
            # remove the zip file after extracting
            os.remove(file_path)
            
def get_data_paths(train_path):
    """
    Retrieve paths to the folders named by IDs within the specified training path.
    
    Parameters:
    - train_path (str): Path to the training data directory.
    
    Returns:
    - list: A list of paths to the individual data folders.
    """
    return next(os.walk(train_path))[1]

def preprocess_image(id_, train_path, img_height, img_width, img_channels):
    """
    Preprocess an image file to fit the input requirements of the model.
    
    Parameters:
    - id_ (str): Unique identifier for the image.
    - train_path (str): Path to the training data directory.
    - img_height (int): Desired image height.
    - img_width (int): Desired image width.
    - img_channels (int): Number of image channels.
    
    Returns:
    - ndarray: Processed image array.
    """
    img_file_path = os.path.join(train_path, id_, "images", id_ + ".png")
    img = imread(img_file_path)[:, :, :img_channels]
    img = resize(img, (img_height, img_width), mode='constant', preserve_range=True)
    return img

def preprocess_mask(id_, train_path, img_height, img_width):
    """
    Process mask files associated with each image for segmentation tasks.
    
    Parameters:
    - id_ (str): Unique identifier for the image.
    - train_path (str): Path to the training data directory.
    - img_height (int): Desired mask height.
    - img_width (int): Desired mask width.
    
    Returns:
    - ndarray: Combined mask array for the given image.
    """
    mask_file_dir = os.path.join(train_path, id_, "masks")
    all_masks = os.listdir(mask_file_dir)
    mask = np.zeros((img_height, img_width, 1), dtype=np.bool)
    for mask_file in all_masks:
        mask_path = os.path.join(mask_file_dir, mask_file)
        mask_ = imread(mask_path)
        mask_ = np.expand_dims(resize(mask_, (img_height, img_width), mode='constant', preserve_range=True), axis=-1)
        mask = np.maximum(mask, mask_)
    return mask

def load_dataset(train_path, img_height, img_width, img_channels):
    """
    Loads and preprocesses the dataset from the specified directory for model training.
    
    Parameters:
    - train_path (str): Path to the training data directory.
    - img_height (int): Desired image height.
    - img_width (int): Desired image width.
    - img_channels (int): Number of image channels.
    
    Returns:
    - tuple: Tuple containing arrays for training images and corresponding masks.
    """
    train_ids = get_data_paths(train_path)

    X_train = np.zeros((len(train_ids), img_height, img_width, img_channels), dtype=np.uint8)
    Y_train = np.zeros((len(train_ids), img_height, img_width, 1), dtype=np.bool)

    for n, id_ in enumerate(train_ids):
        X_train[n] = preprocess_image(id_, train_path, img_height, img_width, img_channels)
        Y_train[n] = preprocess_mask(id_, train_path, img_height, img_width)

    return X_train, Y_train
