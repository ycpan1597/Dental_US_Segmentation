"""
Adapted from https://github.com/meetshah1995/pytorch-semseg
"""

import os
import PIL
import cv2
import torch
import numpy as np
import albumentations as albu
from os.path import join as pjoin
import collections
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt

from torch.utils import data
from sklearn import preprocessing

# images in this dataset are not squares! This implies that the ultrasound images could work!
class CamVidDataset:
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.

    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. normalization, shape manipulation, etc.)

    """

    CLASSES = ['sky', 'building', 'pole', 'road', 'pavement',
               'tree', 'signsymbol', 'fence', 'car',
               'pedestrian', 'bicyclist', 'unlabelpythonled']

    def __init__(
            self,
            images_dir,
            masks_dir,
            classes=None,
            augmentation=None,
            preprocessing=None,
    ):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]

        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):

        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0)

        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

    def __len__(self):
        return len(self.ids)


class MRIDataset:  # This class is adapted from the Dataset class used in SMP's example

    CLASSES = ['background', 'tumor']  # can expand to more classes in the future

    def __init__(
            self,
            df,
            classes=None,
            augmentation=None,
            preprocessing=None,
    ):

        self.images_fps = df['image'].tolist()
        self.masks_fps = df['mask'].tolist()
        self.num_files = len(df)
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):

        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (256, 256))  # some images are 512 x 512

        mask = cv2.imread(self.masks_fps[i], 0)  # this loads it in grayscale; right now the mask is 255

        # We want the mask to be 1.0 in value, so we divide by max pixel value (some are 8 bits, some are 16)
        mask = mask/np.max(mask)
        mask = cv2.resize(mask, (256, 256))  # some images are 512 x 512

        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]  # one hot encoding
        mask = np.stack(masks, axis=-1).astype('float')

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

    def __len__(self):
        return self.num_files

class USDataset:

    CLASSES = ['background', 'ring-down', 'gel-cover', 'free gel', 'soft tissue',
               'muscle', 'bone', 'root', 'crown', 'suture', 'lip']

    def __init__(
            self,
            df,
            classes=None,
            augmentation=None,
            preprocessing=None,
            native_mask=False
    ):
        self.images_fps = df['image'].tolist()
        self.masks_fps = df['mask'].tolist()
        self.num_files = len(df)
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]

        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.native_mask = native_mask

    def __getitem__(self, i):

        new_H = 320
        new_W = 281

        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = cv2.resize(image, (426, 320))  # reduce image size but keep the aspect ratio; size is in (W, H) for cv2.resize
        image = cv2.resize(image, (new_W, new_H))

        mask = cv2.imread(self.masks_fps[i], 0)  # this loads it in grayscale

        if not self.native_mask:
            # extract certain classes from mask (e.g. cars)
            masks = [(mask == v) for v in self.class_values]  # one hot encoding
            mask = np.stack(masks, axis=-1).astype('float')

            resized_mask = np.zeros((new_H, new_W, mask.shape[2]))
            for i in range(mask.shape[2]):
                resized_mask[:, :, i] = cv2.resize(mask[:, :, i], (new_W, new_H))
            mask = resized_mask
        else:
            mask = cv2.resize(mask, (new_W, new_H))

        # # mask = cv2.resize(mask, (426, 320))
        # mask = cv2.resize(mask, (281, 320))

        # # extract certain classes from mask (e.g. cars)
        # masks = [(mask == v) for v in self.class_values]  # one hot encoding
        # mask = np.stack(masks, axis=-1).astype('float')

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

    def __len__(self):
        return self.num_files
