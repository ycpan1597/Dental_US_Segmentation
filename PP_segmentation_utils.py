import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as albu
import matplotlib.pyplot as plt

# This function is used to threshold images in the dataset where the mask is greater than the pos_thresh below
# The purpose is to work on a simpler problem (ignore the more challenging cases for now) and focus on getting
# a high IOU for masks that are at least pos_thresh of the whole image
def threshold_images(root, pos_thresh=0.005):
    imgPath = os.path.join(root, 'imagePNG')
    maskPath = os.path.join(root, 'maskPNG')

    imgDir = os.listdir(imgPath)
    maskDir = os.listdir(maskPath)

    imgList, maskList = [], []

    new_img_dir = os.path.join(root, 'imagePNG_pruned')
    new_mask_dir = os.path.join(root, 'maskPNG_pruned')

    for i in range(len(imgDir)):
        if i % 100 == 0:
            print(i)
        img_fn = imgDir[i]
        mask_fn = maskDir[i]
        if (not img_fn.startswith('.')) and (not mask_fn.startswith('.')):
            full_img_fp = os.path.join(root, 'imagePNG', img_fn)
            full_mask_fp = os.path.join(root, 'maskPNG', mask_fn)

            mask = cv2.imread(full_mask_fp, 0)
            if (np.sum(mask)/np.max(mask) / np.size(mask)) > pos_thresh:
                imgList.append(full_img_fp)
                maskList.append(full_mask_fp)
                img = cv2.imread(full_img_fp, 0)
                cv2.imwrite(os.path.join(new_img_dir, img_fn), img)
                cv2.imwrite(os.path.join(new_mask_dir, mask_fn), mask)
    df = pd.DataFrame({'image': imgList, 'mask': maskList})
    return df


def prepare_US_csv(root, tv_ratio, num_files=None, mode=None, cropped=True):

    if cropped:
        print('using cropped images!')
        img_folder = 'imagePNG_cropped'
        mask_folder = 'maskPNG_cropped'
    else:
        img_folder = 'imagePNG'
        mask_folder = 'maskPNG'

    imgList, maskList = [], []
    imgPath = os.path.join(root, img_folder)
    maskPath = os.path.join(root, mask_folder)

    imgDir = sorted(os.listdir(imgPath))
    maskDir = sorted(os.listdir(maskPath))

    MAX_NUM_FILES = int(len(imgDir) * 0.8)  # This way the model can improve as our dataset grows

    if sum(tv_ratio) != 1:
        raise ValueError("Train-validate ratio must add up to 1")

    if (num_files == -1 or num_files > MAX_NUM_FILES):
        print('Using maximum number of training files: {}'.format(MAX_NUM_FILES))
        num_files = MAX_NUM_FILES  # all files

    if mode == "train":
        idxes = np.random.choice(range(MAX_NUM_FILES), num_files, replace=False)
        for i in range(num_files):
            img = imgDir[idxes[i]]
            mask = maskDir[idxes[i]]
            if (not img.startswith('.')) and (not mask.startswith('.')):
                imgList.append(os.path.join(root, img_folder, img))
                maskList.append(os.path.join(root, mask_folder, img))  # file name is the exact same, so just use it!

        df = pd.DataFrame({'image': imgList, 'mask': maskList})
        train_num = int(len(df) * tv_ratio[0])  # Number of training data
        dfTrain = df[:train_num]
        dfVal = df[train_num:-1]
        return dfTrain, dfVal
    else:
        num_test_files = len(imgDir) - MAX_NUM_FILES
        testImgList = []
        testMaskList = []
        # Define dictionary for test files (out of 6820 files)
        test_idxes = np.random.choice(range(MAX_NUM_FILES, len(imgDir)), num_test_files, replace=False)
        for i in range(len(test_idxes)):
            img = imgDir[test_idxes[i]]
            mask = maskDir[test_idxes[i]]
            if (not img.startswith('.')) and (not mask.startswith('.')):
                testImgList.append(os.path.join(root, img_folder, img))
                testMaskList.append(os.path.join(root, mask_folder, img))
        dfTest = pd.DataFrame({'image': testImgList, 'mask': testMaskList})
        return dfTest


# Create csv for matching images and ground truths
def prepare_csv(root, tv_ratio, num_files=None, mode=None):
    MAX_NUM_FILES = 4000

    if sum(tv_ratio) != 1:
        raise ValueError("Train-validate ratio must add up to 1")

    if num_files == -1 or num_files > MAX_NUM_FILES:
        print('Using maximum number of training files: {}'.format(MAX_NUM_FILES))
        num_files = MAX_NUM_FILES  # all files

    imgList, maskList = [], []

    imgPath = os.path.join(root, 'imagePNG_pruned')
    maskPath = os.path.join(root, 'maskPNG_pruned')

    imgDir = sorted(os.listdir(imgPath))
    maskDir = sorted(os.listdir(maskPath))
    print('Image dir has {} files, mask dir has {} files'.format(len(imgDir), len(maskDir)))
    # if len(imgDir) is not len(maskDir):
    #
    #     raise ValueError("Total number of images and masks are not equal")

    if mode == "train":
        idxes = np.random.choice(range(MAX_NUM_FILES), num_files, replace=False)
        for i in range(num_files):
            img = imgDir[idxes[i]]
            mask = maskDir[idxes[i]]
            if (not img.startswith('.')) and (not mask.startswith('.')):
                imgList.append(os.path.join(root, 'imagePNG', img))
                maskList.append(os.path.join(root, 'maskPNG', img))

        df = pd.DataFrame({'image': imgList, 'mask': maskList})
        train_num = int(len(df) * tv_ratio[0])  # Number of training data
        dfTrain = df[:train_num]
        dfVal = df[train_num:-1]
        return dfTrain, dfVal

    elif mode == "test":
        num_test_files = len(imgDir) - MAX_NUM_FILES
        testImgList = []
        testMaskList = []
        # Define dictionary for test files (out of 6820 files)
        test_idxes = np.random.choice(range(MAX_NUM_FILES, len(imgDir)), num_test_files, replace=False)  # arbitrarily test using 3000 files
        for i in range(1000):
            img = imgDir[test_idxes[i]]
            mask = maskDir[test_idxes[i]]
            if (not img.startswith('.')) and (not mask.startswith('.')):
                testImgList.append(os.path.join(root, 'imagePNG', img))
                testMaskList.append(os.path.join(root, 'maskPNG', img))
        dfTest = pd.DataFrame({'image': testImgList, 'mask': testMaskList})
        return dfTest

    else:
        raise ValueError("Mode must be train or test")


def load_checkpoint(model, optimizer, scheduler, filename=None):
    if os.path.isfile(filename):
        print('Loading checkpoint {}'.format(filename))
        print('*** This uses previous hyperparamters except for number of epochs ***')
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch']
        max_score = checkpoint['iou_score']
        model.load_state_dict(checkpoint['state_dict'])
        model.to('cuda')
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])

    else:
        raise ValueError("{} is not a valid checkpoint".format(filename))

    return model, optimizer, scheduler, max_score, start_epoch

# helper function for data visualization
def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image, 'gray')
    plt.show()

# helper function for data visualization
def visualize_2(fn=None, cur_class=None, image=None, gt=None, pr=None, iou=None):
    """PLot images in one row."""
    plt.figure(figsize=(8, 8))
    plt.imshow(image, 'gray', alpha=1)
    plt.imshow(gt, 'Reds', alpha=0.8)
    plt.imshow(pr, 'Blues', alpha=0.5)
    plt.title('%s\nGround truth (red) & prediction (blue) for %s, iou=%.3f' % (fn, cur_class, iou))
    plt.show()


def get_training_augmentation():
    SIZE = 320  # default here is 320, 640 is too large
    train_transform = [

        albu.HorizontalFlip(p=0.5),  # p = probability of applying this transform

        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),

        # Can we play with different sizes?
        albu.PadIfNeeded(min_height=SIZE, min_width=SIZE, always_apply=True, border_mode=0),
        albu.RandomCrop(height=SIZE, width=SIZE, always_apply=True),

        albu.IAAAdditiveGaussianNoise(p=0.2),
        albu.IAAPerspective(p=0.5),

        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.RandomBrightnessContrast(p=1),
                albu.RandomGamma(p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.IAASharpen(p=1),
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.RandomBrightnessContrast(p=1),
                albu.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
    ]
    return albu.Compose(train_transform)


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.PadIfNeeded(384, 480)
    ]
    return albu.Compose(test_transform)

def get_US_training_augmentation(cropped=None):
    if cropped:
        SIZE_H = 320
        SIZE_W = 288
        print('Images are cropped')
    else:
        SIZE_H = 320  # default here is 320, 640 is too large
        SIZE_W = 320

    train_transform = [

        albu.HorizontalFlip(p=0.5),  # p = probability of applying this transform
        albu.ShiftScaleRotate(scale_limit=0.1, rotate_limit=0, shift_limit=0.1, p=0.5, border_mode=0),

        # Can we play with different sizes?
        albu.PadIfNeeded(min_height=SIZE_H, min_width=SIZE_W, always_apply=True, border_mode=0),
        # albu.RandomCrop(height=SIZE_H, width=SIZE_W, always_apply=True), # introduces a lot of black; maybe we don't want this?

        # albu.IAAAdditiveGaussianNoise(p=0.2),
        # albu.IAAPerspective(p=0.5),  # highly unnatural for US to have perspective warping

        # albu.OneOf(
        #     [
        #         albu.CLAHE(p=1),
        #         albu.RandomBrightnessContrast(p=1),
        #         albu.RandomGamma(p=1),
        #     ],
        #     p=0.9,
        # ),
        #
        # albu.OneOf(
        #     [
        #         albu.IAASharpen(p=1),
        #         albu.Blur(blur_limit=3, p=1),
        #         albu.MotionBlur(blur_limit=3, p=1),
        #     ],
        #     p=0.9,
        # ),
        #
        # albu.OneOf(
        #     [
        #         albu.RandomBrightnessContrast(p=1),
        #         albu.HueSaturationValue(p=1),
        #     ],
        #     p=0.9,
        # ),
    ]
    return albu.Compose(train_transform)

def get_US_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.PadIfNeeded(min_height=320, min_width=288)
    ]
    return albu.Compose(test_transform)

def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):  # called by the constructor of Dataset
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)