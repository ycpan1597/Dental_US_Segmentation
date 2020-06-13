import numpy as np
import torch
import tkinter as tk
from tkinter import filedialog
import torch.optim as optim
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
import PP_segmentation_utils as pp
from PP_Datasets import USDataset
import matplotlib.pyplot as plt

use_cuda = True

# Model Definition
ENCODER = 'vgg16'
ENCODER_WEIGHTS = 'imagenet'
# CLASSES = ['ring-down', 'gel-cover', 'free gel', 'soft tissue',
#                'muscle', 'bone', 'root', 'crown', 'suture', 'lip']
CLASSES = ['ring-down', 'gel-cover', 'free gel', 'soft tissue', 'muscle']

ACTIVATION = 'softmax2d'  # could be None for logits or 'softmax2d' for multicalss segmentation
DEVICE = "cuda" if use_cuda else "cpu"

# create segmentation model with pretrained encoder
model = smp.FPN(
    encoder_name=ENCODER,
    encoder_weights=None,
    classes=len(CLASSES),  # only 1 class in this case
    activation=ACTIVATION,
)

loss = smp.utils.losses.DiceLoss(beta=1)
metrics = [
    smp.utils.metrics.IoU(threshold=0.5),
]

preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, pretrained=ENCODER_WEIGHTS)

cropped = True
data_dir = r"G:\My Drive\Umich Research\Dental Segmentation\Data"  # this was we always have the most updated data!
tv_ratio = [0.8, 0.2]
num_files = -1
dfTrain, dfVal = pp.prepare_US_csv(data_dir, tv_ratio, num_files=num_files, mode="train", cropped=cropped)

raw_train_dataset = USDataset(dfTrain, classes=CLASSES)
train_dataset = USDataset(dfTrain, classes=CLASSES,
                          augmentation=pp.get_US_training_augmentation(cropped=cropped),
                          preprocessing=pp.get_preprocessing(preprocessing_fn)
                          )
train_img, train_mask = train_dataset[0]
valid_dataset = USDataset(dfVal, classes=CLASSES,
                          augmentation=pp.get_US_validation_augmentation(),
                          preprocessing=pp.get_preprocessing(preprocessing_fn))

for n in np.random.choice(range(10), 2, replace=False):
    # Strange! Images are compressed down to 320 x 320 for some reason
    img, lab = train_dataset[n]

    fig, (ax1, ax2) = plt.subplots(2, 3)
    # ax1 = plt.subplot(2, 3, 1)
    ax1[0].imshow(img[0, :, :], 'gray')
    ax1[0].set_title('image')

    # ax2 = plt.subplot(2, 3, 2)
    ax1[1].imshow(lab[0, :, :], 'gray')
    ax1[1].set_title('ring-down')

    # ax3 = plt.subplot(2, 3, 3)
    ax1[2].imshow(lab[1, :, :], 'gray')
    ax1[2].set_title('gel-cover')
    #
    # ax4 = plt.subplot(2, 3, 4)
    ax2[0].imshow(lab[2, :, :], 'gray')
    ax2[0].set_title('free gel')
#     #
    # ax5 = plt.subplot(2, 3, 5)
    ax2[1].imshow(lab[3, :, :], 'gray')
    ax2[1].set_title('soft tissue')
#     #
    # ax6 = plt.subplot(2, 3, 6)
    ax2[2].imshow(lab[4, :, :], 'gray')
    ax2[2].set_title('muscle')


model_dir = r"C:\Users\prestonpan\PycharmProjects\Segmentation_example\runs"
#
tk_root = tk.Tk()
model_path = filedialog.askopenfilename(parent=tk_root, initialdir=model_dir,
                                                title='Please select a model (.pth)')
tk_root.destroy()
best_model = torch.load(model_path)
best_model = best_model['model']
dfTest = pp.prepare_US_csv(data_dir, tv_ratio, num_files=-1, mode="test")
test_dataset = USDataset(dfTest, classes=CLASSES,
                         augmentation=pp.get_US_validation_augmentation(),
                         preprocessing=pp.get_preprocessing(preprocessing_fn))
test_dataset_vis = USDataset(dfTest, classes=CLASSES,
                             augmentation=pp.get_US_validation_augmentation())  # no preprocessing!

n = 2
image_vis = test_dataset_vis[n][0].astype('int16')  # arranged in H * W * 3 (3 = RGB channels)
image, gt_masks = test_dataset[n]
# image has 3 channels (RGB) -> shape(image) = 3 * H * W, different from image_vis because of preprocessing
# gt_masks has C classes -> shape(gt_mask) = C * H * W (C = Classes)
#
class_idx = 0
x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)  # shape = 1 * 3 * H * W (3 = RGB channels)
pr_masks = best_model.predict(x_tensor)  # shape = 1 * C* H * W (C = number of classes)
pr_masks = pr_masks.squeeze(0).cpu().numpy().round()

y_gt = torch.from_numpy(gt_masks[class_idx])
y_pr = torch.from_numpy(pr_masks[class_idx])

# #
# plt.close('all')
#
# # Need to plot the original image to make sure the mask and image are aligned!
pp.visualize_2(cur_class='ring-down', image=image_vis, gt=y_gt, pr=y_pr, iou=metrics[0].forward(y_gt, y_pr))

# plt.figure(1)
# plt.subplot(1, 2, 1)
# plt.imshow(y_gt, 'gray')
# plt.subplot(1, 2, 2)
# plt.imshow(y_pr, 'jet')
# plt.figure(2)
# plt.imshow(y_gt, 'gray')
# plt.imshow(y_pr, 'jet', alpha=0.5)
#
#
# DL = smp.utils.losses.DiceLoss(beta=1)
# GDL = smp.utils.losses.GeneralizedDiceLoss()
#
# print('DL = %.4f' % DL.forward(y_pr, y_gt))
# print('GDL = %.4f' % GDL.forward(y_pr, y_gt))
