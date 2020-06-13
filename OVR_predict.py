# predict images based on One-vs-Rest (OVR) architecture
import os
import torch
import numpy as np
import PP_segmentation_utils as pp
from PP_Datasets import USDataset
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp

ENCODER = 'vgg16'  # maybe we stick with this for now?
preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, pretrained='imagenet')
data_dir = r"G:\My Drive\Umich Research\Dental Segmentation\Data"
CLASSES = ['background', 'ring-down', 'gel-cover', 'free gel', 'soft tissue',
           'muscle', 'bone', 'root', 'crown', 'suture', 'lip']

dfTest = pp.prepare_US_csv(data_dir, [0.8, 0.2], cropped=True, num_files=-1, mode="test")  # number of testing images is fixed

model_directories = [
    r"C:\Users\prestonpan\PycharmProjects\Segmentation_example\US_runs\classes=['ring-down']\model.pth",
    r"C:\Users\prestonpan\PycharmProjects\Segmentation_example\US_runs\classes=['gel-cover']\model.pth",
    r"C:\Users\prestonpan\PycharmProjects\Segmentation_example\US_runs\classes=['free gel']\model.pth",
    r"C:\Users\prestonpan\PycharmProjects\Segmentation_example\US_runs\classes=['soft tissue']\model.pth",
    r"MUSCLE",
    r"C:\Users\prestonpan\PycharmProjects\Segmentation_example\US_runs\classes=['bone']\model.pth",
    r"ROOT",
    r"C:\Users\prestonpan\PycharmProjects\Segmentation_example\US_runs\classes=['crown']\model.pth",
    r"SUTURE",
    r"LIP"
]

test_dataset = USDataset(dfTest, classes=CLASSES,
                         augmentation=pp.get_US_validation_augmentation(),
                         preprocessing=pp.get_preprocessing(preprocessing_fn))
test_dataset_vis = USDataset(dfTest, classes=CLASSES,
                             augmentation=pp.get_US_validation_augmentation(),
                             native_mask=True)
pr_masks = np.zeros((320, 288, len(test_dataset_vis)))
gt_masks = np.zeros((320, 288, len(test_dataset_vis)))
for i, (image, mask) in enumerate(test_dataset):
    x_tensor = torch.from_numpy(image).to('cuda').unsqueeze(0)
    gt_masks[:, :, i] = test_dataset_vis[i][1]
    for j in range(len(model_directories)):
        if os.path.exists(model_directories[j]):
            model_dict = torch.load(model_directories[j])
            model = model_dict['model']
            pr_mask = model.predict(x_tensor)
            pr_mask = (pr_mask.squeeze(0).cpu().numpy().round())
            pr_masks[:, :, i] = np.maximum(pr_masks[:, :, i], pr_mask[0, :, :] * (j + 1))


plt.subplot(2, 4, 1)
plt.imshow(gt_masks[:, :, 0], vmin=0, vmax=10)
# plt.title('Ground truth')
plt.subplot(2, 4, 2)
plt.imshow(gt_masks[:, :, 1], vmin=0, vmax=10)
plt.subplot(2, 4, 3)
plt.imshow(gt_masks[:, :, 2], vmin=0, vmax=10)
plt.subplot(2, 4, 4)
plt.imshow(gt_masks[:, :, 3], vmin=0, vmax=10)
#
plt.subplot(2, 4, 5)
plt.imshow(pr_masks[:, :, 0], vmin=0, vmax=10)
# plt.title()
plt.subplot(2, 4, 6)
plt.imshow(pr_masks[:, :, 1], vmin=0, vmax=10)
plt.subplot(2, 4, 7)
plt.imshow(pr_masks[:, :, 2], vmin=0, vmax=10)
plt.subplot(2, 4, 8)
plt.imshow(pr_masks[:, :, 3], vmin=0, vmax=10)

plt.suptitle('Prediction of ring-down, gel-cover, free gel, soft tissue, bone, crown\n(top: ground truth, bottom: prediction')

