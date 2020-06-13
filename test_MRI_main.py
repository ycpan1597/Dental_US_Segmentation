import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
import PP_segmentation_utils as pp
from PP_Datasets import MRI, MRIDataset
import matplotlib.pyplot as plt

use_cuda = True

# Model Definition
ENCODER = 'se_resnext50_32x4d'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = ['tumor']  # only 1 class in this example
ACTIVATION = 'sigmoid'  # could be None for logits or 'softmax2d' for multicalss segmentation
DEVICE = "cuda" if use_cuda else "cpu"

# create segmentation model with pretrained encoder
model = smp.Unet(
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

root = 'D:\\MRI Segmentation\\data'
tv_ratio = [0.8, 0.2]
num_files = 500
dfTrain, dfVal, dfTest = pp.prepare_csv(root, tv_ratio, num_files=num_files)

train_dataset = MRIDataset(dfTrain, classes=CLASSES,
                           augmentation=pp.get_training_augmentation(),
                           preprocessing=pp.get_preprocessing(preprocessing_fn))
valid_dataset = MRIDataset(dfVal, classes=CLASSES,
                           augmentation=pp.get_training_augmentation(),
                           preprocessing=pp.get_preprocessing(preprocessing_fn))
# img, lab = train_dataset[0]
# bg, tu = lab[0, :, :], lab[1, :, :]


# # for i in range(5):
# #     idx = np.random.randint(1, 20)
# #     pp.visualize(img=np.transpose(train_dataset[idx][0], (1, 2, 0)).squeeze(),
# #                  gt=np.transpose(train_dataset[idx][1], (1, 2, 0)).squeeze())
#


# train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=12)
# valid_loader = DataLoader(valid_dataset, batch_size=4, shuffle=False, num_workers=4)
#
optimizer = optim.SGD(model.parameters(), lr=3e-5, momentum=0.90, weight_decay=1e-6, nesterov=True)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8,
                                      gamma=0.7)  # every step_size number of epoch, the lr is multiplied by args.gamma - reduces learning rate due to subgradient method

#
# train_epoch = smp.utils.train.TrainEpoch(
#     model,
#     loss=loss,
#     metrics=metrics,
#     optimizer=optimizer,
#     device=DEVICE,
#     verbose=True,
# )
#
# valid_epoch = smp.utils.train.ValidEpoch(
#     model,
#     loss=loss,
#     metrics=metrics,
#     device=DEVICE,
#     verbose=True,
# )
#
# Allows you to play with one of the trained models (they are very bad at the moment)
best_model = torch.load(r"C:\Users\prestonpan\PycharmProjects\Segmentation_example\best MRI models\05_18_20_2.pth")

test_dataset = MRIDataset(dfTest, classes=CLASSES,
                          preprocessing=pp.get_preprocessing(preprocessing_fn))
test_dataset_vis = MRIDataset(dfTest, classes=CLASSES)  # no preprocessing!

n = np.random.randint(0, 3000, 1)
n = n[0]

image_vis = test_dataset_vis[n][0].astype('int16')  # arranged in H * W * 3 (3 = RGB channels)
image, gt_masks = test_dataset[n]
# image has 3 channels (RGB) -> shape(image) = 3 * H * W, different from image_vis because of preprocessing
# gt_masks has C classes -> shape(gt_mask) = C * H * W (C = Classes)

x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)  # shape = 1 * 3 * H * W (3 = RGB channels)
pr_masks = best_model.predict(x_tensor)  # shape = 1 * C* H * W (C = number of classes)
pr_masks = pr_masks.squeeze(0).cpu().numpy().round()

y_gt = torch.from_numpy(gt_masks[0])
y_pr = torch.from_numpy(pr_masks[0])

plt.close('all')

plt.figure(1)
plt.subplot(1, 2, 1)
plt.imshow(y_gt, 'gray')
plt.subplot(1, 2, 2)
plt.imshow(y_pr, 'jet')
plt.figure(2)
plt.imshow(y_gt, 'gray')
plt.imshow(y_pr, 'jet', alpha=0.5)


DL = smp.utils.losses.DiceLoss(beta=1)
GDL = smp.utils.losses.GeneralizedDiceLoss()

print('DL = %.4f' % DL.forward(y_pr, y_gt))
print('GDL = %.4f' % GDL.forward(y_pr, y_gt))
