import torch
import numpy as np
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp

img_size = [50, 50]
y_gt = np.zeros(img_size)
y_pr = np.zeros(img_size)

xgrid, ygrid = np.meshgrid(range(0, img_size[0]), range(0, img_size[1]))

perc = 1
mask = lambda center, radius: ((xgrid-center[0])**2 + (ygrid-center[1])**2) <= radius**2
r_func = lambda perc: np.sqrt(img_size[0] * img_size[1] * (perc/100) / np.pi)
r = r_func(perc)


y_gt[mask([25, 25], r)] = 1
y_pr[mask([29, 25], r)] = 1

y_gt = torch.from_numpy(y_gt)
y_pr = torch.from_numpy(y_pr)

plt.imshow(y_gt, 'gray')
plt.imshow(y_pr, 'jet', alpha=0.5)

DL = smp.utils.losses.DiceLoss(beta=1)
GDL = smp.utils.losses.GeneralizedDiceLoss()

print('DL = %.4f' % DL.forward(y_pr, y_gt))
print('GDL = %.4f' % GDL.forward(y_pr, y_gt))

