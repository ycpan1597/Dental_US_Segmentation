import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
from itertools import product
from PP_segmentation_utils import threshold_images

parameters = {"lr": [3e-3, 3e-4],
              "batch_size": [4, 8, 16],
              "num_files": [1000, 2000, 4000]}
param_values = [v for v in parameters.values()]
for lr, batch_size, num_files in product(*param_values):
    print(lr, batch_size, num_files)

root = 'D:\\MRI Segmentation\\data'
# maskPath = os.path.join(root, 'maskPNG')
# maskDir = os.listdir(maskPath)
#
start = time.time()
df = threshold_images(root, pos_thresh=0.005)
elapsed = time.time() - start
print('elapsed time = %.2f' % elapsed)

# img_fps = df['image']
# mask_fps = df['mask']
# idx = np.random.randint(0, len(df)-1, 40)
# plt.figure(figsize=(16, 8))
# for i in range(40):
#     img = cv2.imread(img_fps[idx[i]], 0)
#     mask = cv2.imread(mask_fps[idx[i]], 0)
#     plt.subplot(8, 5, i+1)
#     plt.imshow(img, 'gray', alpha=0.8)
#     plt.imshow(mask, 'Reds', alpha=0.5)