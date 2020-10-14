import glob
import numpy as np
import matplotlib.pyplot as plt


img_path = glob.glob(r'images/images/*')
img = []
for i in img_path:
    img.append(plt.imread(i, format='png'))

img = np.array(img)

for i in range(img.shape[0]):
    img[i].flatten()


img = img/255
mean = []
for i in range(img.shape[0]):
    mean.append(img[i].mean())
print(mean)
print(img.shape)
