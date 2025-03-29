import cv2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the image
image_path = "data/test.jpg"
img = cv2.imread(image_path)

# Inverse binary threshold grayscale version of image
img_thr = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 128, 255, cv2.THRESH_BINARY_INV)[1]
print(img_thr.shape)

sns.heatmap(img_thr)
plt.show()

# Count pixels along the y-axis, find peaks
thr_y = 200
y_sum = np.count_nonzero(img_thr, axis=0)
peaks = np.where(y_sum > thr_y)[0]

# Clean peaks
thr_x = 50
temp = np.diff(peaks).squeeze()
idx = np.where(temp > thr_x)[0]
peaks = np.concatenate(([0], peaks[idx+1]), axis=0) + 1

# Save sub-images
for i in np.arange(peaks.shape[0] - 1):
    cv2.imwrite('sub_image_' + str(i) + '.png', img[:, peaks[i]:peaks[i+1]])