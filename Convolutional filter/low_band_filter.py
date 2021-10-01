import cv2 as cv
import numpy as np
import  matplotlib.pylab as plt

def convo_filter(img, mask):
    m, n = img.shape
    img_new = np.zeros([m, n])
    for i in range(1, m-1):
        for j in range(1, n-1):
            temp = img[i-1, j-1]*mask[0, 0]\
                 + img[i-1, j]*mask[0, 1]\
                 + img[i-1, j+1]*mask[0, 2]\
                 + img[i, j-1]*mask[1, 0]\
                 + img[i, j]*mask[1, 1]\
                 + img[i, j+1]*mask[1, 2]\
                 + img[i+1, j-1]*mask[2, 0]\
                 + img[i+1, j]*mask[2, 1]\
                 + img[i+1, j+1]*mask[2, 2]
            img_new[i, j] = temp
    img_new = img_new.astype(dtype=int)
    return img_new

#Mean filter
mean_filter = np.array(([1/9, 1/9, 1/9],
                        [1/9, 1/9, 1/9],
                        [1/9, 1/9, 1/9]), dtype=float)

#Mean_filter_coeff
mean_filter_coeff = np.array(([1/16, 2/16, 1/16],
                              [2/16, 4/16, 2/16],
                              [1/16, 2/16, 1/16]), dtype=float)

#Gaussian_filter
gaussian_filter = np.array(([0.0751/4.8976, 0.1238/4.8976, 0.0751/4.8976],
                            [0.1238/4.8976, 0.2042/4.8976, 0.1238/4.8976],
                            [0.0751/4.8976, 0.1238/4.8976, 0.0751/4.8976]), dtype=float)

# Median filter
def median_filter(img):
    m, n = img.shape
    img_new = np.zeros([m, n])
    for i in range(1, m-1):
        for j in range(1, n-1):
            tmp = [img[i-1, j-1],
                   img[i-1, j],
                   img[i-1, j+1],
                   img[i, j-1],
                   img[i, j],
                   img[i, j+1],
                   img[i+1, j],
                   img[i+1, j+1]]
            tmp = sorted(tmp)
            img_new[i, j] = tmp[4]
    return img_new

# Max filter
def max_filter(img):
    m, n = img.shape
    img_new = np.zeros([m, n])
    for i in range(1, m - 1):
        for j in range(1, n - 1):
            tmp = [img[i - 1, j - 1],
                   img[i - 1, j],
                   img[i - 1, j + 1],
                   img[i, j - 1],
                   img[i, j],
                   img[i, j + 1],
                   img[i + 1, j],
                   img[i + 1, j + 1]]
            tmp = max(tmp)
            img_new[i, j] = tmp
    return img_new

fig = plt.figure(figsize=(16, 9))
(ax1, ax2, ax3), (ax4, ax5, ax6) = fig.subplots(2, 3)

image = cv.imread('test2.tif', 0)
ax1.imshow(image, cmap='gray')
ax1.set_title("Original")

image_mean_filter = convo_filter(image, mean_filter)
ax2.imshow(image_mean_filter, cmap='gray')
ax2.set_title("mean filter")

image_mean_coeff_filter = convo_filter(image, mean_filter_coeff)
ax3.imshow(image_mean_coeff_filter, cmap='gray')
ax3.set_title("mean filter with coeff")

image_gaussian_filter = convo_filter(image, gaussian_filter)
ax4.imshow(image_gaussian_filter, cmap='gray')
ax4.set_title("Gaussian filter")

image_median_filter = median_filter(image)
ax5.imshow(image_median_filter, cmap='gray')
ax5.set_title("Median filter")

image_max_filter = max_filter(image)
ax6.imshow(image_max_filter, cmap='gray')
ax6.set_title("Max filter")

plt.show()

