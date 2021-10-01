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
    img_new = img_new.astype(dtype=np.uint8)
    return img_new

sobel_1 = np.array(([-1, -2, -1],
                    [0, 0, 0],
                    [1, 2, 1]), dtype='float')

sobel_2 = np.array(([-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]), dtype='float')

fig = plt.figure(figsize=(16, 9))
(ax1, ax2, ax3), (ax4, ax5, ax6) = fig.subplots(2, 3)

image = cv.imread('test1.tif', 0)
ax1.imshow(image, cmap='gray')
ax1.set_title("Original")

image_sobel_1 = convo_filter(image, sobel_1)
ax2.imshow(image_sobel_1, cmap='gray')
ax2.set_title("sobel 1 filter")

image_sobel_2 = convo_filter(image, sobel_2)
ax3.imshow(image_sobel_2, cmap='gray')
ax3.set_title("sobel 2 filter")

ax4.axis('off')

image_sobel_sum = image_sobel_1 + image_sobel_2
ax5.imshow(image_sobel_sum, cmap='gray')
ax5.set_title("sobel sum filter")

image_new = image_sobel_sum + image
ax6.imshow(image_new, cmap='gray')
ax6.set_title("Result")

plt.show()
