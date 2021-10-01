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

Laplacian = np.array(([0, 1, 0],
                      [1, -4, 1],
                      [0, 1, 0]), dtype="float")

Laplacian_1 = np.array(([1, 1, 1],
                        [1, -8, 1],
                        [1, 1, 1]), dtype="float")

Laplacian_2 = np.array(([0, -1, 0],
                        [-1, 4, -1],
                        [0, -1, 0]), dtype="float")

Laplacian_3 = np.array(([-1, -1, -1],
                        [-1, 8, -1],
                        [-1, -1, -1]), dtype="float")

Laplacian_5 = np.array(([-1, -1, -1],
                        [-1, 5, -1],
                        [-1, -1, -1]), dtype="float")

fig = plt.figure(figsize=(16, 9))
(ax1, ax2, ax3), (ax4, ax5, ax6) = fig.subplots(2, 3)

image = cv.imread('test2.tif', 0)
ax1.imshow(image, cmap='gray')
ax1.set_title("Original")

Laplacian = convo_filter(image, Laplacian)
ax2.imshow(Laplacian, cmap='gray')
ax2.set_title("Original Laplacian")

minus_Laplacian = image-convo_filter(image, Laplacian)
ax3.imshow(minus_Laplacian, cmap='gray')
ax3.set_title("image - laplacian")

ax4.axis('off')

img_Laplacian_1 = convo_filter(image, Laplacian_1) #Gọi hàm tích chập
ax5.imshow(img_Laplacian_1, cmap='gray')
ax5.set_title("Laplacian ver 1")

img_Laplacian_2 = convo_filter(image, Laplacian_2) #Gọi hàm tích chập
ax6.imshow(img_Laplacian_2, cmap='gray')
ax6.set_title("Laplacian ver 2")

plt.show()
