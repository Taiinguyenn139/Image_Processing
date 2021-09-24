import cv2 as cv
import matplotlib.pyplot as plt

#### Step by step ####
# Step 1 : Calculate probability of each value in image
#          p(r_k) = h(r_k)/(M x N)
#   h(r_k) = n_k is number of value r_k
#   M x N is amount of pixel in Image example : image with shape(400, 500)  M x N = 400 x 500
#
# Step 2 : Calculate PDF (Probability Density Function)
#          s_k = (L-1) x sum(p(r_j)) with j from 0 to k
#
# Step 3 : Round s_k


img = cv.imread('keodan_dau.tif', 0)
img_equal = cv.equalizeHist(img)

fig = plt.figure(figsize=(16, 9))
(ax1, ax2), (ax3, ax4) = fig.subplots(2, 2)

ax1.imshow(img, cmap='gray')
ax1.set_title("Original")

ax2.hist(img)
ax2.set_title("Original Histogram")

ax3.imshow(img_equal, cmap='gray')
ax3.set_title("Image Equalized")

ax4.hist(img_equal)
ax4.set_title("Image Equalized Histogram")

plt.show()

