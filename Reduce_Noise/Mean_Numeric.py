import cv2
import matplotlib.pyplot as plt
import numpy as np


def mean_num_filter(img, size = 3):
    m, n = img.shape[:2]
    result = np.empty([m, n])
    h = (size-1)//2
    padded_img = np.pad(img, (h, h), mode='reflect')

    for i in range(m):
        for j in range(n):
            k = padded_img[i:i+size, j:j+size]
            result[i, j] = np.mean(k)

    return result

if __name__ == '__main__':
    img_noise = cv2.imread('Mean.tif', 0)
    result = mean_num_filter(img_noise, size=5)
    fig = plt.figure(figsize=(16, 9))
    (ax1, ax2) = fig.subplots(1, 2)
    ax1.imshow(img_noise, cmap='gray')
    ax1.set_title("Image with noise")
    ax1.axis("off")

    ax2.imshow(result, cmap='gray')
    ax2.set_title("After reduce noise")
    ax2.axis("off")

    plt.show()