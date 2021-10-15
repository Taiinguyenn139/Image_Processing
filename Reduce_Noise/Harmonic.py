import cv2
import numpy as np
import matplotlib.pyplot as plt

def harmonic(img, size):
    m, n = img.shape
    result = np.zeros([m, n])
    h = (size - 1) // 2
    padded_img = np.pad(img, (h, h), mode='reflect')

    for i in range(m):
        for j in range(n):
            k = padded_img[i:i+size,j:j+size]
            local_mean = np.mean(k)
            kernel = np.sum(m*n/1/k)
            if kernel > local_mean:
                result[i, j] = local_mean
            else:
                result[i, j] = kernel
    return result

if __name__ == '__main__':
    img_noise = cv2.imread('Mean.tif', 0)
    result = harmonic(img_noise, size=9)
    fig = plt.figure(figsize=(16, 9))
    (ax1, ax2) = fig.subplots(1, 2)
    ax1.imshow(img_noise, cmap='gray')
    ax1.set_title("Image with noise")
    ax1.axis("off")

    ax2.imshow(result, cmap='gray')
    ax2.set_title("After reduce noise")
    ax2.axis("off")

    plt.show()