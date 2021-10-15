import cv2
import numpy as np
import matplotlib.pyplot as plt

def mean_geo(img, size):
    m, n = img.shape[:2]
    result = np.zeros((m, n))
    h = (size-1)//2
    padded = np.pad(img, (h, h), mode='reflect')

    for i in range(m):
        for j in range(n):
            k = padded[i:i+size,j:j+size]
            local_mean = np.mean(k)
            kernel = np.prod(k) ** (1.0 / m * n)
            if kernel > local_mean:
                result[i, j]= int(local_mean)
            else:
                result[i,j] = int(kernel)
    return result

if __name__ == '__main__':
    img_noise = cv2.imread('Mean.tif', 0)
    result = mean_geo(img_noise, size=3)
    fig = plt.figure(figsize=(16, 9))
    (ax1, ax2) = fig.subplots(1, 2)
    ax1.imshow(img_noise, cmap='gray')
    ax1.set_title("Image with noise")
    ax1.axis("off")

    ax2.imshow(result, cmap='gray')
    ax2.set_title("After reduce noise")
    ax2.axis("off")

    plt.show()