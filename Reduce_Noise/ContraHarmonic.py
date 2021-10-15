import cv2
import numpy as np
import matplotlib.pyplot as plt

def contra_harmonic(img, size, Q):
    m, n = img.shape[:2]
    result = np.zeros((m, n))
    h = (size - 1) // 2
    padded = np.pad(img, (h, h), mode='reflect')
    padded_numerator = np.power(padded, Q + 1)
    padded_denominator = np.power(padded, Q)

    for i in range(m):
        for j in range(n):
            k_numerator = padded_numerator[i:i+size,j:j+size]
            k_denominator = padded_denominator[i:i+size, j:j+size]
            numerator = np.sum(k_numerator)
            denominator = np.sum(k_denominator)
            kernel = numerator/denominator
            result[i, j] = int(kernel)
    return result

if __name__ == '__main__':
    img_noise = cv2.imread('Mean.tif', 0)
    result = contra_harmonic(img_noise, size=3, Q=1.5)
    fig = plt.figure(figsize=(16, 9))
    (ax1, ax2) = fig.subplots(1, 2)
    ax1.imshow(img_noise, cmap='gray')
    ax1.set_title("Image with noise")
    ax1.axis("off")

    ax2.imshow(result, cmap='gray')
    ax2.set_title("After reduce noise")
    ax2.axis("off")

    plt.show()