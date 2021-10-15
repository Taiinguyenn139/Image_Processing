import numpy as np
import cv2
import matplotlib.pyplot as plt


def Local_Adapt(img, size, var):
    m, n = img.shape
    result = np.zeros([m, n])
    h = (size - 1) // 2
    padded = np.pad(img, (h, h), mode='reflect')
    for i in range(m):
        for j in range(n):
            k = padded[i:i + size, j:j + size]
            local_var = np.var(k)
            local_mean = np.mean(k)
            if local_mean > var:
                result[i, j] = local_mean
            else:
                result[i, j] = padded[i, j] - int((var / local_var) * (padded[i, j] - local_mean))
    return result


if __name__ == "__main__":
    img_nhieu = cv2.imread('Local_Adapt.tif', 0)
    var = 0.15
    size = 7

    result = Local_Adapt(img_nhieu, size, var)

    fig = plt.figure(figsize=(16, 9))
    ax1, ax2 = fig.subplots(1, 2)
    ax1.imshow(img_nhieu, cmap='gray')
    ax1.set_title("Image with noise")
    ax1.axis("off")

    ax2.imshow(result, cmap='gray')
    ax2.set_title("After reduce noise")
    ax2.axis("off")
    plt.show()
