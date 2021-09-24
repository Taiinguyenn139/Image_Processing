import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# maxtrix shape i x j x k
# depth = i
# row = j
# cols = k

def bit_slicing(img, max_depth):
    result = np.zeros(shape=(max_depth, img.shape[0], img.shape[1]))
    for i in range(result.shape[1]):
        for j in range(result.shape[2]):
            str_repre = np.binary_repr(img[i][j], width=max_depth)
            for z in range(max_depth-1, -1, -1):
                result[z][i][j] = float(str_repre[z])*pow(2, z)

    return result

def show_image(option = 0):
    img = cv.imread('dolar.tif', 0)
    fig = plt.figure(figsize=(16, 9))
    ax1, ax2 = fig.subplots(1, 2)

    img = cv.imread('dolar.tif', 0)
    ax1.imshow(img, cmap='gray')
    ax1.set_title("Original")

    tmp = bit_slicing(img, 8)
    ax2.imshow(tmp[8 - option], cmap='gray')
    ax2.set_title("Show".format(option))
    plt.show()


# Option 8 ==> bit 8
# Option 7 ==> bit 7
# Option 6 ==> bit 6
# Option 5 ==> bit 5
# Option 4 ==> bit 4
# Option 3 ==> bit 3
# Option 2 ==> bit 2
# Option 1 ==> bit 1


if __name__ == '__main__':
    show_image(option=8)