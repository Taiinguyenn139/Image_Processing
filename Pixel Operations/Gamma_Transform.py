import cv2 as cv
import matplotlib.pyplot as plt


def gamma_transform(img, gamma, c):
    return  float(c) * pow(img, gamma)

def show_image():
    fig = plt.figure(figsize=(16, 9))
    ax1, ax2 = fig.subplots(1, 2)

    img = cv.imread('sanbay.tif', 0)
    ax1.imshow(img, cmap='gray')
    ax1.set_title("Original")

    reverse_img = gamma_transform(img, 3.0, 1.0)
    ax2.imshow(reverse_img, cmap='gray')
    ax2.set_title("Gamma_transform")
    plt.show()

if __name__ == '__main__':
    show_image()