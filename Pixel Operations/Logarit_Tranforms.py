import cv2 as cv
import matplotlib.pyplot as plt


def logarit_transform(img, c):
    return float(c)*cv.log(1.0+img)

def show_image():
    fig = plt.figure(figsize=(16, 9))
    ax1, ax2 = fig.subplots(1, 2)

    img = cv.imread('log.tif', 0)
    ax1.imshow(img, cmap='gray')
    ax1.set_title("Original")

    reverse_img = logarit_transform(img, 2)
    ax2.imshow(reverse_img, cmap='gray')
    ax2.set_title("Log_Trans")
    plt.show()

if __name__ == '__main__':
    show_image()