import cv2 as cv
import matplotlib.pyplot as plt

def threshold(img, th):
    return img > th

def show_image():
    fig = plt.figure(figsize=(16, 9))
    ax1, ax2 = fig.subplots(1, 2)

    img = cv.imread('keodan_dau.tif', 0)
    ax1.imshow(img, cmap='gray')
    ax1.set_title("Original")

    reverse_img = threshold(img, 117)
    ax2.imshow(reverse_img, cmap='gray')
    ax2.set_title("Threshold")
    plt.show()

if __name__ == '__main__':
    show_image()
