import cv2 as cv
import matplotlib.pyplot as plt

def reverse(img):
    return 255 - img

def show_image():
    fig = plt.figure(figsize=(16, 9))
    ax1, ax2 = fig.subplots(1, 2)

    img = cv.imread('daoanh.tif', 0)
    ax1.imshow(img, cmap='gray')
    ax1.set_title("Original")

    reverse_img = reverse(img)
    ax2.imshow(reverse_img, cmap='gray')
    ax2.set_title("Reverse")
    plt.show()

if __name__ == '__main__':
    show_image()