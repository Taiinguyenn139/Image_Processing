import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('test.tif', 0)
m, n = img.shape[:2]

#Gaussian noise
mean = 10
var = 20
gaussian = np.random.normal(loc=mean, scale=var, size=(m, n))
Gaussian_noise_img = gaussian + img

#Rayleigh noise
var = 10
rayliegh = np.random.rayleigh(scale=var, size=(m, n))
Rayleigh_noise_img = img + rayliegh

#Gamma noise
K = 2.0
var = 20
gamma = np.random.gamma(shape=K, scale=var, size=(m, n))
Gamma_noise_img = img + gamma

#Exponential noise
var = 20
exponential = np.random.exponential(scale=var, size=(m, n))
Exponential_noise_img = img + exponential

#Uniform
a, b = 10, 100
uniform = np.random.uniform(low=a, high=b, size=(m, n))
Uniform_noise_img = img + uniform

#Pepper - Salt noise
rate = 0.05
num_black = int(m*n*rate)
num_white = int(m*n*rate)

m_black = np.random.randint(0, m, num_black)
n_black = np.random.randint(0, n, num_black)
m_white = np.random.randint(0, m, num_white)
n_white = np.random.randint(0, n, num_white)

PS_noise = np.copy(img)
PS_noise[m_black, n_black] = 0
PS_noise[m_white, n_white] = 255

#Fig1
fig1 = plt.figure(figsize=(16, 9))
(ax1, ax2), (ax3, ax4) = fig1.subplots(2, 2)

ax1.imshow(img, 'gray')
ax1.set_title('Original')
ax1.axis('off')

ax2.hist(img.flatten(), bins=256)
ax2.set_title('Histogram')

ax3.imshow(Gaussian_noise_img, 'gray')
ax3.set_title('Gaussian noise')
ax3.axis('off')

ax4.hist(Gaussian_noise_img.flatten(), bins=256)
ax4.set_title('Histogram')
plt.show()

#Fig2
fig2 = plt.figure(figsize=(16, 9))
(ax1, ax2), (ax3, ax4) = fig2.subplots(2, 2)

ax1.imshow(img, 'gray')
ax1.set_title('Original')
ax1.axis('off')

ax2.hist(img.flatten(), bins=256)
ax2.set_title('Histogram')

ax3.imshow(Rayleigh_noise_img, 'gray')
ax3.set_title('Rayleigh noise')
ax3.axis('off')

ax4.hist(Rayleigh_noise_img.flatten(), bins=256)
ax4.set_title('Histogram')
plt.show()

#Fig3
fig3 = plt.figure(figsize=(16, 9))
(ax1, ax2), (ax3, ax4) = fig3.subplots(2, 2)

ax1.imshow(img, 'gray')
ax1.set_title('Original')
ax1.axis('off')

ax2.hist(img.flatten(), bins=256)
ax2.set_title('Histogram')

ax3.imshow(Gamma_noise_img, 'gray')
ax3.set_title('Gamma noise')
ax3.axis('off')

ax4.hist(Gamma_noise_img.flatten(), bins=256)
ax4.set_title('Histogram')
plt.show()

#Fig4
fig4 = plt.figure(figsize=(16, 9))
(ax1, ax2), (ax3, ax4) = fig4.subplots(2, 2)

ax1.imshow(img, 'gray')
ax1.set_title('Original')
ax1.axis('off')

ax2.hist(img.flatten(), bins=256)
ax2.set_title('Histogram')

ax3.imshow(Exponential_noise_img, 'gray')
ax3.set_title('Exponential noise')
ax3.axis('off')

ax4.hist(Exponential_noise_img.flatten(), bins=256)
ax4.set_title('Histogram')
plt.show()

#Fig5
fig5 = plt.figure(figsize=(16, 9))
(ax1, ax2), (ax3, ax4) = fig5.subplots(2, 2)

ax1.imshow(img, 'gray')
ax1.set_title('Original')
ax1.axis('off')

ax2.hist(img.flatten(), bins=256)
ax2.set_title('Histogram')

ax3.imshow(Uniform_noise_img, 'gray')
ax3.set_title('Uniform noise')
ax3.axis('off')

ax4.hist(Uniform_noise_img.flatten(), bins=256)
ax4.set_title('Histogram')
plt.show()

#Fig6
fig6 = plt.figure(figsize=(16, 9))
(ax1, ax2), (ax3, ax4) = fig6.subplots(2, 2)

ax1.imshow(img, 'gray')
ax1.set_title('Original')
ax1.axis('off')

ax2.hist(img.flatten(), bins=256)
ax2.set_title('Histogram')

ax3.imshow(PS_noise, 'gray')
ax3.set_title('Pepper-Salt noise')
ax3.axis('off')

ax4.hist(PS_noise.flatten(), bins=256)
ax4.set_title('Histogram')
plt.show()

