import numpy as np
import matplotlib.pyplot as plt
import cv2
from Fourier_Trans import DFT1D, IDFT1D

def GaussianLP(D0,U,V):
    # H cho filter
    H = np.zeros((U, V))
    D = np.zeros((U, V))
    U0 = int(U / 2)
    V0 = int(V / 2)
    # Tính khoảng cách
    for u in range(U):
        for v in range(V):
            u2 = np.power(u, 2)
            v2 = np.power(v, 2)
            D[u, v] = np.sqrt(u2 + v2)
    # Tính bộ lọc
    for u in range(U):
        for v in range(V):
            H[u, v] = np.exp((-D[np.abs(u - U0), np.abs(v - V0)]**2)/(2*(D0**2)))

    return H

if __name__ == "__main__":
    image = cv2.imread("test.tif", 0)
    image = cv2.resize(src=image, dsize=(100, 100))

    f = np.asarray(image)
    M, N = np.shape(f)

    P, Q = 2*M , 2*N
    shape = np.shape(f)

    f_xy_p = np.zeros((P, Q))
    f_xy_p[:shape[0], :shape[1]] = f


    F_xy_p = np.zeros((P, Q))
    for x in range(P):
        for y in range(Q):
            F_xy_p[x, y] = f_xy_p[x, y] * np.power(-1, x + y)


    dft_cot = dft_hang = np.zeros((P, Q))

    for i in range(P):
        dft_cot[i] = DFT1D(F_xy_p[i])

    for j in range(Q):
        dft_hang[:, j] = DFT1D(dft_cot[:, j])


    H_uv = GaussianLP(30,P,Q)


    G_uv = np.multiply(dft_hang, H_uv)


    idft_cot = idft_hang = np.zeros((P, Q))

    for i in range(P):
        idft_cot[i] = IDFT1D(G_uv[i])

    for j in range(Q):
        idft_hang[:, j] = IDFT1D(idft_cot[:, j])


    g_array = np.asarray(idft_hang.real)
    P, Q = np.shape(g_array)
    g_xy_p = np.zeros((P, Q))
    for x in range(P):
        for y in range(Q):
            g_xy_p[x, y] = g_array[x, y] * np.power(-1, x + y)

    g_xy = g_xy_p[:shape[0], :shape[1]]

    fig = plt.figure(figsize=(16, 9))

    ax1, ax2 = fig.subplots(1, 2)


    ax1.imshow(image, cmap='gray')
    ax1.set_title('Original')
    ax1.axis('off')

    ax2.imshow(g_xy, cmap='gray')
    ax2.set_title('Result')
    ax2.axis('off')

    plt.show()
