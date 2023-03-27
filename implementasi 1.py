import numpy as np
import cv2


def RGBkeYIQ(R, G, B):
    # Normalisasi RGB ke [0, 1]
    R = np.double(R)
    G = np.double(G)
    B = np.double(B)
    if np.max(np.max(R)) > 1.0 or np.max(np.max(G)) > 1.0 or np.max(np.max(B)) > 1.0:
        R = np.double(R) / 255.0
        G = np.double(G) / 255.0
        B = np.double(B) / 255.0

    tinggi, lebar = np.shape(R)
    Y = np.zeros((tinggi, lebar))
    I = np.zeros((tinggi, lebar))
    Q = np.zeros((tinggi, lebar))
    for m in range(tinggi):
        for n in range(lebar):
            Y[m, n] = 0.299 * R[m, n] + 0.587 * G[m, n] + 0.114 * B[m, n]
            I[m, n] = 0.596 * R[m, n] - 0.274 * G[m, n] - 0.322 * B[m, n]
            Q[m, n] = 0.211 * R[m, n] - 0.523 * G[m, n] + 0.312 * B[m, n]

    # Konversi ke jangkauan [0, 255]
    Y = np.uint8(Y * 255)
    I = np.uint8(I * 255)
    Q = np.uint8(Q * 255)

    return Y, I, Q


# Contoh penggunaan
img = cv2.imread('kyo.jpg')
R = img[:, :, 2]
G = img[:, :, 1]
B = img[:, :, 0]
Y, I, Q = RGBkeYIQ(R, G, B)
cv2.imshow('Y', Y)
cv2.imshow('I', I)
cv2.imshow('Q', Q)
cv2.waitKey(0)
cv2.destroyAllWindows()
