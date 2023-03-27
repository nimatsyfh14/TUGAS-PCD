import cv2
import numpy as np

def YCBkeRGB(Y, Cb, Cr):
    # Normalisasi Y, Cb, Cr ke [0, 1]
    Y = np.double(Y)
    Cr = np.double(Cr)
    Cb = np.double(Cb)
    if np.max(np.max(Y)) > 1.0 or np.max(np.max(Cb)) > 1.0 or np.max(np.max(Cr)) > 1.0:
        Y = np.double(Y) / 255.0
        Cr = np.double(Cr) / 255.0
        Cb = np.double(Cb) / 255.0

    tinggi, lebar = np.shape(Y)
    R = np.zeros((tinggi, lebar))
    G = np.zeros((tinggi, lebar))
    B = np.zeros((tinggi, lebar))
    for m in range(tinggi):
        for n in range(lebar):
            R[m,n] = Y[m,n] + 1.402 * Cr[m,n]
            G[m,n] = Y[m,n] - 0.34414 * Cb[m,n] - 0.71414 * Cr[m,n]
            B[m,n] = Y[m,n] + 1.7720 * Cb[m,n]

    # Konversi ke jangkauan [0, 255]
    R = np.uint8(R * 255)
    G = np.uint8(G * 255)
    B = np.uint8(B * 255)

    return R, G, B

# Load gambar dengan OpenCV
img = cv2.imread('kyo.jpg')

# Ubah warna dari BGR ke YCbCr
imgYCbCr = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

# Ambil komponen Y, Cb, Cr
Y = imgYCbCr[:,:,0]
Cb = imgYCbCr[:,:,1]
Cr = imgYCbCr[:,:,2]

# Konversi YCbCr ke RGB menggunakan fungsi YCBkeRGB
R, G, B = YCBkeRGB(Y, Cb, Cr)

# Gabungkan kembali R, G, B menjadi gambar RGB
imgRGB = cv2.merge((R, G, B))

# Tampilkan gambar RGB
cv2.imshow('Gambar RGB', imgRGB)
cv2.waitKey(0)
cv2.destroyAllWindows()
