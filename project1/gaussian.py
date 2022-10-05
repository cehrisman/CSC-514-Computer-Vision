from main import *
import numpy as np
import conv
import math


def make_gaussian_filter(image, sigma):
    size = 8 * sigma + 1
    if not size % 2:
        size = size + 1

    center = size / 2
    kernel = np.zeros((size, size))

    for r in range(size):
        for c in range(size):
            diff = (r - center) ** 2 + (c - center) ** 2
            kernel[r, c] = np.exp(-diff / (2 * sigma ** 2))
    kernel = kernel / np.sum(kernel)

    return kernel


def gaussian_blur(image, sigma, FFT=False):
    kernel = make_gaussian_filter(image, sigma)

    if FFT:
        h, w = image.shape[0], image.shape[1]
        k_h, k_w = kernel.shape[0], kernel.shape[1]

        pad_k = np.zeros(image.shape[:2])
        r = int((h - k_h) / 2)
        c = int((w - k_w) / 2)
        pad_k[r: r + k_h, c: c + k_w] = kernel

        postImg = np.zeros(image.shape)

        for pixels in range(image.shape[2]):
            Fourier_image = np.fft.fft2(image[:, :, pixels])
            Fourier_kernel = np.fft.fft2(pad_k)
            postImg[:, :, pixels] = np.fft.fftshift(np.fft.ifft2(Fourier_image * Fourier_kernel)) / 255
        return postImg
    else:
        postImg = conv.my_imfilter(image, kernel)
        return postImg
