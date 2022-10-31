import cv2
import numpy as np
import math
from matplotlib import pyplot as plt
from skimage.color import rgb2gray
from scipy import signal

gauss_mask = np.array([[1, 4, 7, 4, 1],
                       [4, 16, 26, 16, 4],
                       [7, 26, 41, 26, 7],
                       [4, 16, 26, 16, 4],
                       [1, 4, 7, 4, 1]], np.dtype('float64')) / 273.0

sobel_x = np.array([[3, 0, -3],
                    [10, 0, -10],
                    [3, 0, -3]])

sobel_y = np.array([[3, 10, 3],
                    [0, 0, 0],
                    [-3, -10, -3]])


def harris(img, k, threshold):
    points = []
    height = img.shape[0]
    width = img.shape[1]
    offset = 3
    # Convert image to grayscale
    gray_img = rgb2gray(img)

    # plt.imshow(gray_img, cmap='gray')
    # plt.show()
    gray_img = signal.convolve2d(gray_img, gauss_mask)

    dx = signal.convolve2d(gray_img, sobel_x)
    dy = signal.convolve2d(gray_img, sobel_y)

    # plt.imshow(dxy)
    # plt.show()
    dx2 = np.square(dx)
    dy2 = np.square(dy)
    dxy = np.multiply(dx, dy)

    r_val_matrix = np.zeros((height, width))

    for r in range(offset, height - offset):
        for c in range(offset, width - offset):
            Sum_dx2 = np.sum(dx2[r - offset:r + offset + 1, c - offset:c + offset + 1])
            Sum_dy2 = np.sum(dy2[r - offset:r + offset + 1, c - offset:c + offset + 1])
            Sum_dxy = np.sum(dxy[r - offset:r + offset + 1, c - offset:c + offset + 1])

            det = (Sum_dx2 * Sum_dy2) - (Sum_dxy ** 2)
            trace = Sum_dxy * 2
            response = det - k * (trace ** 2)
            r_val_matrix[r][c] = response

    cv2.normalize(r_val_matrix, r_val_matrix, 1, 0, cv2.NORM_MINMAX)

    for y in range(int(height * 0.05), int(height - height * 0.05)):
        for x in range(int(width * 0.05), int(width - width * 0.05)):
            if r_val_matrix[y, x] > threshold:
                vals = r_val_matrix[y - offset:y + 1 + offset, x - offset:x + 1 + offset]

                if r_val_matrix[y, x] == np.max(vals):
                    points.append((x, y))

    return points
