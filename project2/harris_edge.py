import cv2
import numpy as np
import math


def harris(img, size, k, threshold):
    points = []
    # Convert image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    dx = cv2.Sobel(gray_img, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(gray_img, cv2.CV_32F, 0, 1, ksize=3)
    dxy = dx*dy


    dx2 = dx**2
    dy2 = dy**2
    gaussian_dx2 = cv2.GaussianBlur(dx2, (5, 5), 0)
    gaussian_dy2 = cv2.GaussianBlur(dy2, (5, 5), 0)
    gaussian_dxy = cv2.GaussianBlur(dxy, (5, 5), 0)

    det = gaussian_dx2 * gaussian_dy2 - gaussian_dxy**2
    trace = gaussian_dx2 + gaussian_dy2
    r_matrix = det - (k * trace**2)

    cv2.normalize(r_matrix, r_matrix, 0, 1, cv2.NORM_MINMAX)

    offset = 5
    height = img.shape[0]
    width = img.shape[1]
    for y in range(offset, height - offset):
        for x in range(offset, width - offset):
            if r_matrix[y, x] > threshold:

                # Checks to see if there nearby features that are larger. Prevents cluttering of similar features
                if r_matrix[y, x] == np.max(r_matrix[y - offset:y + 1 + offset, x - offset:x + 1 + offset]):
                    theta = math.degrees(math.atan(dy[y, x]/dx[y, x]))
                    points.append([(x, y), theta, r_matrix[y, x]])

    return points
