import matplotlib.pyplot as plt
import numpy as np
import fft
import cv2
import gaussian
import conv
from scipy import misc
import skimage

sharpen = np.array([[0, -1, 0],
                    [-1, 5, -1],
                    [0, -1, 0]])

outline = np.array([[-1, -1, -1],
                    [-1, 8, -1],
                    [-1, -1, -1]])

random = np.array([[0, 0, 0],
                   [1, 1, 0],
                   [0, 0, 0]])

emboss = np.array([[-2, -1, 0],
                   [-1, 1, 1],
                   [0, 1, 2]])

blur = np.array([[0.0625, 0.125, 0.0625],
                 [0.125, 0.25, 0.125],
                 [0.0625, 0.125, 0.0625]])
lowpass = np.array([[1/9, 1/9, 1/9],
                    [1/9, 1/9, 1/9],
                    [1/9, 1/9, 1/9]])

laplacian = np.array([[0, 1, 0],
                      [1, -4, 1],
                      [0, 1, 0]])

highpass = np.array([[-1, -1, -1, -1, -1],
                     [-1, 1, 2, 1, -1],
                     [-1, 2, 4, 2, -1],
                     [-1, 1, 2, 1, -1],
                     [-1, -1, -1, -1, -1]])
SOBEL = np.asarray([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])
UNEVEN_SOBEL = np.asarray([[-2, -1, 0, 1, 2],
                           [-4, -2, 0, 2, 4],
                           [-2, -1, 0, 1, 2]])


def create_hybrid(image1, image2, sigma1, sigma2, fourier=False):
    hybrid = np.asarray([])

    if len(image1.shape) < 3:
        image1.reshape(image1.shape[0], image1.shape[1], 1)

    if len(image2.shape) < 3:
        image2.reshape(image2.shape[0], image2.shape[1], 1)

    if not fourier:
        lowPass_image = fft.lowPass(image1, sigma1)
        highPass_image = fft.highPass(image2, sigma2)
        # highPass_image = conv.my_imfilter(image2, outline / 2)
        hybrid = lowPass_image + highPass_image
        print_pyramid(hybrid)
    else:
        highImg = fft.highPass(image2, sigma1, True)
        lowImg = fft.lowPass(image1, sigma2, True)
        hybrid = highImg + lowImg
        print_pyramid(hybrid, True)
    return hybrid


def print_pyramid(image, FFT=False):
    scale_percent = 220  # percent of original size
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    layer = resized.copy()
    layer2 = cv2.pyrDown(layer)
    layer3 = cv2.pyrDown(layer2)
    layer4 = cv2.pyrDown(layer3)
    layer5 = cv2.pyrDown(layer4)
    if FFT:
        cv2.imshow("FFT 1", layer)
        cv2.imshow("FFT 2", layer2)
        cv2.imshow("FFT 3", layer3)
        cv2.imshow("FFT 4", layer4)
        cv2.imshow("FFT 5", layer5)
    else:
        cv2.imshow("1", layer)
        cv2.imshow("2", layer2)
        cv2.imshow("3", layer3)
        cv2.imshow("4", layer4)
        cv2.imshow("5", layer5)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
