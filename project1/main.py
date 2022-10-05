import numpy as np
from matplotlib import pyplot as plt
import skimage.io

import conv
from conv import *
from fft import *
from gen_hybrid_image import create_hybrid, laplacian

if __name__ == '__main__':

    img1 = cv2.imread(r"data/einstein.bmp")
    img2 = cv2.imread(r"data/marilyn.bmp")

    hybrid = create_hybrid(img1, img2, 1, 2, True)
    cv2.imwrite("result/MarsteinFFT.png", hybrid * 255)
    hybrid = create_hybrid(img1, img2, 1, 2)
    cv2.imwrite("result/Marstein.png", hybrid * 255)

    img1 = cv2.imread(r"data/plane.bmp")
    img2 = cv2.imread(r"data/bird.bmp")

    hybrid = create_hybrid(img1, img2, 2, 2)
    highPass_image = my_imfilter(img1, laplacian)
    cv2.imwrite("result/Highpass.png", highPass_image * 255)
    cv2.imwrite("result/Blane.png", hybrid * 255)
    hybrid = create_hybrid(img1, img2, 2, 2, True)
    cv2.imwrite("result/BlaneFFT.png", hybrid * 255)

    img1 = cv2.imread(r"data/dog.bmp")
    img2 = cv2.imread(r"data/cat.bmp")

    hybrid = create_hybrid(img1, img2, 7, 5, True)
    cv2.imwrite("result/CogFFT.png", hybrid * 255)
    hybrid = create_hybrid(img1, img2, 7, 5)
    cv2.imwrite("result/Cog.png", hybrid * 255)

    img1 = cv2.imread(r"data/submarine.bmp")
    img2 = cv2.imread(r"data/fish.bmp")

    hybrid = create_hybrid(img1, img2, 5, 4, True)
    cv2.imwrite("result/FibmarineFFT.png", hybrid * 255)
    hybrid = create_hybrid(img1, img2, 5, 4)
    cv2.imwrite("result/Fibmarine.png", hybrid * 255)

    img1 = cv2.imread(r"data/bicycle.bmp")
    img2 = cv2.imread(r"data/motorcycle.bmp")

    hybrid = create_hybrid(img1, img2, 7, 5)
    cv2.imwrite("result/cycle.png", hybrid * 255)
    hybrid = create_hybrid(img1, img2, 7, 5, True)
    cv2.imwrite("result/cycleFFT.png", hybrid * 255)
