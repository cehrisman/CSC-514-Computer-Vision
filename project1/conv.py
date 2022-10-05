from main import *
import numpy as np


def my_imfilter(image, kernel):
    kernel = np.flipud(np.fliplr(kernel))

    if kernel.ndim != 2:
        print("ERROR: Kernel dimension incorrect. Only 2D accepted")
        return

    (kernH, kernW) = kernel.shape[:2]
    (height, width) = image.shape[:2]

    imgCopy = image

    padW = kernW // 2
    padH = kernH // 2
    postImg = np.zeros(((image.shape[0] + padH * 2), (image.shape[1] + padW * 2), image.shape[2]))

    for row in range(padH, height - padH - 2):
        for col in range(padW, width - padW - 2):
            for pixel in range(image.shape[2]):
                center = image[row - padH: row + padH + 1, col - padW: col + padW + 1, pixel]

                postImg[row, col, pixel] = (center * kernel).sum() / 255

    postImg = postImg[0: height, 0: width]
    return postImg
