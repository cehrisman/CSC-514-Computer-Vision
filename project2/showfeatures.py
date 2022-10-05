import numpy as np
import cv2
import matplotlib
from matplotlib import pyplot as plt
import math


#   Draws a feature square on a given image with the center point, scale and orientation angle given
def drawSquare(img, point, scale, theta):
    length = scale
    rot_square = (point, (length, length), theta)

    box = cv2.boxPoints(rot_square)
    box = np.int0(box)
    print(box)
    cv2.drawContours(img, [box], 0, (0, 0, 255), 1)
    midPoint = ((box[2][0] + box[1][0]) / 2, (box[1][1] + box[2][1]) / 2)
    print(midPoint)
    midPoint = np.int0(midPoint)
    cv2.line(img, point, midPoint, (0, 0, 255), 1)

    return img
