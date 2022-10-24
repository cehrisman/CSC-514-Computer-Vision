import numpy as np
import cv2
import matplotlib
from matplotlib import pyplot as plt
import math


#   Draws a feature square on a given image with the center point, scale and orientation angle given
def drawSquare(img, point, scale, theta):
    length = scale
    theta = theta * math.pi / 180.0

    # rot_square = (point, (length, length), theta * math.pi / 180.0)
    # box = cv2.boxPoints(rot_square)
    # box = np.int0(box)
    # cv2.drawContours(img, [box], 0, (0, 0, 255), 1)
    p1 = (int(point[0] - 0.5 * length * (math.sin(theta) + math.cos(theta))),
          int(point[1] - 0.5 * length * (math.sin(theta) - math.cos(theta))))

    p2 = (int(point[0] + 0.5 * length * (math.sin(theta) - math.cos(theta))),
          int(point[1] - 0.5 * length * (math.sin(theta) + math.cos(theta))))

    p3 = (int(point[0] + 0.5 * length * (math.sin(theta) + math.cos(theta))),
          int(point[1] + 0.5 * length * (math.sin(theta) - math.cos(theta))))

    p4 = (int(point[0] - 0.5 * length * (math.sin(theta) - math.cos(theta))),
          int(point[1] + 0.5 * length * (math.sin(theta) + math.cos(theta))))


    cv2.line(img, p1, p2, (0, 0, 255), 1)
    cv2.line(img, p2, p3, (0, 0, 255), 1)
    cv2.line(img, p3, p4, (0, 0, 255), 1)
    cv2.line(img, p4, p1, (0, 0, 255), 1)

    midpoint = (int((p4[0] + p1[0]) / 2), int((p4[1] + p1[1]) / 2))
    y = int(point[1] + 10 * math.cos(theta))
    x = int(point[0] + 10 * math.sin(theta))
    # print([y, x])
    cv2.line(img, point, midpoint, (0, 0, 255), 1)

    return img
