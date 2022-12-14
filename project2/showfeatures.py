import numpy as np
import cv2
import matplotlib
from matplotlib import pyplot as plt
import math


#   Draws a feature square on a given image with the center point, scale and orientation angle given
def drawSquare(img, point, theta):
    length = 10


    p1 = (int(point[0] - 0.5 * length * (math.sin(theta) + math.cos(theta))),
          int(point[1] - 0.5 * length * (math.sin(theta) - math.cos(theta))))

    p2 = (int(point[0] + 0.5 * length * (math.sin(theta) - math.cos(theta))),
          int(point[1] - 0.5 * length * (math.sin(theta) + math.cos(theta))))

    p3 = (int(point[0] + 0.5 * length * (math.sin(theta) + math.cos(theta))),
          int(point[1] + 0.5 * length * (math.sin(theta) - math.cos(theta))))

    p4 = (int(point[0] - 0.5 * length * (math.sin(theta) - math.cos(theta))),
          int(point[1] + 0.5 * length * (math.sin(theta) + math.cos(theta))))

    # print(p1)
    cv2.line(img, p1, p2, (255, 0, 0), 1)
    cv2.line(img, p2, p3, (255, 0, 0), 1)
    cv2.line(img, p3, p4, (255, 0, 0), 1)
    cv2.line(img, p4, p1, (255, 0, 0), 1)

    midpoint = (int((p3[0] + p4[0]) / 2), int((p3[1] + p4[1]) / 2))
    # print([y, x])
    cv2.line(img, point, midpoint, (255, 0, 0), 1)

    return img
