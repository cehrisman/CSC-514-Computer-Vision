import numpy as np
import cv2
import skimage
import showfeatures
import harris_edge
import matchFeatures
from matplotlib import pyplot as plt

def main():

    f, axarr = plt.subplots(1, 2)
    img = skimage.io.imread("data/Yosemite/Yosemite1.jpg")

    points = harris_edge.harris(img, 0.06, .03)
    for i in range(len(points)):
        img = showfeatures.drawSquare(img, points[i][0], 40 * points[i][2], points[i][1])

    axarr[0].imshow(img)

    img2 = skimage.io.imread("data/Yosemite/Yosemite2.jpg")
    points1 = harris_edge.harris(img2, 0.06, .03)

    for i in range(len(points1)):
        img2 = showfeatures.drawSquare(img2, points1[i][0], 40 * points1[i][2], points1[i][1])
    axarr[1].imshow(img2)
    plt.show()
    matchFeatures.match(img, img2, points, points1)

if __name__ == "__main__":
    main()
