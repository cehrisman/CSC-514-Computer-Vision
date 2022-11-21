import numpy as np
import cv2
import skimage
import showfeatures
import harris
import matchFeatures
import makeDescriptor
from matplotlib import pyplot as plt


def main():
    f, axarr = plt.subplots(1, 2)
    #img = skimage.io.imread("data/Yosemite/Yosemite1.jpg")
    #img = skimage.transform.rotate(img, 20)
    img = skimage.io.imread("data/wall/img1.ppm")
    #img = skimage.io.imread("data/graf/img1.ppm")
    #img = skimage.io.imread("data/triangle/t1.jpg")
    img_copy = np.copy(img)
    points = harris.harris(img, 0.04, .2)
    points = np.asarray(points)
    mags1, thetas1 = makeDescriptor.computeGradient_Mag(img, points)
    for i in range(len(points)):
        img = showfeatures.drawSquare(img, points[i], thetas1[i])

    axarr[0].imshow(img)
    img2 = skimage.io.imread("data/Yosemite/Yosemite2.jpg")
    #img2 = skimage.transform.rotate(img2, 20)
    img2 = skimage.io.imread("data/graf/img2.ppm")
    #img2 = skimage.io.imread("data/wall/img2.ppm")
    #img2 = skimage.io.imread("data/triangle/t1.jpg")
    img2_copy = np.copy(img2)
    points1 = harris.harris(img2, 0.04, .3)
    points1 = np.asarray(points1)
    mags2, thetas2 = makeDescriptor.computeGradient_Mag(img2, points1)
    for i in range(len(points1)):
        img2 = showfeatures.drawSquare(img2, points1[i], thetas2[i])
    axarr[1].imshow(img2)
    plt.show()

    matches = matchFeatures.match(img_copy, img2_copy, points, points1, 0.75)

    if matches is not None:
        matchFeatures.showMatches(img_copy, img2_copy, matches, points, points1)


if __name__ == "__main__":
    main()
