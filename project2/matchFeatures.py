import numpy as np
import math
from matplotlib import pyplot as plt

import skimage.transform
from skimage.color import rgb2gray
from matplotlib.patches import ConnectionPatch


def match(img1, img2, pts1, pts2, threshold):
    img1 = rgb2gray(img1)
    img2 = rgb2gray(img2)
    best_match = []
    for i in range(len(pts1)):
        curr_best = None
        best_SSD = 9999999
        for j in range(len(pts2)):
            SSD = 0

            # get 18x18 sample
            sample1 = img1[pts1[i][1] - 9: pts1[i][1] + 10, pts1[i][0] - 9: pts1[i][0] + 10]
            sample2 = img2[pts2[j][1] - 9: pts2[j][1] + 10, pts2[j][0] - 9: pts2[j][0] + 10]

            x_gradient = sample1[9, 10] - sample1[9, 8]
            y_gradient = sample1[10, 9] - sample1[8, 9]

            theta1 = math.atan2(y_gradient, x_gradient)

            x_gradient = sample2[9, 10] - sample2[9, 10]
            y_gradient = sample2[10, 9] - sample2[10, 9]

            theta2 = math.atan2(y_gradient, x_gradient)

            # rotate the sample negative the gradient to align it to 0
            rot1 = skimage.transform.rotate(sample1, -theta1)
            rot2 = skimage.transform.rotate(sample2, -theta2)

            # grab center 7x7 from rotated sample
            compare_samp1 = rot1[6:13, 6:13]
            compare_samp2 = rot2[6:13, 6:13]

            # Compute difference between all pixels and then square then sum
            SSD = np.linalg.norm(np.subtract(compare_samp1, compare_samp2))
            if SSD < threshold:
                if SSD < best_SSD:
                    best_SSD = SSD
                    # print(SSD, (i, j))
                    curr_best = (i, j)

        best_match.append(curr_best)
    return best_match


def showMatches(img, img2, matches, points, points1):
    fig = plt.figure()

    a1 = fig.add_subplot(1, 2, 1)
    a1.imshow(img)  # fragment is the name of the first image
    a2 = fig.add_subplot(1, 2, 2)
    a2.imshow(img2)  # complete is the name of the second image

    for i in range(len(matches)):
        if matches[i] is not None:
            con = ConnectionPatch(xyB=points[matches[i][0]], xyA=points1[matches[i][1]], coordsA="data", coordsB="data",
                                  axesA=a2, axesB=a1, color="red")

            a2.add_artist(con)
    plt.show()
