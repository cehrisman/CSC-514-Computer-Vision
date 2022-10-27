import cv2
import numpy as np
import math
from matplotlib import pyplot as plt

import skimage.transform
from skimage.color import rgb2gray

def match(img1, img2, pts1, pts2):

    img1 = rgb2gray(img1)
    img2 = rgb2gray(img2)
    f, axarr1 = plt.subplots(1, 2)
    best_matches = []
    arr1 = np.array(pts1, dtype=object)
    arr2 = np.array(pts2, dtype=object)
    print(arr1.shape)
    for i in range(arr1.shape[0]):
        for j in range(arr2.shape[0]):
            orientation1 = arr1[i][1]
            orientation2 = arr2[j][1]
            pt1 = arr1[i][0]
            pt2 = arr2[i][0]

            window1 = img1[pt1[1] - 8:pt1[1] + 9, pt1[0] - 8:pt1[0] + 9]
            window2 = img2[pt2[1] - 8:pt2[1] + 9, pt2[0] - 8:pt2[0] + 9]

            rot1 = skimage.transform.rotate(window1, -1 * orientation1)
            rot2 = skimage.transform.rotate(window2, -1 * orientation2)

            compare_win1 = rot1[4:14, 4:14]
            compare_win2 = rot2[4:14, 4:14]
            axarr1[0].imshow(compare_win1, cmap='gray')
            axarr1[1].imshow(compare_win2, cmap='gray')
            plt.show()
