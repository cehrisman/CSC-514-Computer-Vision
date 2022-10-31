import numpy as np
import math
from skimage.color import rgb2gray


# makes a basic descriptor using the gradient of the point to find magnitude and orientation of gradient
def computeGradient_Mag(img, pts):
    img = rgb2gray(img)
    grad_mag = np.zeros(len(pts))
    theta = np.zeros(len(pts))

    for i in range(len(pts)):
        x_gradient = img[pts[i][1], pts[i][0] + 1] - img[pts[i][1], pts[i][0] - 1]
        y_gradient = img[pts[i][1] + 1, pts[i][0]] - img[pts[i][1] - 1, pts[i][0]]

        grad_mag[i] = math.sqrt((x_gradient**2) + (y_gradient**2))
        theta[i] = math.atan2(y_gradient, x_gradient)

    return grad_mag, (theta * 180/math.pi)


