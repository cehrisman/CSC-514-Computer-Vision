import matplotlib.pyplot as plt
import numpy as np
import cv2
import skimage


# Gather user click inputs for points to warp
def show_image_get_clicks(img):
    fig = plt.figure(figsize=(10, 5))
    pts = []
    num_imgs = len(img)
    rows = 1
    cols = 2

    for i in range(len(img)):
        plt.imshow(img[i])
        pts.append(plt.ginput(-1))
        plt.axis('off')
        plt.title('User Point Selection')

    pts = np.asarray(pts, dtype=object)
    print("Click Coords: ")
    print(pts)
    return pts


# Computes a homography matrix given matching points
def computeHomography(src, dest):
    n = src.shape[0]
    A = []
    for i in range(n):
        A.append(getPartialA(src[i], dest[i]))
    A = np.concatenate(A, axis=0)

    u, s, vh = np.linalg.svd(A, full_matrices=True)
    homography = vh[-1].reshape((3, 3))
    return homography


def getPartialA(src, dest):
    x, y = src[0], src[1]
    x_dest, y_dest = dest[0], dest[1]

    return np.array([[0, 0, 0, -x, -y, -1, y_dest * x, y_dest * y, y_dest],
                     [x, y, 1, 0, 0, 0, -x_dest * x, -x_dest * y, -x_dest]])


# Applies the Homography matrix - H
def warpPerspective(src, H):
    dest = np.zeros_like(src)
    r, c, _ = np.shape(src)
    H = np.linalg.inv(H)
    print("H matrix: ")
    print(H)
    for i in range(r):
        for j in range(c):
            x_warp = ((H[0][0] * j) + (H[0][1] * i + H[0][2])) // \
                     ((H[2][0] * j) + (H[2][1] * i + H[2][2]))
            y_warp = ((H[1][0] * j) + (H[1][1] * i + H[1][2])) // \
                     ((H[2][0] * j) + (H[2][1] * i + H[2][2]))
            x_warp = round(x_warp)
            y_warp = round(y_warp)
            if abs(x_warp) >= c or abs(y_warp) >= r:
                pass
            else:
                val = src[y_warp, x_warp]
                dest[i, j] = val

    return dest


def matchFeatures(img1, img2):
    img1_g = cv2.cvtColor(skimage.img_as_ubyte(img1), cv2.COLOR_RGB2GRAY)
    img2_g = cv2.cvtColor(skimage.img_as_ubyte(img2), cv2.COLOR_RGB2GRAY)

    sift = cv2.SIFT_create()

    kp1, des1 = sift.detectAndCompute(img1_g, None)
    kp2, des2 = sift.detectAndCompute(img2_g, None)

    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = list(sorted(matches, key=lambda x: x.distance))[0:20]

    list_kp1 = [kp1[mat.queryIdx].pt for mat in matches]
    list_kp2 = [kp2[mat.trainIdx].pt for mat in matches]
    return np.asarray(list_kp1), np.asarray(list_kp2)
