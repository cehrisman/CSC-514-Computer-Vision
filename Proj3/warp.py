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
    return pts


# Computes a homography matrix given matching points
def computeHomography(src, dest):
    n = 2 * min([len(src), len(dest)])
    A = np.zeros((n, 9))

    for i in range(int(n / 2)):
        x_d, y_d = dest[i]
        x_s, y_s = src[i]
        A[2 * i, :] = np.asarray([[x_s, y_s, 1, 0, 0, 0, -x_d * x_s, -x_d * y_s, -x_d]])
        A[2 * i + 1, :] = np.asarray([[0, 0, 0, x_s, y_s, 1, -y_d * x_s, -y_d * y_s, -y_d]])

    u, s, vh = np.linalg.svd(A)
    homography = vh[-1].reshape((3, 3))
    return homography


# Applies the Homography matrix - H
def warpPerspective(src, H):
    dest = np.zeros_like(src)
    r, c, _ = np.shape(src)
    for i in range(r):
        for j in range(c):

            x_warp, y_warp = computeWarp([j, i, 1], H)
            if abs(x_warp) >= c or abs(y_warp) >= r:
                pass
            else:
                val = src[y_warp, x_warp]
                dest[i, j] = val

    return dest


def computeWarp(pt, H):
    pt_warp = np.matmul(H, pt)
    return pt_warp / pt_warp[-1]


def matchFeatures(img1, img2):
    img1_g = cv2.cvtColor(skimage.img_as_ubyte(img1), cv2.COLOR_RGB2GRAY)
    img2_g = cv2.cvtColor(skimage.img_as_ubyte(img2), cv2.COLOR_RGB2GRAY)

    sift = cv2.SIFT_create()

    kp1, des1 = sift.detectAndCompute(img1_g, None)
    kp2, des2 = sift.detectAndCompute(img2_g, None)

    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = list(sorted(matches, key=lambda x: x.distance))[0:20]

    # img = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # plt.imshow(img), plt.show()
    list_kp1 = [kp1[mat.queryIdx].pt for mat in matches]
    list_kp2 = [kp2[mat.trainIdx].pt for mat in matches]
    return np.asarray(list_kp1), np.asarray(list_kp2)


def applyProjection(H, img1, img2):
    edges = np.zeros((4, 2))

    for (i, x, y) in [(0, 0, 0), (1, 0, len(img2)), (2, len(img2[0]), 0), (3, len(img2[0]), len(img2))]:
        pt = np.asarray([x, y, 1]).reshape(-1, 1)
        pt2 = np.linalg.inv(H).dot(pt)
        edges[i] = [pt2[0] / pt2[2], pt2[1] / pt2[2]]

    ([minx, miny], [maxx, maxy]) = (np.min(edges, axis=0), np.max(edges, axis=0))
    x, pad_x, y, pad_y = (len(img1[0]), 0, len(img1), 0)
    if maxx > x:
        x = int(maxx) + 1
    if minx < 0:
        pad_x = -int(minx)
        x += pad_x
    if maxy > y:
        y = int(maxy) + 1
    if miny < 0:
        pad_y = -int(miny)
        y += pad_y

    projection = np.zeros((y, x, 3), dtype='int32')

    for r in range(len(img1)):
        for c in range(len(img1[0])):
            projection[r + pad_y, c + pad_x] = img1[r, c]

    for r in range(y):
        for c in range(x):
            pt = np.asarray([c - pad_x, r - pad_y, 1]).reshape(-1, 1)
            pt2 = computeWarp(pt, H)
            xx, yy = (int(pt2[0] / pt2[2]), int(pt2[1] / pt2[2]))
            if xx > 0 and yy > 0:
                try:

                    projection[r, c] = img2[yy, xx]
                except:
                    pass

    return projection


def warpImageToArea(img):
    kp1, kp2 = show_image_get_clicks(img)

    corners = np.array([[0, 0],
                        [0, len(img[1])],
                        [len(img[1][0]), 0],
                        [len(img[1][0]), len(img[1])]])
    print(kp1)
    h = computeHomography(kp1, corners)

    img = applyProjection(h, img[0], img[1])

    return img
