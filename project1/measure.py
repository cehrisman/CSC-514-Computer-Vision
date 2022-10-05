import cv2
import numpy as np
import time
import matplotlib.pyplot as plt


fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

img = cv2.imread("data/gigi.jpg")

scale_percent = 0  # percent of original size
# .25 MPix
resized = cv2.resize(img, (300, 255))

cv2.imwrite("result/test.jpg", resized)

times = np.array([])
kern_size = np.array([])
image_size = np.array([])
# starts at an image of .25 MPix size and then increases to 8MPix over 10 intervals.
# it applies all odd sized kernels between 3x3 -> 15x15
for t in range(0, 110, 10):
    for i in range(3, 17, 2):
        for j in range(3, 17, 2):
            kernel = np.ones((i, j))
            print(i, j)
            print(resized.shape)
            start = time.time()
            cv2.filter2D(resized, ddepth=-1, kernel=kernel)
            end = time.time()
            times = np.append(times, end - start)
            kern_size = np.append(kern_size, i * j)
            image_size = np.append(image_size, resized.shape[0] * resized.shape[1])
    scale_percent = t
    width = int(3011 * scale_percent / 100)
    height = int(2150 * scale_percent / 100)
    resized = cv2.resize(img, (300 + height, 255 + width), interpolation=cv2.INTER_AREA)

surf = ax.scatter(kern_size, image_size, times)
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()