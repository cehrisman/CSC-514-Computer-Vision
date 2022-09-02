import glob
from skimage import io
import numpy as np
import PIL
import time
import cv2

A = io.imread(r"C:\Users\Caleb\School\ClassWork\CSC-514\project0\Images\grizzlypeakg.png")
(m1, n1) = A.shape
start = time.time()

for i in range(m1):
    for j in range(n1):
        if A[i,j] <= 10 :
            A[i,j] = 0

loop_end = time.time() - start

# Logical Index Method
img = PIL.Image.open(r"C:\Users\Caleb\School\ClassWork\CSC-514\project0\Images\grizzlypeakg.png")
np_img = np.array(img)

start = time.time()

np_img[np_img <= 10] = 0

np_time = time.time() - start


print("For loop method output\n --- Took %.4f seconds to process ---" % loop_end)

print("For loop method output\n --- Took %.4f seconds to process ---" % np_time)

# reading in image data from Images Folder
path = glob.glob(r"C:\Users\Caleb\School\ClassWork\CSC-514\project0\Images\*.*")
cv_img = []
for img in path:
    n = cv2.imread(img)
    cv_img.append(n)

print(cv_img)


# Converting to NumPy arrays
np_images = np.asarray(cv_img, dtype=object)
loop_img = np_images

# For loop method
start = time.time()
for img in np_images:
    (m1, n1) = img.shape
    for i in range(m1):
        for j in range(n1):
            if img[i,j] <= 10 :
                img[i,j] = 0

end = time.time() - start
print("For loop method\n --- Took %.4f seconds to process ---" % end)

# Logical Index Method
start = time.time()

for img in np_images:
    img[img <= 10] = 0

end = time.time() - start

print("Logical Indexing method\n --- Took %.4f seconds to process ---" % end)


path = glob.glob(r"C:\Users\Caleb\School\ClassWork\CSC-514\project0\Images\*.*")
cv_img = []
for img in path:
    n = cv2.imread(img)
    cv_img.append(n)

print(cv_img)

path = glob.glob(r"C:\Users\Caleb\School\ClassWork\CSC-514\project0\Images\*.*")
cv_img_RGB = []
for img in path:
    n = cv2.imread(img)
    cv_img_RGB.append(n)

loop_img_RGB = np.asarray(cv_img, dtype=object)
np_img_RGB = loop_img_RGB

start = time.time()
for img in loop_img_RGB:
    (m1, n1, k1) = img.shape
    for i in range(m1):
        for j in range(n1):
            for k in range(k1):
                if img[i,j,k] <= 10 :
                    img[i,j,k] = 0

end = time.time() - start
print("For loop method output\n --- Took %.4f seconds to process ---" % end)

start = time.time()

for img in np_img_RGB:
    img[img <= 10] = 0

np_time = time.time() - start

print("For loop method output\n --- Took %.4f seconds to process ---" % np_time)





