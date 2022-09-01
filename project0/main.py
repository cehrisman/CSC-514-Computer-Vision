import skimage
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
import PIL
from PIL import ImageOps
import time

A = io.imread('grizzlypeakg.png')
(m1, n1) = A.shape
start = time.time()

for i in range(m1):
    for j in range(n1):
        if A[i,j] <= 10 :
            A[i,j] = 0

loop_end = time.time() - start

# Logical Index Method
img = PIL.Image.open("grizzlypeakg.png")
np_img = np.array(img)

start = time.time()

np_img[np_img <= 10] = 0

np_time = time.time() - start


# Plotting
fig = plt.figure(figsize=(10, 7))
rows = 2
columns = 1

fig.add_subplot(rows, columns, 1)
plt.imshow(A, cmap='gray')
plt.axis('off')
plt.title("For loop method output\n --- Took %.4f seconds to process ---" % loop_end)

fig.add_subplot(rows, columns, 2)
plt.imshow(np_img, cmap='gray')
plt.axis('off')
plt.title("For loop method output\n --- Took %.4f seconds to process ---" % np_time)
plt.show(block='false') # show window without stopping program

bridge_img = PIL.Image.open("bridge.jpg")
bridge_img = ImageOps.grayscale(bridge_img)
mountain_img = PIL.Image.open("mountain.jpg")
mountain_img = ImageOps.grayscale(mountain_img)
mountains_img = PIL.Image.open("mountains.jpg")
mountains_img = ImageOps.grayscale(mountains_img)
nature_img = PIL.Image.open("nature.jpg")
nature_img = ImageOps.grayscale(nature_img)
_img = PIL.Image.open("bridge.jpg")
bridge_img = ImageOps.grayscale(bridge_img)
np_images = np.array(10)
np_images[0] = np.array(bridge_img)




