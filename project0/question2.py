from skimage import io
import matplotlib.pyplot as plt
import cv2
import numpy as np

I = io.imread(r"C:\Users\Caleb\School\ClassWork\CSC-514\project0\Images\gigi.jpg")
hsv = cv2.cvtColor(I, cv2.COLOR_BGR2HSV)
hsv[:,:,2] = hsv[:,:,2]*0.5

img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


fig = plt.figure(figsize=(10, 7))
fig.add_subplot(1, 2, 1)
plt.imshow(I)
plt.axis('off')
plt.title("Original")

fig.add_subplot(1, 2, 2)

# showing image
plt.imshow(img)
plt.axis('off')
plt.title("Dimmed")

plt.show()