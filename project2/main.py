import numpy as np
import cv2
import showfeatures
import harris_edge

def main():
    img = cv2.imread("data/Yosemite/Yosemite1.jpg")
    points = harris_edge.harris(img, 7, 0.06, .3)

    for i in range(len(points)):
        img = showfeatures.drawSquare(img, points[i][0], 25 * points[i][2], points[i][1])

    cv2.imshow("Draw Square", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
