import numpy as np
import cv2
import showfeatures


def main():
    img = cv2.imread("data/Yosemite/Yosemite1.jpg")
    img = showfeatures.drawSquare(img, (50, 50), 20, 0)

    cv2.imshow("Draw Square", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
