from matplotlib import pyplot as plt
import numpy as np
import cv2
import warp


def main():
    #img1 = cv2.imread(r"Benchmarks/Custom/IMG_1788.JPG")
    img1 = cv2.imread(r"Benchmarks/Custom/fear.jpg")
    #img1 = cv2.imread(r"Benchmarks/quad/DSC_0006.JPG")
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    #img2 = cv2.imread(r"Benchmarks/Custom/IMG_1787.JPG")
    img2 = cv2.imread(r"Benchmarks/Custom/rhoover.jpg")
    #img2 = cv2.imread(r"Benchmarks/quad/DSC_0007.JPG")
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    try:
        img_data = np.array([np.asarray(img1), np.asarray(img2)], dtype=object)
    except:
        img_data = np.array([np.asarray(img1), np.asarray(img2)])

    #kpt1, kpt2 = warp.matchFeatures(img1, img2)
    #pts = warp.show_image_get_clicks(img_data)
    #h = warp.computeHomography(pts[1], pts[0])
    #h = warp.computeHomography(kpt2, kpt1)

    # warped = warp.warpPerspective(img2, h)
    #img_test = warp.applyProjection(h, img1, img2)
    img_test = warp.warpImageToArea(img_data)
    plt.imshow(img_test)
    plt.show()


if __name__ == "__main__":
    main()
