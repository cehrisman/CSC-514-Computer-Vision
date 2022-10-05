from numpy.fft import fft2, ifft2, fftshift, ifftshift
from scipy import misc
import numpy as np
import skimage

from gaussian import make_gaussian_filter
from main import *
from conv import *
from gaussian import *
import cv2


# Using gaussian blur for the low pass
def lowPass(image, cutoff, FFT=False):
    return gaussian_blur(image, cutoff, FFT)


def highPass(image, cutoff, FFT=False):
    return (image / 255) - lowPass(image, cutoff, FFT)


