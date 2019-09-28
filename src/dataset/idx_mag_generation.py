import cv2
import math
import numpy as np
from scipy.ndimage import filters
#from pylab import *
from lbp import *

class IndexMapGen:

    def __init__(self, num_bins = 9, method = "HOG"):
        self.nbins = num_bins
        self.name = method
        self.channels = 1

    def gen_idx_mag_hog_swig(self, img, hogcache):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        input = hog.Mat.from_array(blur)
        hogcache.computeGradient(input, hog.Size(0, 0), hog.Size(0, 0))

        indexes = np.asarray(hogcache.qangle, dtype=np.int32)
        mag = np.asanyarray(hogcache.grad)
        return indexes, mag

    def gen_idx_mag_hog(self, img):

        sigma = 3
        angleScale = (float)(self.nbins / math.pi)
        height, width = img.shape  # if img is gray scale, it has only two values
        imx = np.zeros(img.shape)
        filters.gaussian_filter(img, (sigma, sigma), (0, 1), imx)
        imy = np.zeros(img.shape)
        filters.gaussian_filter(img, (sigma, sigma), (1, 0), imy)

        magnitude, angle = cv2.cartToPolar(imx, imy, angleInDegrees=False)
        angle = angle * angleScale - 0.5
        hidx = np.floor(angle)
        angle = angle - hidx

        indexes = np.zeros((height, width, 2), np.uint32)
        mag = np.ones((height, width, 2), np.float32)

        mag[:, :, 1] = np.multiply(magnitude, angle)
        mag[:, :, 0] = magnitude - mag[:, :, 1]

        indexes[:, :, 0] = hidx
        hidx = hidx + 1
        hidx[hidx == self.nbins] = 0
        indexes[:, :, 1] = hidx

        return indexes, mag

    def gen_idx_mag_lbp(self, img):
        height, width = img.shape #if img is gray scale, it has only two values
        blur = cv2.GaussianBlur(img, (3, 3), 0)
        indexes = np.zeros((height, width, 1), np.uint32)
        mag = np.ones((height, width, 1), np.float32)

        for i in range(0, height):
            for j in range(0, width):
                indexes[i, j] = lbp_calculated_pixel(blur, i, j)

        return indexes, mag


    def gen_idx_mag_lbp_edge(self, img):

        sigma = 3
        imx = zeros(img.shape)
        filters.gaussian_filter(img, (sigma, sigma), (0, 1), imx)
        imy = zeros(img.shape)
        filters.gaussian_filter(img, (sigma, sigma), (1, 0), imy)

        temp = sqrt(imx ** 2 + imy **2)

        height, width = img.shape

        indexes = np.zeros((height, width, 1), np.uint32)
        mag = np.ones((height, width, 1), np.float32)

        for i in range(0, height):
            for j in range(0, width):
                indexes[i, j] = lbp_calculated_pixel(temp, i, j)

        return indexes, mag

    def gen_idx_mag_quatization_16bin(self, img):
        height, width = img.shape
        indexes = np.zeros((height, width, 2), np.uint32)
        mag = np.ones((height, width, 2), np.float32)

        g = img.astype('int')
        idx0 = (g - 8) / 16
        idx1 = idx0 + 1

        for i in range(0, height):
            for j in range(0, width):
                if idx0[i, j] < 0:
                    indexes[i, j,0] = 0
                    mag[i, j, 0] = 0
                    indexes[i, j, 1] = 0
                    mag[i, j, 1] = (8 + g[i, j])/16.0
                elif idx0[i, j] == 15:
                    indexes[i, j, 0] = 15
                    mag[i, j, 0] = (264 - g[i, j])/16.0
                    indexes[i, j, 1] = 15
                    mag[i, j, 1] = 0
                else:
                    indexes[i, j, 0] = idx0[i, j]
                    indexes[i, j, 1] = idx1[i, j]
                    mag[i, j, 0] = (8 + 16 * idx1[i, j] - g[i, j])/16.0
                    mag[i, j, 1] = 1.0 - mag[i, j, 0]

        return indexes, mag


    def get_index_generation_fun(self):

        if self.name == "HOG":
            self.channels = 2
            return self.gen_idx_mag_hog
        elif self.name == "LBP":
            self.channels = 1
            assert self.nbins == 255
            return self.gen_idx_mag_lbp
        elif self.name == "LBP-EDGE":
            self.channels = 1
            assert self.nbins == 255
            return self.gen_idx_mag_lbp_edge
        elif self.name == "QUANTIZE":
            self.channels = 2
            assert self.nbins == 16
            return self.gen_idx_mag_quatization_16bin
        else:
            return None