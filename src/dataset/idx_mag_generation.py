import cv2
import math
import numpy as np
from scipy.ndimage import filters
from lbp import *


# import hog


class IndexMapGen:

    def __init__(self, num_bins=9, method="HOG"):
        self.nbins = num_bins
        self.name = method
        self.channels = 1

        # self.desc = hog.HOGDescriptor(hog.Size(18, 36), hog.Size(12, 12), hog.Size(6, 6), hog.Size(6, 6), 9, 1, -1, hog.HOGDescriptor.L2Hys, 0.2, True)
        # self.hogcache = hog.HOGCache()
        # self.hogcache.init(self.desc, hog.Size(1242, 375), hog.Size(0, 0), hog.Size(0, 0), False, hog.Size(1, 1))

    def gen_idx_mag_hog_swig(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        input = hog.Mat.from_array(blur)
        self.hogcache.computeGradient(input, hog.Size(0, 0), hog.Size(0, 0))

        indexes = np.asarray(self.hogcache.qangle, dtype=np.int32)
        mag = np.asanyarray(self.hogcache.grad)
        return indexes, mag

    def gen_idx_mag_hog(self, img):

        sigma = 3
        angleScale = (self.nbins / math.pi)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        height, width = img.shape  # if img is gray scale, it has only two values
        imx = np.zeros(img.shape)
        filters.gaussian_filter(img, (sigma, sigma), (0, 1), imx)
        imy = np.zeros(img.shape)
        filters.gaussian_filter(img, (sigma, sigma), (1, 0), imy)

        # angle is from 0 to 2pi
        magnitude, angle = cv2.cartToPolar(imx, imy, angleInDegrees=False)
        angle = angle * angleScale - 0.5
        hidx = np.floor(angle)
        angle = angle - hidx
        angle_diff = 1.0 - angle

        indexes = np.zeros((height, width, 2), np.int32)
        mag = np.ones((height, width, 2), np.float32)

        hidx[hidx < 0] += self.nbins
        hidx[hidx >= self.nbins] -= self.nbins

        indexes[:, :, 0] = hidx.astype(np.int32)

        hidx = hidx + 1
        hidx[hidx >= self.nbins] = 0
        indexes[:, :, 1] = hidx.astype(np.int32)

        mag[:, :, 0] = np.multiply(magnitude, angle_diff).astype(np.float32)
        mag[:, :, 1] = np.multiply(magnitude, angle).astype(np.float32)
        return indexes, mag

    def gen_idx_mag_lbp(self, img):
        height, width = img.shape  # if img is gray scale, it has only two values
        blur = cv2.GaussianBlur(img, (3, 3), 0)
        indexes = np.zeros((height, width, 1), np.int32)
        mag = np.ones((height, width, 1), np.float32)

        for i in range(0, height):
            for j in range(0, width):
                indexes[i, j] = lbp_calculated_pixel(blur, i, j)

        return indexes, mag

    def gen_idx_mag_lbp_edge(self, img):

        sigma = 3
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imx = np.zeros(img.shape)
        filters.gaussian_filter(img, (sigma, sigma), (0, 1), imx)
        imy = np.zeros(img.shape)
        filters.gaussian_filter(img, (sigma, sigma), (1, 0), imy)

        temp = np.sqrt(imx ** 2 + imy ** 2)

        height, width = img.shape

        indexes = np.zeros((height, width, 1), np.int32)
        mag = np.ones((height, width, 1), np.float32)

        for i in range(0, height):
            for j in range(0, width):
                indexes[i, j] = lbp_calculated_pixel(temp, i, j)

        return indexes, mag

    def gen_idx_mag_quatization_16bin(self, img):

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        height, width = img.shape
        indexes = np.zeros((height, width, 2), np.int32)
        mag = np.ones((height, width, 2), np.float32)

        g = img.astype('int')
        idx0 = (g - 8) / 16
        idx1 = idx0 + 1

        indexes[:, :, 0] = idx0
        indexes[:, :, 1] = idx1
        mag[:, :, 0] = ((8 + 16 * idx1 - g) / 16.0).astype(np.float32)
        mag[:, :, 1] = (1.0 - mag[:, :, 0]).astype(np.float32)

        A = idx0 < 0
        idx0[A] = 0
        mag[A, 0] = 0
        mag[A, 1] = (8 + g[A])/16.0

        B = (idx0 == 15)
        idx1[B] = 15
        mag[B, 0] = (264 - g[B])/16.0
        mag[B, 1] = 0

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
