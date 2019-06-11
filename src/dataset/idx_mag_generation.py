import cv2
import hog
import numpy as np


def gen_idx_mag_hog(img, hogcache):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    input = hog.Mat.from_array(blur)
    hogcache.computeGradient(input, hog.Size(0, 0), hog.Size(0, 0))

    indexes = np.asarray(hogcache.qangle, dtype=np.int32)
    mag = np.asanyarray(hogcache.grad)
    return indexes, mag


def gen_idx_mag_hog_batch(imgs, hogcache):

    input_shape = imgs.shape
    batch_size = input_shape[0]

    batch_idxs = np.zeros((batch_size, input_shape[1], input_shape[2], 2), dtype=np.int32)
    batch_mags = np.zeros((batch_size, input_shape[1], input_shape[2], 2), dtype=np.float32)

    for i in range(batch_size):
        idx_, mag_ = gen_idx_mag_hog(imgs[i], hogcache)
        batch_idxs[i] = idx_
        batch_mags[i] = mag_
    return batch_idxs, batch_mags