# Author: Songlin Piao

"""IDXDet model (base class)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from nn_skeleton import ModelSkeleton

class IDXDet(ModelSkeleton):
  def __init__(self, mc, input_channels=2): #in case of HOG, there are two input channels

      self.idx_channels = input_channels
      ModelSkeleton.__init__(self, mc)

  def _init_input(self):
      mc = self.mc

      self.ph_index_input = tf.placeholder(
          tf.int32, [mc.BATCH_SIZE, mc.IMAGE_HEIGHT, mc.IMAGE_WIDTH, self.idx_channels],
          name='index_input'
      )

      self.ph_mag_input = tf.placeholder(
          tf.float32, [mc.BATCH_SIZE, mc.IMAGE_HEIGHT, mc.IMAGE_WIDTH, self.idx_channels],
          name='mag_input'
      )

      if mc.IS_TRAINING:

          self.FIFOQueue = tf.FIFOQueue(
              capacity=mc.QUEUE_CAPACITY,
              dtypes=[tf.float32, tf.int32, tf.float32, tf.float32, tf.float32,
                      tf.float32, tf.float32],
              shapes=[[mc.IMAGE_HEIGHT, mc.IMAGE_WIDTH, 3],
                      [mc.IMAGE_HEIGHT, mc.IMAGE_WIDTH, self.idx_channels],
                      [mc.IMAGE_HEIGHT, mc.IMAGE_WIDTH, self.idx_channels],
                      [mc.ANCHORS, 1],
                      [mc.ANCHORS, 4],
                      [mc.ANCHORS, 4],
                      [mc.ANCHORS, mc.CLASSES]]
          )

          self.enqueue_op = self.FIFOQueue.enqueue_many(
              [self.ph_image_input, self.ph_index_input, self.ph_mag_input, self.ph_input_mask,
               self.ph_box_delta_input, self.ph_box_input, self.ph_labels]
          )

          self.image_input, self.index_input, self.mag_input, self.input_mask, self.box_delta_input, \
          self.box_input, self.labels = tf.train.batch(
              self.FIFOQueue.dequeue(), batch_size=mc.BATCH_SIZE,
              capacity=mc.QUEUE_CAPACITY)
      else:
          self.image_input = self.ph_image_input
          self.index_input = self.ph_index_input
          self.mag_input = self.ph_mag_input
          self.input_mask = self.ph_input_mask
          self.box_delta_input = self.ph_box_delta_input
          self.box_input = self.ph_box_input
          self.labels = self.ph_labels
