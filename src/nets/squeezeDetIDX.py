# Author: Songlin Piao 06/06/2019

"""SqueezeDet model with IDX CONV layer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import joblib
import tensorflow as tf
from nets.IDXDet import IDXDet

class SqueezeDetIDX(IDXDet):
  def __init__(self, mc, gpu_id=0, input_channels=2):
    with tf.device('/gpu:{}'.format(gpu_id)):
      IDXDet.__init__(self, mc, input_channels)

      self._add_forward_graph()
      self._add_interpretation_graph()
      self._add_loss_graph()
      self._add_train_graph()
      self._add_viz_graph()

  def _add_forward_graph(self):
    """NN architecture."""

    mc = self.mc
    if mc.LOAD_PRETRAINED_MODEL:
        assert tf.gfile.Exists(mc.PRETRAINED_MODEL_PATH), \
            'Cannot find pretrained model at the given path:' \
            '  {}'.format(mc.PRETRAINED_MODEL_PATH)
        self.caffemodel_weight = joblib.load(mc.PRETRAINED_MODEL_PATH)
        # self.caffemodel_weight = cPickle.load(open(mc.PRETRAINED_MODEL_PATH))

    conv1 = self._conv_layer( #375 x 1242
        'conv1', self.image_input, filters=64, size=3, stride=2,
        padding='SAME', freeze=True) #188 x 621

    idxconv1 = self._idx_conv2d_layer([self.index_input, self.mag_input], 2, 2, name='idxconv1', cellsize_=[7, 7],
                                      cells_=[2, 2],
                                      offset_=[0, 0, 1, -1, -1, 1, 1, 1, 0, 1, 1, 0, -1, 0], anchorsize_=[1, 1],
                                      relu=False, biased=False) #188 x 621

    concat1 = tf.concat(axis=-1, values=[conv1, idxconv1], name='concat1')
    bn1 = self.bn_layer(concat1, 'bn1', relu=False)

    pool1 = self._pooling_layer(
        'pool1', bn1, size=3, stride=2, padding='SAME')

    fire2 = self._fire_layer(
        'fire2', pool1, s1x1=16, e1x1=64, e3x3=64, freeze=False)
    fire3 = self._fire_layer(
        'fire3', fire2, s1x1=16, e1x1=64, e3x3=64, freeze=False)
    pool3 = self._pooling_layer(
        'pool3', fire3, size=3, stride=2, padding='SAME')

    fire4 = self._fire_layer(
        'fire4', pool3, s1x1=32, e1x1=128, e3x3=128, freeze=False)
    fire5 = self._fire_layer(
        'fire5', fire4, s1x1=32, e1x1=128, e3x3=128, freeze=False)
    pool5 = self._pooling_layer(
        'pool5', fire5, size=3, stride=2, padding='SAME')

    fire6 = self._fire_layer(
        'fire6', pool5, s1x1=48, e1x1=192, e3x3=192, freeze=False)
    fire7 = self._fire_layer(
        'fire7', fire6, s1x1=48, e1x1=192, e3x3=192, freeze=False)
    fire8 = self._fire_layer(
        'fire8', fire7, s1x1=64, e1x1=256, e3x3=256, freeze=False)
    fire9 = self._fire_layer(
        'fire9', fire8, s1x1=64, e1x1=256, e3x3=256, freeze=False)

    # Two extra fire modules that are not trained before
    fire10 = self._fire_layer(
        'fire10', fire9, s1x1=96, e1x1=384, e3x3=384, freeze=False)
    fire11 = self._fire_layer(
        'fire11', fire10, s1x1=96, e1x1=384, e3x3=384, freeze=False)
    dropout11 = tf.nn.dropout(fire11, self.keep_prob, name='drop11')

    num_output = mc.ANCHOR_PER_GRID * (mc.CLASSES + 1 + 4)
    self.preds = self._conv_layer(
        'conv12', dropout11, filters=num_output, size=3, stride=1,
        padding='SAME', xavier=False, relu=False, stddev=0.0001)

  def _fire_layer(self, layer_name, inputs, s1x1, e1x1, e3x3, stddev=0.01,
                  freeze=False):
      """Fire layer constructor.

      Args:
        layer_name: layer name
        inputs: input tensor
        s1x1: number of 1x1 filters in squeeze layer.
        e1x1: number of 1x1 filters in expand layer.
        e3x3: number of 3x3 filters in expand layer.
        freeze: if true, do not train parameters in this layer.
      Returns:
        fire layer operation.
      """

      sq1x1 = self._conv_layer(
          layer_name + '/squeeze1x1', inputs, filters=s1x1, size=1, stride=1,
          padding='SAME', stddev=stddev, freeze=freeze)
      ex1x1 = self._conv_layer(
          layer_name + '/expand1x1', sq1x1, filters=e1x1, size=1, stride=1,
          padding='SAME', stddev=stddev, freeze=freeze)
      ex3x3 = self._conv_layer(
          layer_name + '/expand3x3', sq1x1, filters=e3x3, size=3, stride=1,
          padding='SAME', stddev=stddev, freeze=freeze)

      return tf.concat([ex1x1, ex3x3], 3, name=layer_name + '/concat')
