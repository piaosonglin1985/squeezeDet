# Author: Bichen Wu (bichen@berkeley.edu) 08/25/2016

"""SqueezeDet Demo.

In image detection mode, for a given image, detect objects and draw bounding
boxes around them. In video detection mode, perform real-time detection on the
video stream.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import time
import os
import glob

import numpy as np
import tensorflow as tf

from config import *
from train import _draw_box
import dataset.hog as hog
from nets.squeezeDetIDX import SqueezeDetIDX
from nets.squeezeDetPlusIDX import SqueezeDetPlusIDX
from dataset.idx_mag_generation import gen_idx_mag_hog

#python src/demoIdxConv.py --mode video --checkpoint /home/songlin/squeezeDetLog/train/model.ckpt-48500 --input_path ./KITTI/tracking/data_tracking/testing/image_02/0011/%06d.png
#--out_dir ./data/outidx --demo_net squeezeDet+

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'mode', 'image', """'image' or 'video'.""")
tf.app.flags.DEFINE_string(
    'checkpoint', './data/model_checkpoints/squeezeDet/model.ckpt-87000',
    """Path to the model parameter file.""")
tf.app.flags.DEFINE_string(
    'input_path', './data/sample.png',
    """Input image or video to be detected. Can process glob input such as """
    """./data/00000*.png.""")
tf.app.flags.DEFINE_string(
    'out_dir', './data/out/', """Directory to dump output image or video.""")
tf.app.flags.DEFINE_string(
    'demo_net', 'squeezeDet', """Neural net architecture.""")

desc = hog.HOGDescriptor(hog.Size(18, 36), hog.Size(12, 12), hog.Size(6, 6), hog.Size(6, 6), 9, 1, -1,
                             hog.HOGDescriptor.L2Hys, 0.2, True)
cache = hog.HOGCache()
cache.init(desc, hog.Size(1242, 375), hog.Size(0, 0), hog.Size(0, 0), False, hog.Size(1, 1))


def video_demo():
  """Detect videos."""

  cap = cv2.VideoCapture(FLAGS.input_path)

  # Define the codec and create VideoWriter object
  # fourcc = cv2.cv.CV_FOURCC(*'XVID')
  #fourcc = cv2.VideoWriter_fourcc(*'x264')
  fourcc = cv2.VideoWriter_fourcc(*'MJPG')
  #in_file_name = os.path.split(FLAGS.input_path)[1]
  out_file_name = os.path.join(FLAGS.out_dir, 'out_result.avi')
  out = cv2.VideoWriter(out_file_name, fourcc, 30.0, (1242, 375), True)
  # out = VideoWriter(out_file_name, frameSize=(1242, 375))
  #out.open()

  assert FLAGS.demo_net == 'squeezeDet' or FLAGS.demo_net == 'squeezeDet+', \
      'Selected nueral net architecture not supported: {}'.format(FLAGS.demo_net)

  with tf.Graph().as_default():
    # Load model
    if FLAGS.demo_net == 'squeezeDet':
      mc = kitti_squeezeDet_config()
      mc.BATCH_SIZE = 1
      # model parameters will be restored from checkpoint
      mc.LOAD_PRETRAINED_MODEL = False
      model = SqueezeDetIDX(mc, FLAGS.gpu)
    elif FLAGS.demo_net == 'squeezeDet+':
      mc = kitti_squeezeDetPlus_config()
      mc.BATCH_SIZE = 1
      mc.LOAD_PRETRAINED_MODEL = False
      model = SqueezeDetPlusIDX(mc, FLAGS.gpu)

    # we should not restore variable with the name 'iou' because it depends on the batch size and it is fixed to zero initially.
    restore_var = [v for v in tf.global_variables() if
                    "iou" not in v.name]  # Keep only the variables, whose name does not contain 'iou'
    saver = tf.train.Saver(restore_var)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
      saver.restore(sess, FLAGS.checkpoint)

      saver.save(sess, "/tmp/resave_checkpoint_idxconv", global_step=0)
      tf.train.write_graph(sess.graph.as_graph_def(), "/tmp", "squeezedet_idxconv.pbtx")

      times = {}
      count = 0
      while cap.isOpened():
        t_start = time.time()
        count += 1
        out_im_name = os.path.join(FLAGS.out_dir, str(count).zfill(6)+'.jpg')

        # Load images from video and crop
        ret, frame = cap.read()
        if ret==True:
          # crop frames
          #frame = frame[500:-205, 239:-439, :]
          idx_, mag_ = gen_idx_mag_hog(frame, cache)
          im_input = frame.astype(np.float32) - mc.BGR_MEANS
        else:
          break

        t_reshape = time.time()
        times['reshape']= t_reshape - t_start

        # Detect
        det_boxes, det_probs, det_class = sess.run(
            [model.det_boxes, model.det_probs, model.det_class],
            feed_dict={model.image_input:[im_input], model.index_input:[idx_], model.mag_input:[mag_]})

        print("pred_class_probs shape: ", det_class.shape)
        print("pred_confidence_score: ", det_probs.shape)
        print("bbox_delta: ", det_boxes.shape)

        t_detect = time.time()
        times['detect']= t_detect - t_reshape
        
        # Filter
        final_boxes, final_probs, final_class = model.filter_prediction(
            det_boxes[0], det_probs[0], det_class[0])

        keep_idx    = [idx for idx in range(len(final_probs)) \
                          if final_probs[idx] > mc.PLOT_PROB_THRESH]
        final_boxes = [final_boxes[idx] for idx in keep_idx]
        final_probs = [final_probs[idx] for idx in keep_idx]
        final_class = [final_class[idx] for idx in keep_idx]

        t_filter = time.time()
        times['filter']= t_filter - t_detect

        # Draw boxes
        _draw_box(
            frame, final_boxes,
            [mc.CLASS_NAMES[idx]+': (%.2f)'% prob \
                for idx, prob in zip(final_class, final_probs)],
            cdict=mc.cls2clr
        )

        t_draw = time.time()
        times['draw']= t_draw - t_filter

        cv2.imwrite(out_im_name, frame)
        out.write(frame)

        times['total']= time.time() - t_start

        # time_str = ''
        # for t in times:
        #   time_str += '{} time: {:.4f} '.format(t[0], t[1])
        # time_str += '\n'
        time_str = 'Total time: {:.4f}, detection time: {:.4f}, filter time: '\
                   '{:.4f}'. \
            format(times['total'], times['detect'], times['filter'])

        print (time_str)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
  # Release everything if job is finished
  cap.release()
  out.release()
  cv2.destroyAllWindows()


def image_demo():
  """Detect image."""

  assert FLAGS.demo_net == 'squeezeDet' or FLAGS.demo_net == 'squeezeDet+', \
      'Selected nueral net architecture not supported: {}'.format(FLAGS.demo_net)

  with tf.Graph().as_default():
    # Load model
    if FLAGS.demo_net == 'squeezeDet':
      mc = kitti_squeezeDet_config()
      mc.BATCH_SIZE = 1
      # model parameters will be restored from checkpoint
      mc.LOAD_PRETRAINED_MODEL = False
      model = SqueezeDetIDX(mc, FLAGS.gpu)
    elif FLAGS.demo_net == 'squeezeDet+':
      mc = kitti_squeezeDetPlus_config()
      mc.BATCH_SIZE = 1
      mc.LOAD_PRETRAINED_MODEL = False
      model = SqueezeDetPlusIDX(mc, FLAGS.gpu)

    saver = tf.train.Saver(model.model_params)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
      saver.restore(sess, FLAGS.checkpoint)

      for f in glob.iglob(FLAGS.input_path):
        im = cv2.imread(f)
        im = im.astype(np.float32, copy=False)
        im = cv2.resize(im, (mc.IMAGE_WIDTH, mc.IMAGE_HEIGHT))
        input_image = im - mc.BGR_MEANS

        # Detect
        det_boxes, det_probs, det_class = sess.run(
            [model.det_boxes, model.det_probs, model.det_class],
            feed_dict={model.image_input:[input_image]})

        # Filter
        final_boxes, final_probs, final_class = model.filter_prediction(
            det_boxes[0], det_probs[0], det_class[0])

        keep_idx    = [idx for idx in range(len(final_probs)) \
                          if final_probs[idx] > mc.PLOT_PROB_THRESH]
        final_boxes = [final_boxes[idx] for idx in keep_idx]
        final_probs = [final_probs[idx] for idx in keep_idx]
        final_class = [final_class[idx] for idx in keep_idx]

        # Draw boxes
        _draw_box(
            im, final_boxes,
            [mc.CLASS_NAMES[idx]+': (%.2f)'% prob \
                for idx, prob in zip(final_class, final_probs)],
            cdict=mc.cls2clr
        )

        file_name = os.path.split(f)[1]
        out_file_name = os.path.join(FLAGS.out_dir, 'out_'+file_name)
        cv2.imwrite(out_file_name, im)
        print ('Image detection output saved to {}'.format(out_file_name))


def main(argv=None):
  if not tf.gfile.Exists(FLAGS.out_dir):
    tf.gfile.MakeDirs(FLAGS.out_dir)
  if FLAGS.mode == 'image':
    image_demo()
  else:
    video_demo()

if __name__ == '__main__':
    tf.app.run()
