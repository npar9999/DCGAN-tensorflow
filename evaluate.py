from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
import glob

from model import DCGAN
from utils import pp, save_images
from input_pipeline_rendered_data import make_image_producer

flags = tf.app.flags
flags.DEFINE_string("checkpoint_dir", "checkpoint_sketches_to_rendered", "Directory name to restore the checkpoints from")
flags.DEFINE_string("continue_from", None, 'Continues from the given run, None does restore the most current run [None]')
flags.DEFINE_string("continue_from_iteration", None, 'Continues from the given iteration (of the given run), '
                                                     'None does restore the most current iteration [None]')
flags.DEFINE_integer("random_seed", 42, 'Seed for random vector z [42]')
flags.DEFINE_integer("num_samples", 64, 'Number of different samples to produce for every sketch [64]')
flags.DEFINE_string("test_images_folder", 'test_images_folder', 'Folder to pull test images from (all files with .png extension will be processed).')
FLAGS = flags.FLAGS

def main(_):
    pp.pprint(flags.FLAGS.__flags)

    runs = sorted(map(int, next(os.walk(FLAGS.checkpoint_dir))[1]))
    if FLAGS.continue_from:
        run_folder = FLAGS.continue_from
    else:
        run_folder = str(runs[-1]).zfill(3)

    used_checkpoint_dir = os.path.join(FLAGS.checkpoint_dir, run_folder)
    print('Restoring from ' + FLAGS.checkpoint_dir)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.01)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
          test_files = sorted(glob.glob('test_sketches/*.png'))
          FLAGS.batch_size = len(test_files)
          with tf.device('/cpu:0'):
              test_sketch_producer = make_image_producer(test_files, 1, 'test_sketches', 64,
                                                         shuffle=False, whiten='sketch', color=False, augment=False)
              test_sketches = tf.train.batch([test_sketch_producer], batch_size=FLAGS.batch_size)

              dcgan = DCGAN(sess, batch_size=FLAGS.batch_size, is_train=False)

          # Important: Since not all variables are restored, some need to be initialized here.
          tf.initialize_all_variables().run()
          dcgan.load(used_checkpoint_dir, FLAGS.continue_from_iteration)

          coord = tf.train.Coordinator()
          threads = tf.train.start_queue_runners(sess=sess, coord=coord)
          try:
              np.random.seed(FLAGS.random_seed)
              batch_z = np.random.uniform(-1, 1, [FLAGS.num_samples, 1, dcgan.z_dim])

              # Every img in one batch should share the same random vector, use numpy broadcasting here to achieve that.
              batch_z_shape = [FLAGS.num_samples, FLAGS.batch_size, dcgan.z_dim]
              batch_z_all = np.zeros(batch_z_shape)
              batch_z_all[:, :, :] = batch_z

              batch_sketches = test_sketches.eval()
              one_chair_different_randoms = np.zeros([FLAGS.batch_size, FLAGS.num_samples, dcgan.image_size, dcgan.image_size, 3])
              for i in xrange(FLAGS.num_samples):
                  batch_z = batch_z_all[i, :, :]
                  img = sess.run(dcgan.G,
                                 feed_dict={dcgan.z: batch_z,
                                            dcgan.sketches: batch_sketches})
                  filename_out = 'test_sketches_to_rendered_out/{}_img.png'.format(str(i).zfill(3))
                  grid_size = np.ceil(np.sqrt(FLAGS.batch_size))
                  save_images(img, [grid_size, grid_size], filename_out)
                  for j in xrange(FLAGS.batch_size):
                      one_chair_different_randoms[j, i, :, :, :] = img[j, :, :, :]
                  filename_out = 'test_sketches_to_rendered_out/{}_sketch.png'.format(str(i).zfill(3))
                  save_images(batch_sketches, [grid_size, grid_size], filename_out)

              for j, file_name in enumerate(test_files):
                  grid_size = np.ceil(np.sqrt(FLAGS.num_samples))
                  name_without_ext = os.path.splitext(os.path.basename(file_name))[0]
                  save_images(one_chair_different_randoms[j, :, :, :, :], [grid_size, grid_size],
                              'test_sketches_to_rendered_out/{}_different_randoms.png'.format(name_without_ext))

          except tf.errors.OutOfRangeError as e:
              print('Done')
          finally:
              # When done, ask the threads to stop.
              coord.request_stop()
              # And wait for them to actually do it.
              coord.join(threads)


if __name__ == '__main__':
    tf.app.run()
