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


def activations_to_images(V):
    activation_channels = int(V.get_shape()[3])
    min = tf.reduce_min(V)
    max = tf.reduce_max(V)
    V = (V - min) / (max - min)
    V = tf.pad(V, [[0, 0], [1, 1], [1, 1], [0, 0]])
    V = tf.transpose(V, (0, 3, 1, 2))
    return V, activation_channels


def main(_):
    pp.pprint(flags.FLAGS.__flags)

    runs = sorted(map(int, next(os.walk(FLAGS.checkpoint_dir))[1]))
    if FLAGS.continue_from:
        run_folder = FLAGS.continue_from
    else:
        run_folder = str(runs[-1]).zfill(3)

    used_checkpoint_dir = os.path.join(FLAGS.checkpoint_dir, run_folder)
    print('Restoring from ' + FLAGS.checkpoint_dir)


    with tf.Session(config=tf.ConfigProto(device_count={'GPU': 1})) as sess:
          test_files = sorted(glob.glob('test_sketches/*.png'))
          FLAGS.batch_size = len(test_files)
          test_sketch_producer = make_image_producer(test_files, 1, 'test_sketches', 64,
                                                     shuffle=False, whiten='sketch', color=False, augment=False)
          test_sketches = tf.train.batch([test_sketch_producer], batch_size=FLAGS.batch_size)

          dcgan = DCGAN(sess, batch_size=FLAGS.batch_size, is_train=False)

          # Define tensor for visualizing abstract representation.
          Vs = [activations_to_images(x) for x in [dcgan.s0, dcgan.s1, dcgan.s2, dcgan.abstract_representation]]

          # Important: Since not all variables are restored, some need to be initialized here.
          tf.initialize_all_variables().run()
          loaded_iteration_string = dcgan.load(used_checkpoint_dir, FLAGS.continue_from_iteration)

          output_folder = os.path.join(FLAGS.checkpoint_dir, run_folder, 'test_images', loaded_iteration_string)
          if not os.path.exists(output_folder):
              os.makedirs(output_folder)
          print('Writing output to ' + output_folder + '/*.png')

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
              grid_size = np.ceil(np.sqrt(FLAGS.batch_size))
              save_images(batch_sketches, [grid_size, grid_size], os.path.join(output_folder, 'sketches.png'))

              one_chair_different_randoms = np.zeros([FLAGS.batch_size, FLAGS.num_samples,
                                                      dcgan.image_size, dcgan.image_size, 3])

              for i in xrange(FLAGS.num_samples):
                  batch_z = batch_z_all[i, :, :]
                  img = sess.run(dcgan.G, feed_dict={dcgan.z: batch_z,
                                                     dcgan.sketches: batch_sketches})
                  filename_out = os.path.join(output_folder, '{}_img.png'.format(str(i).zfill(3)))
                  save_images(img, [grid_size, grid_size], filename_out)
                  for j in xrange(FLAGS.batch_size):
                      one_chair_different_randoms[j, i, :, :, :] = img[j, :, :, :]
                  if i == 0:
                      activations = sess.run([dcgan.abstract_representation] + [x for x, _ in Vs],
                                             feed_dict={dcgan.z: batch_z, dcgan.sketches: batch_sketches})
                      # Unpack correctly.
                      abstract_rep = activations[0]
                      activations = activations[1:]


              for j, file_name in enumerate(test_files):
                  grid_size = np.ceil(np.sqrt(FLAGS.num_samples))
                  name_without_ext = os.path.splitext(os.path.basename(file_name))[0]
                  filename_out = os.path.join(output_folder, '{}_different_randoms.png'.format(name_without_ext))
                  save_images(one_chair_different_randoms[j, :, :, :, :], [grid_size, grid_size], filename_out)

                  # Visualize abstract representation.
                  for idx, (_, channel_count) in enumerate(Vs):
                      grid_size = np.ceil(np.sqrt(channel_count))
                      filename_out = os.path.join(output_folder, '{}_layer_{}.png'.format(name_without_ext, idx))
                      save_images(activations[idx][j, :, :, :], [grid_size, grid_size], filename_out,
                                  invert=False, channels=1)

              for k in [1, 5, 10, 20, 50, 100, 200, 300]:
                abstract_rep_hacked = np.copy(abstract_rep)
                for j, file_name in enumerate(test_files):
                    # Find highest non-z activation (last few slices are random parameters only)
                    sum_activations = np.sum(abstract_rep[j, :, :, 0:-dcgan.z_dim], axis=(0, 1))
                    strongest_k_activations = np.argpartition(sum_activations, -k)[-k:]
                    # Set strongest activation to zero.
                    abstract_rep_hacked[j, :, :, strongest_k_activations] = 0

                grid_size = np.ceil(np.sqrt(FLAGS.batch_size))
                img = sess.run(dcgan.G_abstract, feed_dict={dcgan.abstract: abstract_rep_hacked})
                filename_out = os.path.join(output_folder,
                                            '000_img_without_highest_{}_activation.png'.format(str(k).zfill(3)))
                save_images(img, [grid_size, grid_size], filename_out)

          except tf.errors.OutOfRangeError as e:
              print('Done')
          finally:
              # When done, ask the threads to stop.
              coord.request_stop()
              # And wait for them to actually do it.
              coord.join(threads)


if __name__ == '__main__':
    tf.app.run()
