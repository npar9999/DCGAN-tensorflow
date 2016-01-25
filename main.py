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
flags.DEFINE_integer("epoch", 25, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_string("checkpoint_dir", "checkpoint_sketches_to_rendered", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("summary_dir", "summary_sketches_to_rendered", "Directory name to save the summaries [checkpoint]")
flags.DEFINE_boolean("is_train", True, "True for training, False for testing [False]")
flags.DEFINE_string("continue_from", None, 'Continues from the given run, None does start training from scratch [None]')
FLAGS = flags.FLAGS

def main(_):
    pp.pprint(flags.FLAGS.__flags)

    if not os.path.exists(FLAGS.summary_dir):
        os.makedirs(FLAGS.summary_dir)
    runs = sorted(map(int, next(os.walk(FLAGS.summary_dir))[1]))
    if len(runs) == 0:
        run_nr = 0
    else:
        run_nr = runs[-1] + 1
    run_folder = str(run_nr).zfill(3)

    FLAGS.summary_dir = os.path.join(FLAGS.summary_dir, run_folder)
    FLAGS.checkpoint_dir = os.path.join(FLAGS.checkpoint_dir, run_folder)
    if FLAGS.is_train:
        gpu_options= tf.GPUOptions()
    else:
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.01)
    with tf.Session(config=tf.ConfigProto(log_device_placement=True, gpu_options=gpu_options)) as sess:
        if FLAGS.is_train:
            dcgan = DCGAN(sess, batch_size=FLAGS.batch_size)
            if not os.path.exists(FLAGS.checkpoint_dir):
                os.makedirs(FLAGS.checkpoint_dir)
            if not os.path.exists(FLAGS.summary_dir):
                os.makedirs(FLAGS.summary_dir)

            dcgan.train(FLAGS, run_folder)
        else:
            test_files = sorted(glob.glob('test_sketches/*.png'))
            FLAGS.batch_size = len(test_files)
            with tf.device('/cpu:0'):
                test_sketch_producer = make_image_producer(test_files, 1, 'test_sketches', 64,
                                                           shuffle=False, whiten='sketch', color=False, augment=False)
                test_sketches = tf.train.batch([test_sketch_producer], batch_size=FLAGS.batch_size)

                dcgan = DCGAN(sess, batch_size=FLAGS.batch_size, is_train=False)
                if dcgan.G.get_shape()[3] == 3:
                    sketches_for_display = tf.concat(3, [dcgan.sketches, dcgan.sketches, dcgan.sketches])
                else:
                    sketches_for_display = dcgan.sketches
                # Put it together with sketch again for easy comparison
                if FLAGS.batch_size == 1:
                    sample_with_sketch = tf.concat(0, [dcgan.G, sketches_for_display])

            run_restored = FLAGS.continue_from
            used_checkpoint_dir = os.path.join(os.path.dirname(FLAGS.checkpoint_dir), run_restored)

            # Important: Since not all variables are restored, some need to be initialized here.
            tf.initialize_all_variables().run()
            dcgan.load(used_checkpoint_dir)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            try:
                num_versions = 64
                batch_z_shape = [num_versions, FLAGS.batch_size, dcgan.z_dim]
                np.random.seed(42)
                batch_z = np.random.uniform(-1, 1, [num_versions, 1, dcgan.z_dim])

                # Every img in one batch should share the same random vector, use numpy broadcasting here to achieve that.
                batch_z_all = np.zeros(batch_z_shape)
                batch_z_all[:, :, :] = batch_z

                batch_sketches = test_sketches.eval()
                one_chair_different_randoms = np.zeros([FLAGS.batch_size, num_versions, dcgan.image_size, dcgan.image_size, 3])
                for i in xrange(num_versions):
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

                for j in xrange(FLAGS.batch_size):
                    grid_size = np.ceil(np.sqrt(num_versions))
                    save_images(one_chair_different_randoms[j, :, :, :, :], [grid_size, grid_size],
                                'test_sketches_to_rendered_out/chair_{}_different_randoms.png'.format(str(j).zfill(3)))

            except tf.errors.OutOfRangeError as e:
                print('Done')
            finally:
                # When done, ask the threads to stop.
                coord.request_stop()
                # And wait for them to actually do it.
                coord.join(threads)


if __name__ == '__main__':
    tf.app.run()
