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
flags.DEFINE_integer("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("image_size", 64, "The size of image to use (will be center cropped) [108]")
flags.DEFINE_string("checkpoint_dir", "checkpoint_sketches", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("summary_dir", "summary_sketches", "Directory name to save the summaries [checkpoint]")
flags.DEFINE_boolean("is_train", True, "True for training, False for testing [False]")
FLAGS = flags.FLAGS

def main(_):
    pp.pprint(flags.FLAGS.__flags)

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
            dcgan = DCGAN(sess, image_size=FLAGS.image_size, batch_size=FLAGS.batch_size)
            if not os.path.exists(FLAGS.checkpoint_dir):
                os.makedirs(FLAGS.checkpoint_dir)
            if not os.path.exists(FLAGS.summary_dir):
                os.makedirs(FLAGS.summary_dir)

            dcgan.train(FLAGS)
        else:
            test_files = glob.glob('test_sketches/*64x64.png')
            FLAGS.batch_size = 1    
            with tf.device('/cpu:0'):
                test_sketch_producer = make_image_producer(test_files, 1, 'test_sketches', 64,
                                                          shuffle=False, whiten=False, color=False, capacity=1)
                test_sketches = tf.train.batch([test_sketch_producer], batch_size=FLAGS.batch_size)

                dcgan = DCGAN(sess, image_size=FLAGS.image_size, batch_size=FLAGS.batch_size, test_sketches=test_sketches)

            used_checkpoint_dir = os.path.join(os.path.dirname(FLAGS.checkpoint_dir), '023') 
            # Important: Since not all variables are restored, some need to be initialized here.
            tf.initialize_all_variables().run()
            dcgan.load(used_checkpoint_dir)
            coord = tf.train.Coordinator()
            tf.train.start_queue_runners(sess=sess, coord=coord)
            try:
                for filename in test_files:
                    samples = sess.run(dcgan.G)
                    save_images(samples, [1, 1], '{}_depth.png'.format(filename))
            except tf.errors.OutOfRangeError as e:
                print('Done training -- epoch limit reached')
                raise e
            finally:
                # When done, ask the threads to stop.
                coord.request_stop()


if __name__ == '__main__':
    tf.app.run()
