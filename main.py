from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

from model import DCGAN
from utils import pp

flags = tf.app.flags
flags.DEFINE_integer("epoch", 25, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.0001, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_string("checkpoint_dir", "checkpoint_sketches_to_rendered", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("summary_dir", "summary_sketches_to_rendered", "Directory name to save the summaries [checkpoint]")
flags.DEFINE_string("continue_from", None, 'Continues from the given run, None does start training from scratch [None]')
flags.DEFINE_integer("continue_from_iteration", None, 'Continues from the given iteration (of the given run), '
                                                     'None does restore the most current iteration [None]')
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
    with tf.Session(config=tf.ConfigProto()) as sess:
        dcgan = DCGAN(sess, batch_size=FLAGS.batch_size)
        if not os.path.exists(FLAGS.checkpoint_dir):
            os.makedirs(FLAGS.checkpoint_dir)
        if not os.path.exists(FLAGS.summary_dir):
            os.makedirs(FLAGS.summary_dir)

        dcgan.train(FLAGS, run_folder)

if __name__ == '__main__':
    tf.app.run()
