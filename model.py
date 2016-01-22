import os
import time
import tensorflow as tf

from ops import *
from utils import *
from input_pipeline_rendered_data import get_chair_pipeline_training_from_dump

class DCGAN(object):
    def __init__(self, sess,
                 batch_size=64, sample_size = 64, image_shape=[64, 64, 3],
                 y_dim=None, z_dim=16, gf_dim=64, df_dim=64,
                 gfc_dim=512, dfc_dim=1024, c_dim=3, is_train=True):
        """

        Args:
            sess: TensorFlow session
            batch_size: The size of batch. Should be specified before training.
            y_dim: (optional) Dimension of dim for y. [None]
            z_dim: (optional) Dimension of dim for Z. [100]
            gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
            df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
            gfc_dim: (optional) Dimension of gen untis for for fully connected layer. [1024]
            dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
            c_dim: (optional) Dimension of image color. [3]
        """
        self.sess = sess
        self.batch_size = batch_size
        self.sample_size = sample_size
        self.image_shape = image_shape

        self.y_dim = y_dim
        self.z_dim = z_dim

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim

        self.c_dim = c_dim

        # batch normalization : deals with poor initialization helps gradient flow
        self.d_bn1 = batch_norm(is_train, name='d_bn1')
        self.d_bn2 = batch_norm(is_train, name='d_bn2')
        self.d_bn3 = batch_norm(is_train, name='d_bn3')

        self.g_bn0 = batch_norm(is_train, name='g_bn0')
        self.g_bn1 = batch_norm(is_train, name='g_bn1')
        self.g_bn2 = batch_norm(is_train, name='g_bn2')
        self.g_bn3 = batch_norm(is_train, name='g_bn3')

        self.g_s_bn1 = batch_norm(is_train, name='g_s_bn1')
        self.g_s_bn2 = batch_norm(is_train, name='g_s_bn2')
        self.g_s_bn3 = batch_norm(is_train, name='g_s_bn3')
        self.g_s_bn4 = batch_norm(is_train, convolutional=False, name='g_s_bn4')

        self.build_model(is_train)

    def build_model(self, is_train):
        if self.y_dim:
            self.y= tf.placeholder(tf.float32, [None, self.y_dim], name='y')

        self.image_size = 64
        sketches, images = get_chair_pipeline_training_from_dump('recolor_chairs.tfrecords', self.batch_size,
                                                                 10000, image_size=self.image_size )
        self.images = images
        if is_train:
            self.sketches = sketches
            if self.z_dim:
                self.z = tf.random_uniform([self.batch_size, self.z_dim], minval=-1, maxval=1, dtype=tf.float32)
        else:
            self.sketches = tf.placeholder(tf.float32, [None, self.sample_size,
                                                        self.sample_size, 1])
            if self.z_dim:
                self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')

        with tf.variable_scope('generator') as scope:
            self.G = self.generator(self.sketches, self.z)

        with tf.variable_scope('discriminator') as scope:
            self.D = self.discriminator(self.images, self.sketches)

            self.D_ = self.discriminator(self.G, self.sketches, reuse=True)

        with tf.variable_scope('discriminator_loss') as scope:
            self.d_loss_real = binary_cross_entropy_with_logits(tf.ones_like(self.D), self.D)
            self.d_loss_fake = binary_cross_entropy_with_logits(tf.zeros_like(self.D_), self.D_)
            self.d_loss = self.d_loss_real + self.d_loss_fake

        with tf.variable_scope('generator_loss') as scope:
            self.g_loss = binary_cross_entropy_with_logits(tf.ones_like(self.D_), self.D_)

        self.bn_assigners = tf.group(*batch_norm.assigners)

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.make_summary_ops()
        self.saver = tf.train.Saver(self.d_vars + self.g_vars +
                                    batch_norm.shadow_variables,
                                    max_to_keep=0)


    def train(self, config, run_string="???"):
        """Train DCGAN"""

        d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
                          .minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
                          .minimize(self.g_loss, var_list=self.g_vars)

        # See that moving average is also updated with g_optim.
        with tf.control_dependencies([g_optim]):
            g_optim = tf.group(self.bn_assigners)

        tf.initialize_all_variables().run()
        if config.continue_from:
            checkpoint_dir = os.path.join(os.path.dirname(config.checkpoint_dir), config.continue_from)
            print('Loading variables from ' + checkpoint_dir)
            self.load(checkpoint_dir)

        start_time = time.time()

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)
        summary_op = tf.merge_all_summaries()
        summary_writer = tf.train.SummaryWriter(config.summary_dir, graph_def=self.sess.graph_def)

        try:
            # Training
            counter = 0
            while not coord.should_stop():
                # Update D and G network
                tic = time.time()

                _, _, errD_fake, errD_real, errG = self.sess.run([d_optim, g_optim, self.d_loss_fake,
                                                                  self.d_loss_real, self.g_loss])
                # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                self.sess.run(g_optim)
                toc = time.time()

                counter += 1
                duration = toc - tic
                print("Run: %s, Step: [%4d] time: %5.1f, last iter: %1.2f (%1.4f e/s), d_loss: %.8f, g_loss: %.8f"
                    % (run_string, counter, toc - start_time, duration, self.batch_size / duration, errD_fake+errD_real, errG))
                if counter % 50 == 0:
                    summary_str = self.sess.run(summary_op)
                    summary_writer.add_summary(summary_str, counter)

                if np.mod(counter, 500) == 2:
                    samples, sample_images, sample_sketches = self.sess.run([self.G, self.images, self.sketches])
                    save_images(samples, [8, 8], os.path.join(config.summary_dir, 'train_%s.png' % counter))
                    save_images(sample_images, [8, 8], os.path.join(config.summary_dir, 'train_%s_images.png' % counter))
                    save_images(sample_sketches, [8, 8], os.path.join(config.summary_dir, 'train_%s_sketches.png' % counter))

                if np.mod(counter, 2000) == 100:
                    self.save(config.checkpoint_dir, counter)


        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()
            coord.join(threads)

    def discriminator(self, image, sketches, reuse=False, y=None):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        concated = tf.concat(3, [image, sketches])
        self.d_h0 = lrelu(conv2d(concated, self.df_dim, name='d_h0_conv'))
        h1 = lrelu(self.d_bn1(conv2d(self.d_h0, self.df_dim*2, name='d_h1_conv')))
        h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv')))
        h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, name='d_h3_conv')))
        h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h3_lin')

        return tf.nn.sigmoid(h4)

    def generator(self, sketches, z=None, y=None):
        s0 = lrelu(conv2d(sketches, self.df_dim, name='g_s0_conv'))
        s1 = lrelu(self.g_s_bn1(conv2d(s0, self.df_dim * 2, name='g_s1_conv')))
        s2 = lrelu(self.g_s_bn2(conv2d(s1, self.df_dim * 4, name='g_s2_conv')))
        s3 = lrelu(self.g_s_bn3(conv2d(s2, self.df_dim * 8, name='g_s3_conv')))
        # Size after 4 convolutions with stride 2.
        downsampled_size = self.image_size // 2 ** 4

        # s3_flat = tf.reshape(s3, [self.batch_size, self.gf_dim*8*downsampled_size*downsampled_size])
        # self.abstract_representation = lrelu(self.g_s_bn4(linear(s3_flat, self.gfc_dim, 'g_s4_lin')))
        # if z:
        #     self.abstract_representation = tf.concat(1, [self.abstract_representation, z])
        #
        # # project `abstract representation` and reshape
        # h0 = tf.reshape(linear(self.abstract_representation,
        #                        self.gf_dim*8*downsampled_size*downsampled_size, 'g_h0_lin'),
        #                 [-1, downsampled_size, downsampled_size, self.gf_dim * 8])
        # h0 = tf.nn.relu(self.g_bn0(h0))

        z_slices = tf.mul(tf.ones([self.batch_size, downsampled_size, downsampled_size, self.z_dim]),
                          tf.reshape(self.z, [self.batch_size, 1, 1, self.z_dim]))
        self.abstract_representation = tf.concat(3, [s3, z_slices])

        h1 = deconv2d(self.abstract_representation, [self.batch_size, downsampled_size * 2,
                                                     downsampled_size * 2, self.gf_dim*4 + self.z_dim],
                      name='g_h1')
        h1 = tf.nn.relu(self.g_bn1(h1))

        h2 = deconv2d(h1, [self.batch_size, downsampled_size * 4, downsampled_size * 4, self.gf_dim*2], name='g_h2')
        h2 = tf.nn.relu(self.g_bn2(h2))

        h3 = deconv2d(h2, [self.batch_size, downsampled_size * 8, downsampled_size * 8, self.gf_dim*1], name='g_h3')
        h3 = tf.nn.relu(self.g_bn3(h3))

        h4 = deconv2d(h3, [self.batch_size, downsampled_size * 16, downsampled_size * 16, self.c_dim], name='g_h4')

        return tf.nn.tanh(h4)


    def make_summary_ops(self):
        tf.image_summary('generator', self.G)
        tf.image_summary('sketch', self.sketches)
        tf.image_summary('images', self.images)
        tf.scalar_summary('d_loss_fake', self.d_loss_fake)
        tf.scalar_summary('d_loss_real', self.d_loss_real)
        tf.scalar_summary('g_loss', self.g_loss)
        tf.scalar_summary('d_loss', self.d_loss)
        tf.histogram_summary('abstract_representation', self.abstract_representation)
        tf.histogram_summary('d_h0', self.d_h0)
        if self.z_dim:
            with tf.variable_scope('z_stats') as scope:
                length = tf.sqrt(tf.reduce_sum(tf.square(self.z)))
                tf.scalar_summary('length_z', length)

    def save(self, checkpoint_dir, step):
        model_name = "DCGAN.model"

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            ckpt_file = os.path.join(checkpoint_dir, ckpt_name)
            print('Reading variables to be restored from ' + ckpt_file)
            self.saver.restore(self.sess, ckpt_file)
        else:
            raise Exception(" [!] Testing, but %s not found" % checkpoint_dir)
