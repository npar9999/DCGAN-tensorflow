import os
import time
from glob import glob
import tensorflow as tf

from ops import *
from utils import *
from input_pipeline_rendered_data import get_chair_pipeline_training

class DCGAN(object):
    def __init__(self, sess, image_size=108, 
                 batch_size=64, sample_size = 64, image_shape=[64, 64, 3],
                 y_dim=None, z_dim=100, gf_dim=64, df_dim=64,
                 gfc_dim=1024, dfc_dim=1024, c_dim=3, test_sketches=None, dataset_name='default'):
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
        self.image_size = image_size
        self.sample_size = sample_size
        self.image_shape = image_shape

        self.y_dim = y_dim
        self.z_dim = z_dim

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim

        self.c_dim = 3

        # batch normalization : deals with poor initialization helps gradient flow
        self.d_bn1 = batch_norm(batch_size, name='d_bn1')
        self.d_bn2 = batch_norm(batch_size, name='d_bn2')

        if not self.y_dim:
            self.d_bn3 = batch_norm(batch_size, name='d_bn3')

        self.g_bn0 = batch_norm(batch_size, name='g_bn0')
        self.g_bn1 = batch_norm(batch_size, name='g_bn1')
        self.g_bn2 = batch_norm(batch_size, name='g_bn2')

        self.g_s_bn1 = batch_norm(batch_size, name='g_s_bn1')
        self.g_s_bn2 = batch_norm(batch_size, name='g_s_bn2')
        self.g_s_bn3 = batch_norm(batch_size, name='g_s_bn3')

        if not self.y_dim:
            self.g_bn3 = batch_norm(batch_size, name='g_bn3')

        self.dataset_name = dataset_name
        self.build_model(test_sketches)

    def build_model(self, test_sketches=None):
        if self.y_dim:
            self.y= tf.placeholder(tf.float32, [None, self.y_dim], name='y')

        sketches, images = get_chair_pipeline_training(self.batch_size, 1000)
        self.images = images
        if test_sketches is None:
            self.sketches = sketches
        else:
            self.sketches = test_sketches

        with tf.variable_scope('generator') as scope:
            self.G = self.generator(self.sketches, 1)

        with tf.variable_scope('discriminator') as scope:
            self.D = self.discriminator(self.images, self.sketches)

            self.D_ = self.discriminator(self.G, self.sketches, reuse=True)

        with tf.variable_scope('discriminator_loss') as scope:
            self.d_loss_real = binary_cross_entropy_with_logits(tf.ones_like(self.D), self.D)
            self.d_loss_fake = binary_cross_entropy_with_logits(tf.zeros_like(self.D_), self.D_)
            self.d_loss = self.d_loss_real + self.d_loss_fake

        with tf.variable_scope('generator_loss') as scope:
            self.g_loss = binary_cross_entropy_with_logits(tf.ones_like(self.D_), self.D_)


        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.make_summary_ops()
        self.saver = tf.train.Saver(self.d_vars + self.g_vars, max_to_keep=0)

    def train(self, config):
        """Train DCGAN"""

        d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
                          .minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
                          .minimize(self.g_loss, var_list=self.g_vars)
        tf.initialize_all_variables().run()
        counter = 1
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
                _, _, errD_fake, errD_real, errG =  self.sess.run([d_optim, g_optim, self.d_loss_fake,
                                                                   self.d_loss_real, self.g_loss])
                toc = time.time()
                # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                #self.sess.run(g_optim)

                counter += 1
                print("Step: [%4d] time: %4f, last iter: %1.4f, d_loss: %.8f, g_loss: %.8f" \
                    % (counter, toc - start_time, toc - tic, errD_fake+errD_real, errG))
                if counter % 50 == 0:
                    summary_str = self.sess.run(summary_op)
                    summary_writer.add_summary(summary_str, counter)


                if np.mod(counter, 500) == 2:
                    self.save(config.checkpoint_dir, counter)
                    samples, sample_sketches = self.sess.run([self.G, self.sketches])
                    save_images(samples, [8, 8], os.path.join(config.summary_dir, 'train_%s.png' % counter))
                    save_images(sample_sketches, [8, 8], os.path.join(config.summary_dir, 'train_%s_sketches.png' % counter))


        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()

    def discriminator(self, image, sketches, reuse=False, y=None):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        if not self.y_dim:
            concated = tf.concat(3, [image, sketches])
            h0 = lrelu(conv2d(concated, self.df_dim, name='d_h0_conv'))
            h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
            h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv')))
            h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, name='d_h3_conv')))
            h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h3_lin')

            return tf.nn.sigmoid(h4)
        else:
            yb = tf.reshape(y, [None, 1, 1, self.y_dim])
            x = conv_cond_concat(image, yb)

            h0 = lrelu(spatial_conv(x, self.c_dim + self.y_dim))
            h0 = conv_cond_concat(h0, yb)

            h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim + self.y_dim)))
            h1 = tf.reshape(h1, [h1.get_shape()[0], -1])
            h1 = tf.concat(1, [h1, y])

            h2 = lrelu(self.d_bn2(linear(h1, self.dfc_dim, 'd_h2_lin')))
            h2 = tf.concat(1, [h2, y])

            return tf.nn.sigmoid(linear(h2, 1, 'd_h3_lin'))

    def generator(self, sketches, output_dimensions=3, y=None):
        s0 = lrelu(conv2d(sketches, self.df_dim, name='g_s0_conv'))
        s1 = lrelu(self.g_s_bn1(conv2d(s0, self.df_dim * 2, name='g_s1_conv')))
        s2 = lrelu(self.g_s_bn2(conv2d(s1, self.df_dim * 4, name='g_s2_conv')))
        s3 = lrelu(self.g_s_bn3(conv2d(s2, self.df_dim * 8, name='g_s3_conv')))
        self.abstract_representation = s3

        if not self.y_dim:
            # project `z` and reshape
            # h0 = tf.reshape(linear(s3, self.gf_dim*8*4*4, 'g_h0_lin'),
            #                 [-1, 4, 4, self.gf_dim * 8])
            # h0 = tf.nn.relu(self.g_bn0(h0))

            h1 = deconv2d(s3, [self.batch_size, 8, 8, self.gf_dim*4], name='g_h1')
            h1 = tf.nn.relu(self.g_bn1(h1))

            h2 = deconv2d(h1, [self.batch_size, 16, 16, self.gf_dim*2], name='g_h2')
            h2 = tf.nn.relu(self.g_bn2(h2))

            h3 = deconv2d(h2, [self.batch_size, 32, 32, self.gf_dim*1], name='g_h3')
            h3 = tf.nn.relu(self.g_bn3(h3))

            h4 = deconv2d(h3, [self.batch_size, 64, 64, output_dimensions], name='g_h4')

            return tf.nn.tanh(h4)
        else:
            yb = tf.reshape(y, [None, 1, 1, self.y_dim])
            z = tf.concat(1, [z, y])

            h0 = tf.nn.relu(self.g_bn0(linear(z, self.gfc_dim, 'g_h0_lin')))
            h0 = tf.concat(1, [h0, y])

            h1 = tf.nn.relu(self.g_bn1(linear(z, self.gf_dim*2*7*7, 'g_h1_lin')))
            h1 = tf.reshape(h1, [None, 7, 7, self.gf_dim * 2])
            h1 = conv_cond_concat(h1, yb)

            h2 = tf.nn.relu(self.g_bn2(deconv2d(h1, self.gf_dim, name='g_h2')))
            h2 = conv_cond_concat(h2, yb)

            return tf.nn.sigmoid(deconv2d(h2, self.c_dim, name='g_h3'))

    def make_summary_ops(self):
        tf.image_summary('generator', self.G)
        tf.image_summary('sketch', self.sketches)
        tf.image_summary('images', self.images)
        tf.scalar_summary('d_loss_fake', self.d_loss_fake)
        tf.scalar_summary('d_loss_real', self.d_loss_real)
        tf.scalar_summary('g_loss', self.g_loss)
        tf.scalar_summary('d_loss', self.d_loss)
        tf.histogram_summary('abstract_representation', self.abstract_representation)


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
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
        else:
            raise Exception(" [!] Testing, but %s not found" % checkpoint_dir)
