import pygame
import numpy as np
import tensorflow as tf
from model import DCGAN
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import matplotlib.cm as cm
import os, threading, time
from input_pipeline_rendered_data import preprocess

flags = tf.app.flags

flags.DEFINE_string("checkpoint_dir", "checkpoint_sketches_to_rendered", "Directory name to restore the checkpoints from")
flags.DEFINE_string("continue_from", None, 'Continues from the given run, None does restore the most current run [None]')
flags.DEFINE_string("continue_from_iteration", None, 'Continues from the given iteration (of the given run), '
                                                     'None does restore the most current iteration [None]')
FLAGS = flags.FLAGS

class MouseButtons:
    LEFT = 1
    MIDDLE = 2
    RIGHT = 3
    WHEEL_UP = 4
    WHEEL_DOWN = 5


class SketchScreen:

    def __init__(self):
        self.draw_on = False
        self.last_pos = (0, 0)
        self.radius = 5
        self.screen = pygame.display.set_mode((512,512))

    def roundline(self, srf, color, start, end, radius=1):
        dx = end[0]-start[0]
        dy = end[1]-start[1]
        distance = max(abs(dx), abs(dy))
        for i in range(distance):
            x = int(start[0]+float(i)/distance*dx)
            y = int(start[1]+float(i)/distance*dy)
            pygame.draw.circle(srf, color, (x, y), radius)


    def brush_at(self, pos, do_roundline=False):
        # color = screen.get_at(pos)
        # color += pygame.Color(strength, strength, strength)
        color = pygame.Color(self.strength, self.strength, self.strength)
        pygame.draw.circle(self.screen, color, pos, self.radius)
        if do_roundline:
            self.roundline(self.screen, color, pos, self.last_pos,  self.radius)

    def get_content_as_np_array(self):
        return pygame.surfarray.array3d(self.screen)[:, : ,0] / (32.0 * 8)


    def enter_loop(self):
        try:
            while True:
                e = pygame.event.wait()
                if e.type == pygame.QUIT:
                    raise StopIteration
                if e.type == pygame.MOUSEBUTTONDOWN:
                    if e.button == MouseButtons.LEFT:
                        self.strength = 50
                    elif e.button == MouseButtons.RIGHT:
                        self.strength = 0

                    self.brush_at(e.pos)
                    self.draw_on = True
                if e.type == pygame.MOUSEBUTTONUP:
                    self.draw_on = False
                if e.type == pygame.MOUSEMOTION:
                    if self.draw_on:
                        self.brush_at(e.pos, True)
                    self.last_pos = e.pos
                pygame.display.flip()


        except StopIteration:
            pass

        pygame.quit()


class OutputScreen:
    def __init__(self, size):
        data = np.zeros([size, size, 3])

        fig, (ax_input, ax_output) = plt.subplots(2, 1)
        my_extent = [-100, 100, -100, 100]
        self.imshow_window = ax_output.imshow(data, interpolation='nearest', aspect='equal')
        self.downsampled_input = ax_input.imshow(data, interpolation='nearest', aspect='equal',
                                                 cmap=cm.Greys_r)

        self.sliders = []
        for i in xrange(4):
            ax = fig.add_axes([0.2, 0.03 * i, 0.65, 0.03])
            s = Slider(ax, 'Random ' + str(i), -1, 1, valinit=0)
            self.sliders.append(s)

        fig.show()

    def update_content(self, input, output):
        self.downsampled_input.set_data(np.transpose(input, [1, 0]))
        self.imshow_window.set_data(np.transpose(output, [1, 0, 2]))
        plt.draw()


def main(_):
    runs = sorted(map(int, next(os.walk(FLAGS.checkpoint_dir))[1]))
    if FLAGS.continue_from:
        run_folder = FLAGS.continue_from
    else:
        run_folder = str(runs[-1]).zfill(3)

    used_checkpoint_dir = os.path.join(FLAGS.checkpoint_dir, run_folder)
    print('Restoring from ' + FLAGS.checkpoint_dir)


    sc = SketchScreen()

    draw_thread = threading.Thread(target=sc.enter_loop)
    draw_thread.start()

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.01)
    upscaled_size = 256


    output_screen = OutputScreen(upscaled_size)

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        with tf.device('/cpu:0'):
            dcgan = DCGAN(sess, batch_size=1, is_train=False)
            full_sketch = tf.placeholder(tf.float32, [512, 512])
            small_sketch = tf.image.resize_bilinear(tf.reshape(full_sketch, [1, 512, 512, 1]), [64, 64])
            small_sketch = preprocess(small_sketch, 64, whiten='sketch', color=False, augment=False)
            small_sketch = tf.reshape(small_sketch, [1, 64, 64, 1])

        tf.initialize_all_variables().run()
        dcgan.load(used_checkpoint_dir, FLAGS.continue_from_iteration)

        while draw_thread.is_alive:
            try:
                s = sess.run(small_sketch, feed_dict={full_sketch: sc.get_content_as_np_array()})
                unnormed_small_sketch = (np.reshape(s, [64, 64]) + 1) / 2

                z = np.reshape(np.asarray([slider.val for slider in output_screen.sliders], dtype=np.float32), [1, 4])
                img = sess.run(dcgan.G, feed_dict={dcgan.z: z, dcgan.sketches: s})

                unnormed_img = (np.reshape(img, [64, 64, 3]) + 1) / 2
                output_screen.update_content(unnormed_small_sketch, unnormed_img)
                print('Updated image')
                plt.pause(0.5)
            except pygame.error:
                print('Pygame stoped, shutting down.')
                break

if __name__ == '__main__':
    tf.app.run()
