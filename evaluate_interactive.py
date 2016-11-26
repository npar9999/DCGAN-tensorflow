from __future__ import division

import pygame
from pygame.locals import *
import numpy as np
import tensorflow as tf
from model import DCGAN
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import os, threading, time
from input_pipeline_rendered_data import preprocess, make_image_producer
import scipy.misc, random, glob, re

flags = tf.app.flags

flags.DEFINE_string("checkpoint_dir", "checkpoint_sketches_to_rendered",
                    "Directory name to restore the checkpoints from")
flags.DEFINE_string("continue_from", '039',
                    'Continues from the given run, None does restore the most current run [None]')
flags.DEFINE_string("continue_from_iteration", None, 'Continues from the given iteration (of the given run), '
                                                     'None does restore the most current iteration [None]')
FLAGS = flags.FLAGS


class MouseButtons:
    LEFT = 1
    MIDDLE = 2
    RIGHT = 3
    WHEEL_UP = 4
    WHEEL_DOWN = 5


class UndoStack:
    def __init__(self, screen, size=20):
        self.screen = screen
        self.undo_size = size
        self.undo_stack = [None] * self.undo_size
        self.undo_idx = -1

    def push(self):
        self.undo_idx += 1
        self.undo_stack[self.undo_idx % self.undo_size] = pygame.image.tostring(self.screen, "RGBA")

    def restore_current_idx(self):
        s = pygame.image.fromstring(self.undo_stack[self.undo_idx % self.undo_size], self.screen.get_size(), 'RGBA')
        self.screen.blit(s, (0, 0))

    def pop_forward(self):
        self.undo_idx += 1
        self.restore_current_idx()

    def pop_backward(self):
        if self.undo_idx == 0:
            print('Reached end of stack')
        else:
            self.undo_idx -= 1
            self.restore_current_idx()


class SketchScreen:
    def __init__(self, title):
        pygame.init()
        self.draw_on = False
        self.last_pos = (0, 0)
        self.radius = 5
        self.screen = pygame.display.set_mode((512, 512))
        pygame.display.set_caption(title + ', press H for help')
        self.strength = 125
        self.undo = UndoStack(self.screen)
        self.undo.push()
        self.set_paint_cursor(self.radius)
        self.showing_help = False
        self.save = False

    def set_paint_cursor(self, radius):
        paint_cursor_strings = []
        grid_size = int(np.ceil(2 * radius / 8)) * 8
        print(radius, grid_size)
        for x in xrange(grid_size):
            paint_cursor_string = ''
            for y in xrange(grid_size):
                if (x - grid_size / 2) ** 2 + (y - grid_size / 2) ** 2 <= radius * radius:
                    paint_cursor_string += 'X'
                else:
                    paint_cursor_string += ' '
            paint_cursor_strings.append(paint_cursor_string)
        paint_cursor, paint_mask = pygame.cursors.compile(paint_cursor_strings)
        pygame.mouse.set_cursor((grid_size, grid_size), (grid_size // 2, grid_size // 2), paint_cursor, paint_mask)

    def show_help(self):
        if self.showing_help:
            self.undo.pop_backward()
        else:
            self.undo.push()
            self.screen.fill((0, 0, 0, 0))
            font = pygame.font.SysFont('monospace', 24)
            messages = [['Available keys:', ''],
                        ['L', 'Load a training image'],
                        ['C', 'Clear screen'],
                        ['S', 'Save current sketch/output']
                        ['Z', 'Undo last stroke'],
                        ['X', 'Redo last stroke'],
                        ['+', 'Increase brush radius'],
                        ['-', 'Decrease brush radius'],
                        ['Keypad Nr', 'Brightness of stroke'],
                        ['H', 'Show/Hide this help'],
                        ['LM pressed', 'Draw'],
                        ['RM pressed', 'Erase']]
            y = 10
            for label, message in messages:
                l = font.render(label, 1, (255, 255, 255))
                m = font.render(message, 1, (255, 255, 255))
                self.screen.blit(l, (10, y))
                self.screen.blit(m, (200, y))
                y += 30

        self.showing_help = not self.showing_help

    def roundline(self, srf, color, start, end, radius=1):
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        distance = max(abs(dx), abs(dy))
        for i in range(distance):
            x = int(start[0] + float(i) / distance * dx)
            y = int(start[1] + float(i) / distance * dy)
            pygame.draw.circle(srf, color, (x, y), radius)

    def brush_at(self, pos, do_roundline=False):
        # color = screen.get_at(pos)
        # color += pygame.Color(strength, strength, strength)
        color = pygame.Color(self.strength, self.strength, self.strength)
        pygame.draw.circle(self.screen, color, pos, self.radius)
        if do_roundline:
            self.roundline(self.screen, color, pos, self.last_pos, self.radius)

    def get_content_as_np_array(self):
        return pygame.surfarray.array3d(self.screen)[:, :, 0].astype(np.float32) / (32.0 * 8)

    def enter_loop(self):
        try:
            while True:
                e = pygame.event.wait()
                if e.type == pygame.QUIT:
                    raise StopIteration
                elif e.type == pygame.KEYDOWN:
                    pressed = pygame.key.get_pressed()
                    if pressed[K_h]:
                        self.show_help()
                    if not self.showing_help:
                        if pressed[K_l]:
                            # l: Loads initial image
                            filename = random.sample(['1aeb17f89e1bea954c6deb9ede0648df_r_036_sketch_64x64.png'], 1)
                            loaded_img = pygame.transform.scale(pygame.image.load('test_sketches/' + filename[0]),
                                                                (512, 512))
                            self.screen.blit(loaded_img, (0, 0))
                            self.undo.push()
                        elif pressed[K_c]:
                            # c: Clears screen
                            self.screen.fill((0, 0, 0, 0))
                            self.undo.push()
                        elif pressed[K_s]:
                            self.save = True
                        elif pressed[K_z]:
                            self.undo.pop_backward()
                        elif pressed[K_x]:
                            self.undo.pop_forward()
                        elif pressed[K_KP_PLUS] or pressed[K_PLUS]:
                            self.radius = min(self.radius + 3, 15)
                            self.set_paint_cursor(self.radius)
                        elif pressed[K_KP_MINUS] or pressed[K_MINUS]:
                            self.radius = max(self.radius - 3, 2)
                            self.set_paint_cursor(self.radius)
                        else:
                            for x in range(K_KP1, K_KP9 + 1):
                                if pressed[x]:
                                    self.strength = (x - K_KP0) * 255 // 9
                            for x in range(K_1, K_9 + 1):
                                if pressed[x]:
                                    self.strength = (x - K_0) * 255 // 9

                elif not self.showing_help:
                    if e.type == pygame.MOUSEBUTTONDOWN:
                        if e.button == MouseButtons.RIGHT:
                            self.previous_strength = self.strength
                            self.strength = 0
                        else:
                            self.previous_strength = None

                        self.brush_at(e.pos)
                        self.draw_on = True
                    elif e.type == pygame.MOUSEBUTTONUP:
                        if self.previous_strength is not None:
                            self.strength = self.previous_strength
                        self.draw_on = False
                        self.undo.push()
                    elif e.type == pygame.MOUSEMOTION:
                        if self.draw_on:
                            self.brush_at(e.pos, True)
                        self.last_pos = e.pos
                pygame.display.flip()


        except StopIteration:
            pass

        pygame.quit()


class OutputScreen:
    def __init__(self, size, z_dim, c_dim):
        self.c_dim = c_dim

        fig, (ax_input, ax_output) = plt.subplots(2, 1)
        data = np.zeros([size, size, 3])
        self.imshow_window = ax_output.imshow(data, interpolation='nearest', aspect='equal')
        self.downsampled_input = ax_input.imshow(data, interpolation='nearest', aspect='equal')

        self.sliders = []
        if z_dim:
            for i in xrange(4):
                ax = fig.add_axes([0.2, 0.03 * i, 0.65, 0.03])
                s = Slider(ax, 'Random ' + str(i), -1, 1, valinit=0)
                self.sliders.append(s)

        fig.show()

    def update_content(self, input, output):
        self.downsampled_input.set_data(np.repeat(input, 3, 2))
        if self.c_dim == 1:
            output = np.repeat(output, 3, 2)
        self.imshow_window.set_data(output)
        plt.draw()


def main(_):
    runs = sorted(map(int, next(os.walk(FLAGS.checkpoint_dir))[1]))
    if FLAGS.continue_from:
        run_folder = FLAGS.continue_from
    else:
        run_folder = str(runs[-1]).zfill(3)

    used_checkpoint_dir = os.path.join(FLAGS.checkpoint_dir, run_folder)
    print('Restoring from ' + FLAGS.checkpoint_dir)

    with tf.Session(config=tf.ConfigProto(device_count={'GPU': 0})) as sess:
        dcgan = DCGAN(sess, batch_size=1, is_train=False)
        full_sketch = tf.placeholder(tf.float32, [512, 512])
        # TODO: Scale full_sketch to fill out image.
        small_sketch = tf.image.resize_area(tf.reshape(full_sketch, [1, 512, 512, 1]), [64, 64])
        small_sketch = preprocess(small_sketch, 64, whiten='sketch', color=False, augment=False)
        small_sketch = tf.transpose(small_sketch, [1, 0, 2])
        small_sketch = tf.reshape(small_sketch, [1, 64, 64, 1])

        # Directly feed sketch
        # test_sketch = make_image_producer(['test_sketches/part_of_recolor_experiment.png'], 10, 'rendered_producer',
        #                                   64,
        #                                   shuffle=False, filename_seed=1, whiten='sketch', color=False,
        #                                   augment=False)
        # test_sketch = tf.train.batch([test_sketch], batch_size=1)

        tf.initialize_all_variables().run()
        ckpt_name = dcgan.load(used_checkpoint_dir, FLAGS.continue_from_iteration)
        print('Successfully reconstructed network with variables.')
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        # s = sess.run(test_sketch)
        # print("Got sketch")

        sc = SketchScreen('Variables restored from: ' + ckpt_name)

        draw_thread = threading.Thread(target=sc.enter_loop)
        draw_thread.start()

        output_screen = OutputScreen(64, dcgan.z_dim, dcgan.c_dim)
        param_stats = tf.contrib.tfprof.model_analyzer.print_model_analysis(
                tf.get_default_graph(),
                tfprof_options=tf.contrib.tfprof.model_analyzer.
                    TRAINABLE_VARS_PARAMS_STAT_OPTIONS)
        print('total_params: %d\n' % param_stats.total_parameters)
        tf.contrib.tfprof.model_analyzer.print_model_analysis(
                tf.get_default_graph(),
                tfprof_options=tf.contrib.tfprof.model_analyzer.FLOAT_OPS_OPTIONS)

        # Get timings of one sample run
        run_metadata = tf.RunMetadata()
        _ = sess.run(dcgan.G,
                     options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
                     run_metadata=run_metadata,
                     feed_dict = {dcgan.sketches: np.zeros([1, 64, 64, 1], dtype=np.float32),
                                  dcgan.z: np.zeros([1, 4], dtype=np.float32)})
        tf.contrib.tfprof.model_analyzer.print_model_analysis(
                tf.get_default_graph(),
                run_meta=run_metadata,
                tfprof_options=tf.contrib.tfprof.model_analyzer.PRINT_ALL_TIMING_MEMORY)

        while draw_thread.is_alive:
            try:
                A = sc.get_content_as_np_array()
                B = np.argwhere(A)
                if len(B) > 0:
                    (ystart, xstart), (ystop, xstop) = B.min(0), B.max(0) + 1
                    full_sketch_trimmed = A[ystart:ystop, xstart:xstop]
                    resize_factor = 512 / max(full_sketch_trimmed.shape)
                    full_sketch_resized = scipy.misc.imresize(full_sketch_trimmed, resize_factor)
                    x_padding_needed = 512 - full_sketch_resized.shape[0]
                    y_padding_needed = 512 - full_sketch_resized.shape[1]
                    full_sketch_final = np.pad(full_sketch_resized,
                                               ((int(np.floor(x_padding_needed / 2)),
                                                 int(np.ceil(x_padding_needed / 2))),
                                                (int(np.floor(y_padding_needed / 2)),
                                                 int(np.ceil(y_padding_needed / 2)))),
                                               'constant', constant_values=0)
                else:
                    # Empty dummy image
                    full_sketch_final = np.zeros([512, 512], dtype=np.float32)
                s = sess.run(small_sketch, feed_dict={full_sketch: full_sketch_final})

                unnormed_small_sketch = (np.reshape(s, [64, 64, 1]) + 1) / 2

                # Dumps painted sketch.

                feed = {dcgan.sketches: s}
                if dcgan.z_dim:
                    z = np.reshape(np.asarray([slider.val * 10 for slider in output_screen.sliders], dtype=np.float32),
                                   [1, 4])
                    feed[dcgan.z] = z
                start = time.clock()
                img = sess.run(dcgan.G, feed_dict=feed)
                end = time.clock()

                unnormed_img = np.reshape(img, [64, 64, dcgan.c_dim])
                unnormed_img = (unnormed_img + 1) / 2

                output_screen.update_content(unnormed_small_sketch, unnormed_img)
                print('Updated image, computed in {:.5}s'.format(end - start))
                if sc.save:
                    sc.save = False
                    path = os.path.join(used_checkpoint_dir, 'interactive')
                    if not os.path.exists(path):
                        os.makedirs(path)
                    files = sorted(glob.glob(os.path.join(path, ckpt_name + '*.png')))
                    if files:
                        current = int(re.search('_n_(\d)+', files[-1]).group(1)) + 1
                    else:
                        current = 0
                    path_with_basename = os.path.join(path, ckpt_name + '_n_' + str(current).zfill(3))
                    scipy.misc.imsave(path_with_basename + '_sketch.png', full_sketch_final)
                    scipy.misc.imsave(path_with_basename + '_sketch_small.png', np.reshape(unnormed_small_sketch, [64, 64]))
                    scipy.misc.imsave(path_with_basename + '_output.png', unnormed_img)
                    print('Saved to ' + path)
                plt.pause(0.5)
            except pygame.error:
                print('Pygame stoped, shutting down.')
                break

        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    tf.app.run()
