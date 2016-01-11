import re, os
import tensorflow as tf
import numpy as np


def get_files(folder, file_regexp):
  files = []
  for root, dirnames, filenames in os.walk(folder):
      for filename in filenames:
        if re.match(file_regexp, filename):
          files.append(os.path.join(root, filename))
  return sorted(files)

def get_files_cached(folder, type, regexp, reload=True):
  f = os.path.join(folder, '{}_files.txt'.format(type))
  if os.path.isfile(f) and not reload:
    return open(f).readlines()
  else:
    filelist = get_files(folder, regexp)
    with open(f, 'w') as file_write:
      file_write.write('\n'.join(filelist))
    return filelist


def get_albedo_files(folder, size_suffix='64x64'):
  return get_files_cached(folder, 'albedo', '.*r_\d{3}_albedo\.png0001_' + size_suffix + '\.png$')


def get_normal_files(folder, size_suffix='64x64'):
  return get_files_cached(folder, 'normal', '.*r_\d{3}_normal\.png0001_' + size_suffix + '\.png$')


def get_depth_files(folder, size_suffix='64x64'):
  return get_files_cached(folder, 'depth', '.*r_\d{3}_depth\.png0001_' + size_suffix + '\.png$')


def get_sketch_files(folder, size_suffix='64x64'):
  return get_files_cached(folder, 'sketch', '.*r_\d{3}_sketch_' + size_suffix + '.png$')


def preprocess(image_tensor, img_size, whiten=True, color=False):
  # Use same seed for flipping for ever tensor, so they'll be flipped the same.
  seed = 42
  if color:
    out = tf.reshape(image_tensor, [img_size, img_size, 3])
  else:
    out = tf.reshape(image_tensor, [img_size, img_size, 1])
  out = tf.image.random_flip_left_right(out, seed)
  if whiten:
    # Bring to range [-1, 1]
    out = tf.cast(out, tf.float32) * (2. / 255) - 1
  else:
    out = tf.cast(out, tf.float32) * (1. / 255)
  return out


# Puts reading/preprocessing on CPU by default.
def get_chair_pipeline(batch_size, epochs):
  chair_folder = '/home/moser/shapenet_chairs_rendered2'
  sketch_folder = '/home/moser/shapenet_chairs_sketched2'
  filename_seed = 233
  capacity = 256
  img_size = 64
  size_suffix = str(img_size) + 'x' + str(img_size)
  with tf.device('/cpu:0'):

    with tf.variable_scope('depth_producer') as scope:
      gray_filename_queue = tf.train.string_input_producer(get_depth_files(chair_folder, size_suffix),
                                                           num_epochs=epochs, seed=filename_seed, capacity=capacity)
      _, gray_files = tf.WholeFileReader(scope.name).read(gray_filename_queue)
      depth = preprocess(tf.image.decode_png(gray_files, 1), img_size, whiten=True, color=False)

    with tf.variable_scope('sketch_producer') as scope:
      sketch_filename_queue = tf.train.string_input_producer(get_sketch_files(sketch_folder, size_suffix),
                                                             num_epochs=epochs, seed=filename_seed, capacity=capacity)
      _, sketch_files = tf.WholeFileReader(scope.name).read(sketch_filename_queue)
      sketches = preprocess(tf.image.decode_png(sketch_files, 1), img_size, whiten=False)

  return tf.train.batch([sketches, depth], batch_size=batch_size, num_threads=1, capacity=256 * 16)
