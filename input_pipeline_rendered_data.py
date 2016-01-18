import re, os
import tensorflow as tf
import numpy as np
import glob

def get_files(folder, file_regexp):
  files = []
  for root, dirnames, filenames in os.walk(folder):
      for filename in filenames:
        if re.match(file_regexp, filename):
          files.append(os.path.join(root, filename))
  return sorted(files)

def get_files_cached(folder, type, regexp, reload=False):
  f = os.path.join(folder, '{}_files.txt'.format(type))
  if os.path.isfile(f) and not reload:
    return [filename.strip() for filename in open(f).readlines()]
  else:
    filelist = get_files(folder, regexp)
    with open(f, 'w') as file_write:
      file_write.write('\n'.join(filelist))
    return filelist


def get_rendered_files(folder, size_suffix='64x64'):
  return get_files_cached(folder, 'rendered', '.*r_\d{3}_' + size_suffix + '\.png$')


def get_albedo_files(folder, size_suffix='64x64'):
  return get_files_cached(folder, 'albedo', '.*r_\d{3}_albedo\.png0001_' + size_suffix + '\.png$')


def get_normal_files(folder, size_suffix='64x64'):
  return get_files_cached(folder, 'normal', '.*r_\d{3}_normal\.png0001_' + size_suffix + '\.png$')


def get_depth_files(folder, size_suffix='64x64'):
  return get_files_cached(folder, 'depth', '.*r_\d{3}_depth\.png0001_' + size_suffix + '\.png$')


def get_sketch_files(folder, size_suffix='64x64'):
  return get_files_cached(folder, 'sketch', '.*r_\d{3}_sketch_' + size_suffix + '.png$')


def preprocess(image_tensor, img_size, whiten=True, color=False,
               augment=True, augment_color=False):
  # Use same seed for flipping for every tensor, so they'll be flipped the same.
  seed = 42
  if color:
    out = tf.reshape(image_tensor, [img_size, img_size, 3])
  else:
    out = tf.reshape(image_tensor, [img_size, img_size, 1])
  if augment:
    out = tf.image.random_flip_left_right(out, seed)
    # TODO: add more data augmentation.
  if augment_color:
    out = tf.image.random_hue(out, 0.5, seed)
    out = tf.image.random_saturation(out, 0.8, 1.2, seed)
  if whiten:
    # Bring to range [-1, 1]
    out = tf.cast(out, tf.float32) * (2. / 255) - 1
  else:
    out = tf.cast(out, tf.float32) * (1. / 255)
  return out

def make_image_producer(files, epochs, name, img_size, shuffle, whiten, color,
                        augment=True, capacity=256, augment_color=False):
  with tf.variable_scope(name) as scope:
    filename_seed = 233
    gray_filename_queue = tf.train.string_input_producer(files, num_epochs=epochs, seed=filename_seed,
                                                         capacity=capacity, shuffle=shuffle)
    _, gray_files = tf.WholeFileReader(scope.name).read(gray_filename_queue)
    channels = 3 if color else 1
    return preprocess(tf.image.decode_png(gray_files, channels), img_size,
                      whiten=whiten, color=color, augment=augment, augment_color=augment_color)


def get_chair_pipeline(batch_size, epochs, img_size, depth_files, sketch_files,
                       shuffle=True, augment_color=False):
  img = make_image_producer(depth_files, epochs, 'rendered_producer', img_size,
                            shuffle, whiten=True, color=True, augment_color=augment_color)
  sketches = make_image_producer(sketch_files, epochs, 'sketch_producer', img_size,
                                 shuffle, whiten=False, color=False)
  return tf.train.batch([sketches, img], batch_size=batch_size, num_threads=1, capacity=256 * 16)


def get_chair_pipeline_training(batch_size, epochs):
  chair_folder = '/home/moser/shapenet_chairs_rendered2'
  sketch_folder = '/home/moser/shapenet_chairs_sketched2'
  img_size = 64
  size_suffix = str(img_size) + 'x' + str(img_size)
  return get_chair_pipeline(batch_size, epochs, img_size, get_rendered_files(chair_folder, size_suffix),
                            get_sketch_files(sketch_folder, size_suffix))

def get_chair_pipeline_training_recolor(batch_size, epochs):
  img_size = 64
  rendered_files = [x.strip() for x in open('recolor_experiment_shaded_images.txt').readlines()]
  sketched_files = [x.strip() for x in open('recolor_experiment_sketch_images.txt').readlines()]

  return get_chair_pipeline(batch_size, epochs, img_size, rendered_files,
                            sketched_files, augment_color=True)


def get_chair_pipeline_training_from_dump(dump_file, batch_size, epochs, min_queue_size=1000):
  with tf.device('/cpu:0'):
    reader = tf.TFRecordReader()
    all_files = glob.glob(dump_file + '*')
    files = tf.train.string_input_producer(all_files, num_epochs=epochs)
    _, serialized_example = reader.read(files)
    features = tf.parse_single_example(
        serialized_example,
        features={'image': tf.FixedLenFeature([], tf.string),
                  'sketch': tf.FixedLenFeature([], tf.string)})
    img_size = 64
    image = tf.decode_raw(features['image'], tf.uint8)
    image.set_shape([img_size * img_size * 3])
    image = preprocess(image, img_size,
                       whiten=False, color=True, augment=True, augment_color=True)
    sketch = tf.decode_raw(features['sketch'], tf.uint8)
    sketch.set_shape([img_size * img_size * 1])
    sketch = preprocess(sketch, img_size,
                        whiten=False, color=False, augment=True)
    return tf.train.shuffle_batch([sketch, image], batch_size=batch_size,
                                  capacity=min_queue_size + batch_size*16,
                                  min_after_dequeue=min_queue_size,
                                  num_threads=2)
