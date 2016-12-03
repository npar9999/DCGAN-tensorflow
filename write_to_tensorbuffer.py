from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from random import shuffle

import numpy
from PIL import Image

from input_pipeline_rendered_data import *

tf.app.flags.DEFINE_string('directory', '/tmp', 'Place to dump file')
tf.app.flags.DEFINE_string('name', 'all_chairs_sketch_rendered_64x64', 'Name of dump to be produced.')
tf.app.flags.DEFINE_integer('examples_per_file', 2000, 'Number of examples per file. [2000]')
tf.app.flags.DEFINE_integer('image_size', 64, 'Excpected width and length of all images, [64]')

FLAGS = tf.app.flags.FLAGS


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def write_to(writer, sketch, image):
  example = tf.train.Example(features=tf.train.Features(feature={
    'image': _bytes_feature(image.tostring()),
    'sketch': _bytes_feature(sketch.tostring())
  }))
  writer.write(example.SerializeToString())


def main(argv):
  filename = os.path.join(FLAGS.directory, FLAGS.name + '.tfrecords')
  writer = tf.python_io.TFRecordWriter(filename)
  # chair_folder = '/home/moser/shapenet_chairs_rendered2'
  # sketch_folder = '/home/moser/shapenet_chairs_sketched2'
  #
  # rendered_files = get_rendered_files(chair_folder)
  # sketched_files = get_sketch_files(sketch_folder)
  # TODO: Check if output files have all same size (except last one).
  rendered_files = [x.strip() for x in open('../shapenet_chairs_rendered2/rendered_files.txt').readlines()]
  sketched_files = [x.strip() for x in open('../shapenet_chairs_sketched2/sketch_files.txt').readlines()]
  if len(sketched_files) != len(rendered_files):
    raise Exception("Not same amount of files for different features.")
  print('Writing {} samples to {}'.format(len(rendered_files), filename))
  zipped = zip(rendered_files, sketched_files)
  shuffle(zipped)
  for idx, (rendered_file, sketch_file) in list(enumerate(zipped)):
    pic = Image.open(rendered_file)
    pic_data = numpy.array(pic.getdata(), dtype=np.uint8)
    if not pic_data.shape[0] == FLAGS.image_size * FLAGS.image_size:
      raise Exception('Unexcpected image size for {}: {}'.format(rendered_file, pic_data.shape))
    if pic_data.shape[1] == 4:
      rendered = pic_data[:, 0:3]
    elif pic_data.shape[1] == 2:
      rendered = numpy.zeros([pic_data.shape[0], 3], dtype=np.uint8)
      for i in xrange(3):
        rendered[:, i] = pic_data[:, 0]
    else:
      raise Exception('Unexpected number of channels!')

    sketch = numpy.array(Image.open(sketch_file).getdata(), dtype=np.uint8)
    if not sketch.shape[0] == FLAGS.image_size * FLAGS.image_size:
      raise Exception('Unexcpected image size for {}: {}'.format(sketch_file, sketch.shape))
    if len(sketch.shape) == 2 and sketch.shape[1] == 2:
      sketch = sketch[:, 0]
    else:
      print('Skipped buggy sketch: ' + sketch_file)
      continue

    write_to(writer, sketch, rendered)

    if idx % 1000 == 1:
      print(idx)
    if idx % FLAGS.examples_per_file == 1 and idx > 1:
      filename = os.path.join(FLAGS.directory, FLAGS.name + '.tfrecords' + str(idx // FLAGS.examples_per_file))
      writer = tf.python_io.TFRecordWriter(filename)


if __name__ == '__main__':
  tf.app.run()
