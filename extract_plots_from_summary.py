from __future__ import division
from tensorflow.python.summary import event_accumulator as ea
import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt

def dump_images_contained(acc):
  for i in xrange(3):
    for image_type in ['generator', 'images']:
      for img in acc.Images('{}/image/{}'.format(image_type, i)):
        with open('{}_{}_{}.png'.format(img.step, i, image_type), 'wb') as f:
          f.write(img.encoded_image_string)


def smooth(x, smoothing):
    return [np.median(x[i-smoothing:i]) for i in range(smoothing, len(x))]


def dump_smoothed_plot(acc, output_name, do_smooth=False, smoothing_fraction=1/8):
  gen_graph = [x.value for x in acc.Scalars('g_loss')]
  dis_graph = [x.value for x in acc.Scalars('d_loss')]
  idx_graph = [x.step for x in acc.Scalars('g_loss')]

  if do_smooth:
    smoothing = int(len(idx_graph) * smoothing_fraction)
    idx_graph = idx_graph[smoothing:]
    gen_graph = smooth(gen_graph, smoothing)
    dis_graph = smooth(dis_graph, smoothing)
    output_name += '_smoothed'

  plt.figure(figsize=(15,7))
  ax = plt.gca()
  ax.plot(idx_graph, gen_graph, label="generator score")
  ax2 = ax.twinx()
  ax2.plot(idx_graph, dis_graph, label="discrimin score", color='green')
  plt.legend(loc="lower left")
  plt.savefig(output_name + '.png')


flags = tf.app.flags
flags.DEFINE_string('summary_dir', 'summary_sketches_to_rendered', 'Folder where runs are placed.')
flags.DEFINE_string("run", None, 'Name of the run you want to extract from.')
FLAGS = flags.FLAGS

def main(_):
  runs = sorted(map(int, next(os.walk(FLAGS.summary_dir))[1]))
  if not runs:
    raise Exception('No runs available!')
  if FLAGS.run is None:
    file = os.path.join(FLAGS.summary_dir, str(runs[-1]).zfill(3))
  else:
    file = os.path.join(FLAGS.summary_dir, FLAGS.run)

  acc = ea.EventAccumulator(file)
  acc.Reload()
  print(acc.Tags())

  #dump_smoothed_plot(acc, 'loss')
  dump_smoothed_plot(acc, 'loss', True)

if __name__ == '__main__':
    tf.app.run()
