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

class GraphGatherer:
  l2_graph = None
  idx_graph = None
  gen_graph = None
  dis_graph = None

  def __init__(self, output_folder, acc):
    self.output_folder = output_folder
    scalars = acc.Tags()['scalars']
    if 'l2_loss' in scalars:
      self.l2_graph = [x.value for x in acc.Scalars('l2_loss')]
      self.idx_graph = [x.step for x in acc.Scalars('l2_loss')]
    if 'g_loss' in scalars and 'd_loss' in scalars:
      self.gen_graph = [x.value for x in acc.Scalars('g_loss')]
      self.dis_graph = [x.value for x in acc.Scalars('d_loss')]
      if self.idx_graph is None:
        self.idx_graph = [x.step for x in acc.Scalars('g_loss')]

  def dump_l2_plot(self, output_name, do_smooth=False, smoothing_fraction=1/8):
    if self.l2_graph is None:
      return
    if do_smooth:
      smoothing = int(len(self.idx_graph) * smoothing_fraction)
      idx_graph = self.idx_graph[:-smoothing]
      l2_graph = smooth(self.l2_graph, smoothing)
    else:
      idx_graph = self.idx_graph
      l2_graph = self.l2_graph

    plt.figure(figsize=(15,7))
    ax = plt.gca()
    ax.plot(idx_graph, l2_graph, label="L2 error")
    plt.savefig(os.path.join(self.output_folder, output_name + '.png'))

  def dump_plot(self, output_name, do_smooth=False, smoothing_fraction=1/8):
    if self.gen_graph is None:
      return
    if do_smooth:
      smoothing = int(len(self.idx_graph) * smoothing_fraction)
      idx_graph = self.idx_graph[:-smoothing]
      gen_graph = smooth(self.gen_graph, smoothing)
      dis_graph = smooth(self.dis_graph, smoothing)
    else:
      idx_graph = self.idx_graph
      gen_graph = self.gen_graph
      dis_graph = self.dis_graph

    plt.figure(figsize=(15,7))
    ax = plt.gca()
    ax.plot(idx_graph, gen_graph, label="generator score")
    ax2 = ax.twinx()
    ax2.plot(idx_graph, dis_graph, label="discrimin score", color='green')
    ax2.legend(loc='upper right')
    ax.legend(loc='upper left')
    plt.savefig(os.path.join(self.output_folder, output_name + '.png'))

flags = tf.app.flags
flags.DEFINE_string('summary_dir', 'summary', 'Folder where runs are placed.')
flags.DEFINE_string("run", '037', 'Name of the run you want to extract from.')
FLAGS = flags.FLAGS

def main(_):
  runs = sorted(next(os.walk(FLAGS.summary_dir))[1])
  if not runs:
    raise Exception('No runs available!')
  if FLAGS.run is None:
    folder = os.path.join(FLAGS.summary_dir, str(runs[-1]).zfill(3))
  else:
    folder = os.path.join(FLAGS.summary_dir, FLAGS.run)

  acc = ea.EventAccumulator(folder)
  acc.Reload()
  print(acc.Tags())
  gatherer = GraphGatherer(folder, acc)
  gatherer.dump_l2_plot('l2')
  gatherer.dump_l2_plot('l2_smoothed', True, smoothing_fraction=1/20)
  gatherer.dump_plot('loss')
  gatherer.dump_plot('loss_smoothed', True, smoothing_fraction=1/20)

if __name__ == '__main__':
    tf.app.run()
