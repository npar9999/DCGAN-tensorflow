"""
Some codes from https://github.com/Newmu/dcgan_code
"""
import math
import pprint

import numpy as np
import scipy.misc

pp = pprint.PrettyPrinter()

get_stddev = lambda x, k_h, k_w: 1 / math.sqrt(k_w * k_h * x.get_shape()[-1])


def get_image(image_path, image_size):
  return transform(imread(image_path), image_size)


def save_images(images, grid_size, image_path, invert=True, channels=3):
  if invert:
    images = inverse_transform(images)
  return imsave(images, grid_size, image_path, channels)


def imread(path):
  return scipy.misc.imread(path).astype(np.float)


def imsave(images, grid_size, path, channels):
  h, w = images.shape[1], images.shape[2]
  img = np.zeros((h * grid_size[0], w * grid_size[1], channels))

  for idx, image in enumerate(images):
    i = idx % grid_size[1]
    j = idx // grid_size[1]
    if channels == 1:
      img[j * h:j * h + h, i * w:i * w + w, 0] = image
    else:
      img[j * h:j * h + h, i * w:i * w + w, :] = image
  # Flatten third dimension if only grayscale image (necessary for scipy image save method).
  if channels == 1:
    img = img.reshape(img.shape[0:2])
  return scipy.misc.imsave(path, img)


def center_crop(x, crop_h, crop_w=None, resize_w=64):
  if crop_w is None:
    crop_w = crop_h
  h, w = x.shape[:2]
  j = int(round((h - crop_h) / 2.))
  i = int(round((w - crop_w) / 2.))
  return scipy.misc.imresize(x[j:j + crop_h, i:i + crop_w],
                             [resize_w, resize_w])


def transform(image, npx=64):
  # npx : # of pixels width/height of image
  cropped_image = center_crop(image, npx)
  return np.array(cropped_image) / 127.5 - 1.


def inverse_transform(images):
  return (images + 1.) / 2.
