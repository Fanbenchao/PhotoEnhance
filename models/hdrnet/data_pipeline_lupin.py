# Copyright 2016 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Data pipeline for HDRnet."""

import abc
import json
import logging
import magic
import numpy as np
import os
import random
import xml.dom.minidom
import tensorflow as tf

log = logging.getLogger("root")
log.setLevel(logging.INFO)

__all__ = [
  'ImageFilesDataPipeline',
  'StyleTransferDataPipeline',
  'HDRpDataPipeline',
  ]

class DataPipeline(object):
  """Abstract operator to stream data to the network.

  Attributes:
    path:
    batch_size: number of sampes per batch.
    capacity: max number of samples in the data queue.
    pass
    min_after_dequeue: minimum number of image used for shuffling the queue.
    shuffle: random shuffle the samples in the queue.
    nthreads: number of threads use to fill up the queue.
    samples: a dict of tensors containing a batch of input samples with keys:
      'image_input' (the source image),
      'image_output' (the filtered image to match),
      'guide_image' (the image used to slice in the grid).
  """

  __metaclass__ = abc.ABCMeta

  def __init__(self, path, batch_size=32,
               capacity=16, 
               min_after_dequeue=4,
               output_resolution=[1080, 1920],
               shuffle=False,
               fliplr=False, 
               flipud=False,
               rotate=False,
               random_crop=False,
               params=None,
               nthreads=1, 
               num_epochs=None):
    self.path = path
    self.batch_size = batch_size
    self.capacity = capacity
    self.min_after_dequeue = min_after_dequeue
    self.shuffle = shuffle
    self.nthreads = nthreads
    self.num_epochs = num_epochs
    self.params = params

    self.output_resolution = output_resolution

    # Data augmentation
    self.fliplr = fliplr
    self.flipud = flipud
    self.rotate = rotate
    self.random_crop = random_crop

    sample = self._produce_one_sample()
    self.samples = self._batch_samples(sample)

  @abc.abstractmethod
  def _produce_one_sample(self):
    pass

  def _batch_samples(self, sample):
    """Batch several samples together."""

    # Batch and shuffle
    if self.shuffle:
      samples = tf.compat.v1.train.shuffle_batch(
          sample,
          batch_size=self.batch_size,
          num_threads=self.nthreads,
          capacity=self.capacity,
          min_after_dequeue=self.min_after_dequeue)
    else:
      samples = tf.compat.v1.train.batch(
          sample,
          batch_size=self.batch_size,
          num_threads=self.nthreads,
          capacity=self.capacity)
    return samples

  def _augment_data(self, inout,nchan=6):
    """Flip, crop and rotate samples randomly."""

    with tf.name_scope('data_augmentation'):
      if self.fliplr:
        inout = tf.image.random_flip_left_right(inout, seed=1234)
      if self.flipud:
        inout = tf.image.random_flip_up_down(inout, seed=3456)
      if self.rotate:
        angle = tf.compat.v1.random_uniform((), minval=0, maxval=4, dtype=tf.int32, seed=4567)
        inout = tf.case([(tf.equal(angle, 1), lambda: tf.image.rot90(inout, k=1)),
                         (tf.equal(angle, 2), lambda: tf.image.rot90(inout, k=2)),
                         (tf.equal(angle, 3), lambda: tf.image.rot90(inout, k=3))],
                        lambda: inout)

      inout.set_shape([None, None, nchan])

      with tf.name_scope('crop'):
        shape = tf.shape(inout)
        new_height = tf.compat.v1.to_int32(self.output_resolution[0])
        new_width = tf.compat.v1.to_int32(self.output_resolution[1])
        height_ok = tf.compat.v1.assert_less_equal(new_height, shape[0])
        width_ok = tf.compat.v1.assert_less_equal(new_width, shape[1])
        with tf.control_dependencies([height_ok, width_ok]):
          if self.random_crop:
            inout = tf.image.random_crop(
                inout, tf.stack([new_height, new_width, nchan]))
          else:
            height_offset = tf.compat.v1.to_int32((shape[0]-new_height)/2)
            width_offset = tf.compat.v1.to_int32((shape[1]-new_width)/2)
            inout = tf.image.crop_to_bounding_box(
                inout, height_offset, width_offset,
                new_height, new_width)

      inout.set_shape([None, None, nchan])
      inout = tf.image.resize(inout, [self.output_resolution[0], self.output_resolution[1]],
                             method=tf.image.ResizeMethod.BICUBIC)
      fullres = inout

      with tf.name_scope('resize'):
        new_size = 256
        inout = tf.image.resize(
            inout, [new_size, new_size],
            method=tf.image.ResizeMethod.BICUBIC)

      return fullres, inout


class ImageFilesDataPipeline(DataPipeline):
  """Pipeline to process pairs of images from a list of image files.

  Assumes path contains:
    - a file named 'filelist.txt' listing the image names.
    - a subfolder 'input'
    - a subfolder 'output'
  """

  def _produce_one_sample(self):
    dirname = os.path.dirname(self.path)
    with open(self.path, 'r') as fid:
      flist = [l.strip() for l in fid.readlines()]
        
    if self.shuffle:
      random.shuffle(flist)
    input_files = []
    output_files = []
    for f in flist:
        pre_name,img_name = f.split('_')[0], f.split('_')[1]
        input_files.append(os.path.join(dirname,'image',pre_name,img_name))
        output_files.append(os.path.join(dirname,'label',pre_name+'.JPG'))

    self.nsamples = len(input_files)

    input_queue, output_queue = tf.compat.v1.train.slice_input_producer(
        [input_files, output_files], shuffle=self.shuffle,
        seed= 123, num_epochs=self.num_epochs)

    if '16-bit' in magic.from_file(input_files[0]):
      input_dtype = tf.uint16
      input_wl = 65535.0
    else:
      input_wl = 255.0
      input_dtype = tf.uint8
    if '16-bit' in magic.from_file(output_files[0]):
      output_dtype = tf.uint16
      output_wl = 65535.0
    else:
      output_wl = 255.0
      output_dtype = tf.uint8

    input_file = tf.io.read_file(input_queue)
    output_file = tf.io.read_file(output_queue)
    
    if os.path.splitext(input_files[0])[-1] == '.jpg': 
      im_input = tf.io.decode_jpeg(input_file, channels=3)
    else:
      im_input = tf.io.decode_png(input_file, dtype=input_dtype, channels=3)

    if os.path.splitext(output_files[0])[-1] == '.jpg': 
      im_output = tf.io.decode_jpeg(output_file, channels=3)
    else:
      im_output = tf.io.decode_png(output_file, dtype=output_dtype, channels=3)
    sample = {}
    with tf.name_scope('normalize_images'):
      im_input = tf.compat.v1.to_float(im_input)/input_wl
      im_output = tf.compat.v1.to_float(im_output)/output_wl

    inout = tf.concat([im_input,im_output], 2)

    fullres, inout = self._augment_data(inout,6)

    sample['lowres_input'] = inout[:, :, :3]
    sample['lowres_output'] = inout[:, :, 3:]
    sample['image_input'] = fullres[:, :, :3]
    sample['image_output'] = fullres[:, :, 3:]
    
    return sample
