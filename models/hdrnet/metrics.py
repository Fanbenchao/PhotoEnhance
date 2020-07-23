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
"""Useful image metrics."""

import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import sys
sys.path.append('/home/lupin/PhotoEnhance/hdrnet_lupin/hdrnet')
from functools import reduce
# import vgg

def tensor_size(tensor):
    from operator import mul
    A = tensor.get_shape()[1:]
    return reduce(mul, (d for d in tensor.get_shape()), 1)
def l2_loss(target, prediction, name=None):
  with tf.compat.v1.keras.backend.name_scope(name, default_name='l2_loss', values=[target, prediction]):
    loss = tf.reduce_mean(tf.square(target-prediction))
  return loss
def loss(target,heatmap,prediction,bilateral_coeffs,vgg_pretrained,name = None):
    with tf.compat.v1.keras.backend.name_scope(name, default_name='loss', values=[target, prediction]):
        heatmap = tf.expand_dims(heatmap,-1)
        mse_loss = tf.reduce_mean(tf.square(target-prediction)*heatmap)
        total_loss = mse_loss
        return total_loss
        

def psnr(target, prediction, name=None):
  with tf.compat.v1.keras.backend.name_scope(name, default_name='psnr_op', values=[target, prediction]):
    squares = tf.square(target-prediction, name='squares')
    squares = tf.reshape(squares, [tf.shape(squares)[0], -1])
    # mean psnr over a batch
    p = tf.reduce_mean((-10/np.log(10))*tf.math.log(tf.reduce_mean(squares, axis=[1])))
  return p
