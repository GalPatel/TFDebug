# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Debug the tf-learn iris example, based on the tf-learn tutorial."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import tempfile

import numpy as np
from six.moves import urllib
import tensorflow as tf

from tensorflow import keras
from tensorflow.contrib.learn.python.learn import experiment
from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python import debug as tf_debug


def main(_):

  # Build dataset
  x_train = np.random.random_integers(256, size=[120, 28, 28, 3]) - 1
  y_train = np.copy(x_train)

  x_val = np.random.random_integers(256, size=[30, 28, 28, 3]) - 1
  y_val = np.copy(x_val)

  model = keras.Sequential()

  # Encoder
  model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1'))
  model.add(keras.layers.AvgPool2D((2, 2), strides=(2, 2), name='pool1'))
  model.add(keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2'))
  model.add(keras.layers.AvgPool2D((2, 2), strides=(2, 2), name='pool2'))

  for i in range(20):
    model.add(keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv%d'%(3+i)))

  # Decoder
  model.add(keras.layers.Conv2DTranspose(
    64, (3, 3), strides=2, activation='relu', padding='same', name='upsample1'))
  model.add(keras.layers.Conv2DTranspose(
    3, (3, 3), strides=2, activation='relu', padding='same', name='upsample2'))

  # Compile model
  model.compile(optimizer=tf.train.GradientDescentOptimizer(learning_rate=1e-3),
                loss='mean_squared_error')


  # Debug
  if FLAGS.debug and FLAGS.tensorboard_debug_address:
    raise ValueError(
        "The --debug and --tensorboard_debug_address flags are mutually "
        "exclusive.")

  hooks = None
  if FLAGS.debug:
    tf.keras.backend.set_session(tf_debug.LocalCLIDebugWrapperSession(tf.Session()))
    #debug_hook = tf_debug.LocalCLIDebugHook(ui_type=FLAGS.ui_type, dump_root=FLAGS.dump_root)
    #hooks = [debug_hook]
  
  elif FLAGS.tensorboard_debug_address:
    debug_hook = tf_debug.TensorBoardDebugHook(FLAGS.tensorboard_debug_address)
    hooks = [debug_hook]

  # Fit model.
  history = model.fit(x_train,
                      y_train,
                      epochs=FLAGS.epochs,
                      batch_size=32,
                      validation_data=(x_val, y_val),
                      verbose=1)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")
  parser.add_argument(
      "--epochs",
      type=int,
      default=40,
      help="Number of steps to run trainer.")
  parser.add_argument(
      "--debug",
      type="bool",
      nargs="?",
      const=True,
      default=False,
      help="Use debugger to track down bad values during training. "
      "Mutually exclusive with the --tensorboard_debug_address flag.")
  parser.add_argument(
      "--tensorboard_debug_address",
      type=str,
      default=None,
      help="Connect to the TensorBoard Debugger Plugin backend specified by "
      "the gRPC address (e.g., localhost:1234). Mutually exclusive with the "
      "--debug flag.")
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
