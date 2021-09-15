# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

"""Runs qualitative evaluation given a model and saved checkpoint."""
import time
from absl import app
from absl import flags
from absl import logging

import gin
import gin.tf
import six
import tensorflow as tf

from tf3d.utils import callback_utils
from tf3d.semantic_segmentation import preprocessor

FLAGS = flags.FLAGS

flags.DEFINE_multi_string('import_module', None, 'List of modules to import.')

flags.DEFINE_string('master', '', 'BNS name of the TensorFlow master to use.')

flags.DEFINE_string('eval_dir', '/tmp/masternet/',
                    'Directory where to write event logs.')

flags.DEFINE_string('ckpt_dir', '/tmp/masternet/',
                    'Directory where to load checkpoint.')

flags.DEFINE_string('config_file', None, 'The path to the config file.')

flags.DEFINE_string('split', 'val', 'The data split to evaluate on.')

flags.DEFINE_multi_string('params', None,
                          'Newline separated list of Gin parameter bindings.')

flags.DEFINE_bool('run_functions_eagerly', False,
                  'Run function eagerly for easy debugging.')

flags.DEFINE_integer(
    'num_steps_per_epoch', 1000,
    'Number of batches to train before saving the model weights again. The next'
    'epoch will continue loading the data stream from where the current epoch'
    'left behind. Used for calculating actual ckpt step number during eval.')

flags.DEFINE_integer(
    'num_steps_per_log', 100,
    'Number of steps to log the eval progress.')


@gin.configurable('evaluation')
def evaluation(model_class=None,
               input_fn=None,
               num_quantitative_examples=1000,
               num_qualitative_examples=50):
  """A function that build the model and eval quali."""

  tensorboard_callback = callback_utils.CustomTensorBoard(
      log_dir=FLAGS.eval_dir,
      batch_update_freq=1,
      split=FLAGS.split,
      num_qualitative_examples=num_qualitative_examples,
      num_steps_per_epoch=FLAGS.num_steps_per_epoch)
  model = model_class()
  val_inputs = input_fn(is_training=False, batch_size=1)
  # build the model before loading it with the checkpoint
  # model.fit(
  #   x=val_inputs,
  #   # callbacks=[backup_checkpoint_callback, checkpoint_callback],
  #   steps_per_epoch=1,
  #   epochs=1,
  #   verbose=1)
  # model.summary()
  checkpoint = tf.train.Checkpoint(
      model=model,
      ckpt_saved_epoch=tf.Variable(initial_value=-1, dtype=tf.int64))
  num_evauated_epoch = -1


  ckpt_path = tf.train.latest_checkpoint(FLAGS.ckpt_dir)
  if ckpt_path:
    ckpt_num_of_epoch = int(ckpt_path.split('/')[-1].split('-')[-1])
    if num_evauated_epoch == ckpt_num_of_epoch:
      logging.info('Found old epoch %d ckpt, skip and will check later.',
                   num_evauated_epoch)
      time.sleep(30)
    try:
      logging.info('Restoring new checkpoint[epoch:%d] at %s',
                   ckpt_num_of_epoch, ckpt_path)
      checkpoint.restore(ckpt_path).expect_partial()
    except tf.errors.NotFoundError:
      logging.info('Restoring from checkpoint has failed. Maybe file missing.'
                   'Try again now.')
      time.sleep(3)
  else:
    logging.info('No checkpoint found at %s, will check again 10 s later..',
                 FLAGS.ckpt_dir)
    time.sleep(10)

  tensorboard_callback.set_epoch_number(ckpt_num_of_epoch)
  logging.info('Start qualitative eval for %d steps...',
               num_quantitative_examples)

  data_file = '/u/yvu2cv/google-research/tf3d/data/shapenet_data/train_data/02691156/000001.pts'
  label_file = '/u/yvu2cv/google-research/tf3d/data/shapenet_data/train_label/02691156/000001.seg'
  f_data = open(data_file, 'r')
  f_label = open(label_file, 'r')
  position = f_data.read()[:-1].split('\n')
  position = [i.split(' ') for i in position]
  # position = [float(y) for x in position for y in x]
  position = [[float(y) for y in x] for x in position]

  label = f_label.read()[:-1].split('\n')
  label = list(map(int, label))

  f_data.close()
  f_label.close()

  import numpy as np
  predict_inputs = {
    'pointcloud/position': tf.convert_to_tensor(np.array(position), dtype=tf.float32),
    'pointcloud/label': tf.convert_to_tensor(np.array(label).reshape((len(label),1)), dtype=tf.int32),
  }
  print('********************predict_inputs')
  print(predict_inputs)
  predict_inputs = preprocessor.preprocess(predict_inputs)
  for key,value in predict_inputs.items():
    if value.shape.ndims == 2:
      predict_inputs[key] = tf.reshape(value,(1,value.shape[0],value.shape[1]))
    elif value.shape.ndims == 1:
      predict_inputs[key] = tf.reshape(value,(1,value.shape[0]))

  print('********************preprocessed_predict_inputs')
  print(predict_inputs)
  predictions = model(predict_inputs, training=False)
  print('********************predictions')
  print(predictions)

  point_predictions = predictions['predicted_object_semantic_points']

  print('********************point_predictions')
  print(point_predictions)
  # point_predictions = tf.strings.as_string(point_predictions)
  # print('********************point_predictions converted to string tensor')
  # print(point_predictions)
  # point_predictions = tf.cast(point_predictions, tf.strings)
  filename = '/u/yvu2cv/google-research/tf3d/shapenet/train_000001_point_prediction.txt'
  # tf.io.write_file(filename, point_predictions.numpy(), name=None)
  # np.savetxt(filename, point_predictions.numpy())
  import numpy as np
  np.save("/u/yvu2cv/google-research/tf3d/shapenet/train_000001_point_prediction.npy",point_predictions.numpy())
  b = np.load("/u/yvu2cv/google-research/tf3d/shapenet/train_000001_point_prediction.npy")
  print('********************loaded np array')
  print(b)
  # with open(filename, 'w') as f:
  #   f.write(point_predictions.numpy())

  # predictions = model.predict(val_inputs)
  # print(predictions)

  # try:
  #   # TODO(huangrui): there is still possibility of crash due to
  #   # not found ckpt files.
  #   model._predict_counter.assign(0)  # pylint: disable=protected-access
  #   tensorboard_callback.set_model(model)
  #   tensorboard_callback.on_predict_begin()
  #   for i, inputs in enumerate(
  #       val_inputs.take(num_quantitative_examples), start=1):
  #     tensorboard_callback.on_predict_batch_begin(batch=i)
  #     outputs = model(inputs, training=False)
  #     model._predict_counter.assign_add(1)  # pylint: disable=protected-access
  #     tensorboard_callback.on_predict_batch_end(
  #         batch=i, logs={'outputs': outputs, 'inputs': inputs})
  #     if i % FLAGS.num_steps_per_log == 0:
  #       logging.info('eval progress %d / %d...', i, num_quantitative_examples)
  #   tensorboard_callback.on_predict_end()
  #
  #   num_evauated_epoch = ckpt_num_of_epoch
  #   logging.info('Finished eval for epoch %d, sleeping for :%d s...',
  #                num_evauated_epoch, 100)
  #   time.sleep(100)
  # except tf.errors.NotFoundError:
  #   logging.info('Restoring from checkpoint has failed. Maybe file missing.'
  #                'Try again now.')
  #   continue


def main(argv):
  del argv
  # Import modules BEFORE running Gin.
  if FLAGS.import_module:
    for module_name in FLAGS.import_module:
      __import__(module_name)

  # First, try to parse from a config file.
  if FLAGS.config_file:
    bindings = None
    if bindings is None:
      with tf.io.gfile.GFile(FLAGS.config_file) as f:
        bindings = f.readlines()
    bindings = [six.ensure_str(b) for b in bindings if b.strip()]
    gin.parse_config('\n'.join(bindings))

  if FLAGS.params:
    gin.parse_config(FLAGS.params)

  if FLAGS.run_functions_eagerly:
    tf.config.experimental_run_functions_eagerly(True)

  if not tf.io.gfile.exists(FLAGS.eval_dir):
    tf.io.gfile.makedirs(FLAGS.eval_dir)

  evaluation()


if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)
  app.run(main)
