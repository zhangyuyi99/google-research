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

"""Shapenet dataset."""

import os
import gin
import gin.tf
import tensorflow as tf
import tensorflow_datasets as tfds

from tf3d.datasets.specs import siemens_specs
from tf3d.datasets.utils import example_parser
from tf3d.utils import box_utils


_FILE_PATTERN = '%s*.pts'
_FILE_PATTERN_TFRECORD = '%s*.tfrecords'

DATASET_DIR = "/u/yvu2cv/google-research/tf3d/data/shapenet_data/train_data/02691156"


DATASET_FORMAT = 'pts'

# def _get_feature_label_keys():
#   """Extracts and returns the dataset feature and label keys."""
#   feature_spec = (
#       siemens_specs.pointcloud_feature_spec(
#           with_annotations=True).get_serialized_info())
#   feature_dict = tfds.core.utils.flatten_nest_dict(feature_spec)
#   feature_keys = []
#   label_keys = []
#   for key in sorted(feature_dict):
#     if 'labels' in key:
#       label_keys.append(key)
#     else:
#       feature_keys.append(key)
#   return feature_keys, label_keys
#
#
# def get_feature_keys():
#   return _get_feature_label_keys()[0]
#
#
# def get_label_keys():
#   return _get_feature_label_keys()[1]


def get_file_pattern(split_name,
                     dataset_dir=DATASET_DIR,
                     dataset_format=DATASET_FORMAT):
  if dataset_format == DATASET_FORMAT:
    return os.path.join(dataset_dir, _FILE_PATTERN % split_name)
  elif dataset_format == 'tfrecord':
    return os.path.join(dataset_dir, _FILE_PATTERN_TFRECORD % split_name)


def get_decode_fn():
  """Returns a tfds decoder.

  Returns:
    A tf.data decoder.
  """

  # def decode_fn(value):
  #   tensors = example_parser.decode_serialized_example(
  #       serialized_example=value,
  #       features=siemens_specs.pointcloud_feature_spec(with_annotations=True))
  #   tensor_dict = tfds.core.utils.flatten_nest_dict(tensors)
  #   return tensor_dict

  def decode_fn(value):

    data = tf.io.read_file(value)

    data_string = data.numpy()
    data_string = data_string[:-1]

    data_string = tf.strings.split(data_string, sep="\n")
    data_string = tf.strings.split(data_string, sep=" ")
    # print(data_string)
    data = tf.strings.to_number(data_string, out_type=tf.dtypes.float32, name=None)

    # print(data)
    return data

###################################################
    # print('***************decode_fn')
    # print(value)
    #
    # import io
    #
    # f = io.StringIO()
    # print(value, file=f)
    #
    # # to get the value back
    # print_value = f.getvalue()
    # f.close()
    #
    # print_value = print_value[print_value.index("'") +1::]
    # path = print_value[:print_value.index("'"):]
    #
    # print(path)
#####################################################


    # path = "/u/yvu2cv/google-research/tf3d/data/shapenet_data/val_data/02691156/012137.pts"
    # print('************************')
    # print(tf.executing_eagerly())
    # path = bytes.decode(value.numpy())

    # with tf.compat.v1.Session() as sess:
    #     path = bytes.decode(value.eval())
    # # path = value.numpy().decode('utf-8')
    # with open(path, 'r') as f:
    #     lines = [line.split() for line in f]
    #     data = np.array(lines)
    #     # data = reshape(data, (batch_size, ,3))
    #     # print(data)
    #     print(data.shape)

    # # label_path = "/u/yvu2cv/google-research/tf3d/data/shapenet_data/val_label/02691156/012137.seg"
    # # with open(label_path, 'r') as f:
    # #     lines = [line.split() for line in f]
    # #     # lines = f.split()
    # #     label = np.array(lines)
    # #     # print(label)
    # #     print(label.shape)
    #
    # # from tf3d import train
    # from tf3d import standard_fields
    # # from tf3d.semantic_segmentation import model
    #
    # num_classes = 4
    #     # voxel_inputs = (inputs[standard_fields.InputDataFields.voxel_features],
    #     #                 inputs[standard_fields.InputDataFields.voxel_xyz_indices],
    #     #                 inputs[standard_fields.InputDataFields.num_valid_voxels]
    #
    # # voxel_features: A tf.float32 tensor of size [b, n, f] where b is batch size, n is the number of voxels and f is the feature size.
    # # voxel_indices: A tf.int32 tensor of size [b, n, 3].
    # # num_valid voxels: A tf.int32 tensor of size [b] containing number of valid voxels in each of the batch examples.
    #
    # inputs = dict()
    # num_valid_voxels = data.shape[0]
    # print(num_valid_voxels)
    # inputs[standard_fields.InputDataFields.voxel_features] = np.reshape(label,(batch_size,num_valid_voxels,1))
    # inputs[standard_fields.InputDataFields.voxel_xyz_indices] = np.reshape(data,(batch_size,num_valid_voxels,3))
    # inputs[standard_fields.InputDataFields.num_valid_voxels] = [num_valid_voxels]
    # print(inputs)

    # dataset = tf.data.Dataset.from_tensor_slices(inputs)
    # dataset = dataset.batch(batch_size=3)
    # return tf.data.Dataset.from_tensor_slices(inputs)

    data = np.reshape(data,(batch_size,num_valid_voxels,3))

    tensor_dict = tf.convert_to_tensor(data, dtype=tf.float32)

    return tensor_dict

  return decode_fn
