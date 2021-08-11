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

"""Siemens pointcloud data feature specifications."""

import tensorflow as tf
import tensorflow_datasets as tfds


def pointcloud_feature_spec(with_annotations=True):
  """Feature specification of pointcloud data.

  Args:
    with_annotations: If true semantic labels for points are
      also present. This is the default (True) for training data.

  Returns:
    Feature specification (tfds) for a single pointcloud data.
  """
  return tfds.features.FeaturesDict({

    # 3D pointcloud data.
    'pointcloud': {
        # Pointcloud positions (Nx3).
        'position': tfds.features.Tensor(shape=(None, 3), dtype=tf.int64),
        # Pointcloud intensity (Nx1).
        'intensity': tfds.features.Tensor(shape=(None, 1), dtype=tf.int64),
        # Pointcloud labesl (Nx1).
        'labels': tfds.features.Tensor(shape=(None, 1), dtype=tf.int64),
   },
})


# Object category labels.
OBJECT_CATEGORY_LABELS = ['2', '6']