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
        'position': tfds.features.Tensor(shape=(None, 3), dtype=tf.float32),
        # Pointcloud labesl (Nx1).
        'label': tfds.features.Tensor(shape=(None, 1), dtype=tf.int64),
   },
})


# Object category labels.
OBJECT_CATEGORY_LABELS = [1,2,3,4]
