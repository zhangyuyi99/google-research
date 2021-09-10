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

#!/bin/bash

VERSION=004
JOB_NAME="seg_shapenet_${VERSION}"
EVAL_DIR="${JOB_NAME}"
CKPT_DIR="/u/yvu2cv/google-research/tf3d/shapenet/log/${EVAL_DIR}/model"

NUM_STEPS_PER_EPOCH=100
LOG_FREQ=100

# Data
DATASET_NAME='shapenet'
EVAL_SPLIT='0'
DATASET_PATH="/u/yvu2cv/google-research/tf3d/data/shapenet_data_tfrecords/val_data_with_labels/02691156" # REPLACE

# Gin config
IMPORT_MODULE='tf3d.gin_imports'
EVAL_GIN_CONFIG="/u/yvu2cv/google-research/tf3d/shapenet/shapenet_eval.gin"

PARAMS="get_tf_data_dataset.dataset_name = '${DATASET_NAME}'
get_tf_data_dataset.dataset_dir = '${DATASET_PATH}'
get_tf_data_dataset.dataset_format = 'tfrecord'
get_tf_data_dataset.split_name = '${EVAL_SPLIT}'
"

echo "EVAL_DIR at ${EVAL_DIR}..."

python -m tf3d.eval \
  --params="${PARAMS}" \
  --import_module="${IMPORT_MODULE}" \
  --config_file="${EVAL_GIN_CONFIG}" \
  --eval_dir="${EVAL_DIR}" \
  --ckpt_dir="${CKPT_DIR}" \
  --run_functions_eagerly=false \
  --num_steps_per_epoch="${NUM_STEPS_PER_EPOCH}" \
  --num_steps_per_log="${LOG_FREQ}" \
  --alsologtostderr
