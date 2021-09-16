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

VERSION=0010
JOB_NAME="aws_seg_shapenet_${VERSION}"
TRAIN_DIR="/home/ec2-user/google-research/tf3d/shapenet/log/train_with_train_data_${JOB_NAME}"



NUM_WORKERS=1
STRATEGY='mirrored'  # set to "multi_worker_mirrored" when NUM_WORKERS > 1
NUM_GPUS=1
BATCH_SIZE=3
LEARNING_RATES=0.2

NUM_STEPS_PER_EPOCH=100
NUM_EPOCHS=100
LOG_FREQ=100

# Data
DATASET_NAME="shapenet"
TRAIN_SPLIT="0"
DATASET_PATH="/home/ec2-user/shapenet_data_tfrecords/val_data_with_labels/02691156" # REPLACE

# Gin config
IMPORT_MODULE="tf3d.gin_imports"
TRAIN_GIN_CONFIG="/home/ec2-user/google-research/tf3d/shapenet/shapenet_train.gin"

PARAMS="get_tf_data_dataset.dataset_name = '${DATASET_NAME}'
get_tf_data_dataset.split_name = '${TRAIN_SPLIT}'
get_tf_data_dataset.dataset_dir = '${DATASET_PATH}'
get_tf_data_dataset.dataset_format = 'tfrecord'
step_decay.initial_learning_rate = ${LEARNING_RATES}"


echo "Deleting TRAIN_DIR at ${TRAIN_DIR}..."
rm -r "${TRAIN_DIR}"

python -m tf3d.train \
  --params="${PARAMS}" \
  --import_module="${IMPORT_MODULE}" \
  --config_file="${TRAIN_GIN_CONFIG}" \
  --train_dir="${TRAIN_DIR}" \
  --num_workers="${NUM_WORKERS}" \
  --num_gpus="${NUM_GPUS}" \
  --run_functions_eagerly=true \
  --num_steps_per_epoch="${NUM_STEPS_PER_EPOCH}" \
  --log_freq="${LOG_FREQ}" \
  --num_epochs="${NUM_EPOCHS}" \
  --distribution_strategy="${STRATEGY}" \
  --batch_size="${BATCH_SIZE}" \
  --alsologtostderr
