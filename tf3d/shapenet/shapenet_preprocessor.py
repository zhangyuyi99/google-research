import numpy as np
import gin
import gin.tf
import tensorflow as tf

batch_size = 1

@gin.configurable('preprocess_data')
def preprocess_data(is_training=True, batch_size=None):

    path = "/u/yvu2cv/google-research/tf3d/data/shapenet_data/val_data/02691156/012137.pts"

    myTensor = tf.convert_to_tensor(path, dtype=tf.string)

    data = tf.io.read_file(myTensor)

    print(type(data))
    data_string = data.numpy()
    print(type(data_string))
    data_string = data_string[:-1]

    data_string = tf.strings.split(data_string, sep="\n")
    data_string = tf.strings.split(data_string, sep=" ")
    # print(data_string)
    data = tf.strings.to_number(data_string, out_type=tf.dtypes.float32, name=None)
    print(type(data))
    # print(data)

    # with open(path, 'r') as f:
    #     lines = [line.split() for line in f]
    #     data = np.array(lines)
    #     # data = reshape(data, (batch_size, ,3))
    #     # print(data)
    #     print(data.shape)
    #
    # label_path = "/u/yvu2cv/google-research/tf3d/data/shapenet_data/val_label/02691156/012137.seg"
    # with open(label_path, 'r') as f:
    #     lines = [line.split() for line in f]
    #     # lines = f.split()
    #     label = np.array(lines)
    #     # print(label)
    #     print(label.shape)

        # model = binvox_rw.read_as_3d_array(f)

    # categories=["02691156", "02773838", "02954340", "02958343",
    #        "03001627", "03261776", "03467517", "03624134",
    #        "03636649", "03642806", "03790512", "03797390",
    #        "03948459", "04099429", "04225987", "04379243"]
    # classes=['Airplane', 'Bag',      'Cap',        'Car',
    #          'Chair',    'Earphone', 'Guitar',     'Knife',
    #          'Lamp',     'Laptop',   'Motorbike',  'Mug',
    #          'Pistol',   'Rocket',   'Skateboard', 'Table']
    # nClasses=[4, 2, 2, 4, 4, 3, 3, 2, 4, 2, 6, 2, 3, 3, 3, 3]
    # classOffsets=np.cumsum([0]+nClasses)
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
    #
    # dataset = tf.data.Dataset.from_tensor_slices(inputs)
    # dataset = dataset.batch(batch_size=3)
    # return tf.data.Dataset.from_tensor_slices(inputs)

preprocess_data()
    # dataset_dict = tf.data.Dataset.batch(
    #       dataset.map(
    #           _process_fn, num_parallel_calls=num_parallel_batches),
    #       batch_size=batch_size,
    #       drop_remainder=True)
    # dataset_dict = dataset_dict.prefetch(num_prefetch_batches)

# dataset = tf.data.Dataset.list_files("/u/yvu2cv/google-research/tf3d/data/shapenet_data/train_data/*/*.pts")
# print(dataset)
# for element in dataset:
#     print(element)
