# Train example neural network on Waymo data
Refer to https://github.com/google-research/google-research/blob/master/tf3d/doc/usage.md
```bash
bash tf3d/semantic_segmentation/scripts/waymo/run_train_locally.sh
```
Note: 
	 
i. If you encounter "ImportError: cannot import name 'string_int_label_map_pb2'", navigate to /tf3d/utils/protos/ and run:
```bash
protoc string_int_label_map.proto --python_out=.
```		
If you see "ModuleNotFoundError: No module named 'tensorflow_sparse_conv_ops'", open the file and change tensorflow_sparse_conv_ops into tf3d.ops.tensorflow_sparse_conv_ops
		
If you see "ImportError: This version of TensorFlow Probability requires TensorFlow version >= 2.5; Detected an installation of version 2.3.0. Please upgrade TensorFlow to proceed.", try:
```bash
pip install tensorflow-probability==0.10.0
```
	
If you see "ModuleNotFoundError: No module named 'gin'", try:
```bash
conda install -c conda-forge gin-config
```
		
If you see "ImportError: No module named 'tensorflow_datasets'", try:
```bash
conda install -c anaconda tensorflow-datasets
```
	           
If you see "ModuleNotFoundError: No module named 'object_detection'", try:
```bash
		pip install tensorflow-object-detection-api
```	
…and many other ModuleNotFoundError, just install the required modules
	
ii. Download the dataset from https://console.cloud.google.com/storage/browser/waymo_open_dataset_tf_example_tf3d?pageState=(%22StorageObjectListTable%22:(%22f%22:%22%255B%255D%22))&prefix=&forceOnObjectsSortingFiltering=false or with command:
```bash
gsutil -m cp -r "gs://waymo_open_dataset_tf_example_tf3d/original_tfrecords/" .
```	
ps. The command line did not work for me on windows. On Linux I cannot log in google cloud sdk with terminal. 
	
Change the dataset path and split in tf3d/semantic_segmentation/scripts/waymo/run_train_locally.sh accordingly and run.
