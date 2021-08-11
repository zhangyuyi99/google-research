# Install Tensorflow3D on CentOS 7
Refer to https://github.com/google-research/google-research/blob/519ccab2b20768783034739611fed68cdee63570/tf3d/doc/setup.md 
1. Install packages:
```bash
sudo yum update
sudo yum install subversion git
```

2. Install CUDA 10.1, nvidia driver and cuDNN 7.6.5:
	
2.1. For CUDA 10.1, follow https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html.
  
Note:  
i. 
```bash
sudo yum install kernel-devel-$(uname -r) kernel-headers-$(uname -r)
```
does not work for me. I emitted the uname to get it work:
```bash
sudo yum install kernel-devel kernel-headers
```
ii. yum sometimes fails to find an available package. Google and specify a downloading url will solve this.
	
iii. Skip the 9.1.2. POWER9 Setup in post-installation actions. According to google, we are unlikely to have a POWER9 unless we are using some super computer…
	
iv. I skipped all the recommended actions
	
2.2. For nvidia driver, follow https://docs.nvidia.com/datacenter/tesla/tesla-installation-notes/index.html
  
Note:  
i. do not use .run file to install driver. Package manager method worked for me.
              
ii. 
```bash
sudo dnf install -y tar bzip2 make automake gcc gcc-c++ pciutils elfutils-libelf-devel libglvnd-devel iptables firewalld vim bind-utils wget
```
I used yum instead of dnf. 

2.3. For cuDNN 7.6.5, follow https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html 
	
	
3. Install miniconda according to:
	https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html
	
  
4. Create conda environment:
```bash
source {$PATH TO YOUR MINICONDA}/etc/profile.d/conda.sh
conda create --name tf3d
conda activate tf3d
```
	
  
5. Within the environment tf3d, install the dependencies:
```bash
pip install -r tf3d/requirements.txt
sudo yum update
sudo yum install protobuf-compiler python3-dev
```


6. Now we prepare the sparse conv ops as well as set up the nvidia-docker:
	
6.1.  change ${TF_FOLDER} and ${CUSTOM_OP_FOLDER} into your local address of cloned tensorflow and custom-op projects
```bash
git clone https://github.com/tensorflow/tensorflow
# Or download https://github.com/tensorflow/tensorflow/archive/v2.3.0.zip and unzip.
cd tensorflow && git checkout v2.3.0 && cd ..
git clone https://github.com/tensorflow/custom-op --depth=1
export TF_FOLDER="PATH_TO_TF_REPO_FOLDER"
export CUSTOM_OP_FOLDER="PATH_TO_CUSTOM_OP_REPO_FOLDER"
mkdir -p tf3d/ops/third_party

cp -a ${TF_FOLDER}/third_party/eigen3 ${TF_FOLDER}/third_party/mkl \
${TF_FOLDER}/third_party/toolchains ${TF_FOLDER}/third_party/BUILD \
${TF_FOLDER}/third_party/eigen.BUILD \
${TF_FOLDER}/third_party/com_google_absl_fix_mac_and_nvcc_build.patch \
${TF_FOLDER}/third_party/com_google_absl.BUILD \
${TF_FOLDER}/third_party/cub.BUILD ${TF_FOLDER}/third_party/repo.bzl \
tf3d/ops/third_party/
cp -a ${CUSTOM_OP_FOLDER}/gpu ${CUSTOM_OP_FOLDER}/tf \
${CUSTOM_OP_FOLDER}/configure.sh tf3d/ops/
```
	
6.2.  set up the nvidia-docker following:
	      https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
	         
6.3.  Enter the docker image with mapped tf3d folder, change ${PATH_TO_FOLDER_WITH_TF3D} according to your local folders
```bash
docker pull tensorflow/tensorflow:2.3.0-custom-op-gpu-ubuntu16
sudo docker run --runtime=nvidia --privileged -it -v ${PATH_TO_FOLDER_WITH_TF3D}:Z -w /working_dir  tensorflow/tensorflow:2.3.0-custom-op-gpu-ubuntu16
```
		
Note: 
i. the content in ${PATH_TO_FOLDER_WITH_TF3D} will be mounted into working_dir in the docker container
                  
ii. If you see "Cannot connect to the Docker daemon at tcp://localhost:2375. Is the docker daemon running?" and none of solutions on stackoverflow works, try:
```bash
sudo docker ps
unset DOCKER_HOST
```
Modify ~/.bashrc file
```bash
vi ~/.bashrc
# Add a line at the bottom:
export DOCKER_HOST='unix:///var/run/docker.sock'
source .bashrc
```
Refer to: https://www.programmersought.com/article/2584623911/      
	
6.4. Within the docker image, enter tf3d/ops folder and run the following to test the building:
```bash	
# Make sure you are using tensorflow version to 2.3.0
pip3 uninstall tensorflow
pip3 install tensorflow==2.3.0
./configure.sh
bazel run sparse_conv_ops_py_test  --experimental_repo_remote_exec --verbose_failures
# For compiling with CPU only, please select tensorflow CPU during running configure.sh.
# If configure.sh replaces tensorflow with tensorflow-cpu, please install tensorflow==2.3.0 again before proceeding to the bazel command to avoid missing header files.
```
Note: 
i. if you run into "#error This file requires compiler and library support for the ISO C++ 2011 standard. This support must be enabled with the -std=c++11 or -std=gnu++11 compiler options.", try:
```bash
bazel run sparse_conv_ops_py_test  --experimental_repo_remote_exec --verbose_failures --cxxopt=-std=c++11
```
			
ii. If you run into "undefined symbol: _ZN10tensorflow8OpKernel11TraceStringEPNS_15OpKernelContextEb", first check the tensorflow was installed using pip or pip3 and the version is 2.3.0. 
		If the error still shows up, obtain path to tensorflow lib by:
```bash
pip show tensorflow
```			
		My tensorflow is installed in /usr/local/lib/python3.6/dist-packages/tensorflow. Change ${PATH_TO_TF_LIB} to your tensorflow location and try:
```bash			
bazel run sparse_conv_ops_py_test  --experimental_repo_remote_exec --verbose_failures --cxxopt=-std=c++11 --cxxopt="-L ${PATH_TO_TF_LIB}" --cxxopt="-ltensorflow_framework" --cxxopt="-O2" --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0"
```
			
iii. If you are using GPU with computation capability <3.5, add the following line to the above bazel command:
```bash
--copt=-DTF_EXTRA_CUDA_CAPABILITIES=3.0
```
	  
6.5. After the test succeeds, copy the shared library to tf3d/ops/tensorflow_sparse_conv_ops folder:
```bash
cp -a bazel-bin/tensorflow_sparse_conv_ops/_sparse_conv_ops.so tensorflow_sparse_conv_ops/
```
If you see permission denied or operation not permitted error, exit the docker image and do chmod files locally, as the docker inherits the local permission. 
	
6.6. Exit the docker image.
	  Enter the parent folder containing the tf3d folder, the sparse conv ops can be imported as follows:
```python
import tf3d.ops.tensorflow_sparse_conv_ops as sparse_conv_ops
```

