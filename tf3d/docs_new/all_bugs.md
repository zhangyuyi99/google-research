# tf3d Toubleshooting
exploration on tensorflow 3d.

Listed are errors in installation and running codes. 

## tensorflow.python.framework.errors_impl.InvalidArgumentError: Need minval < maxval, got 0 >= 0

<details>
<summary>error message</summary>
  
```bash
I0806 10:23:38.867532 139741489559360 train.py:117] Model fit starting for 100 epochs, 100 step per epoch, total batch size:2
I0806 10:23:39.385184 139741489559360 callback_utils.py:370] Saving ckpt for epoch: 0 at /tmp/tf3d_experiment/seg_siemens_001/model
Epoch 1/100
Traceback (most recent call last):
  File "/u/yvu2cv/miniconda3/envs/tf3d/lib/python3.6/runpy.py", line 193, in _run_module_as_main
    "__main__", mod_spec)
  File "/u/yvu2cv/miniconda3/envs/tf3d/lib/python3.6/runpy.py", line 85, in _run_code
    exec(code, run_globals)
  File "/sdl/assem/users/yvu2cv/tf3d/train.py", line 222, in <module>
    app.run(main)
  File "/u/yvu2cv/miniconda3/envs/tf3d/lib/python3.6/site-packages/absl/app.py", line 303, in run
    _run_main(main, args)
  File "/u/yvu2cv/miniconda3/envs/tf3d/lib/python3.6/site-packages/absl/app.py", line 251, in _run_main
    sys.exit(main(argv))
  File "/sdl/assem/users/yvu2cv/tf3d/train.py", line 177, in main
    train(strategy=strategy, write_path=write_path)
  File "/u/yvu2cv/miniconda3/envs/tf3d/lib/python3.6/site-packages/gin/config.py", line 1069, in gin_wrapper
    utils.augment_exception_message_and_reraise(e, err_str)
  File "/u/yvu2cv/miniconda3/envs/tf3d/lib/python3.6/site-packages/gin/utils.py", line 41, in augment_exception_message_and_reraise
    raise proxy.with_traceback(exception.__traceback__) from None
  File "/u/yvu2cv/miniconda3/envs/tf3d/lib/python3.6/site-packages/gin/config.py", line 1046, in gin_wrapper
    return fn(*new_args, **new_kwargs)
  File "/sdl/assem/users/yvu2cv/tf3d/train.py", line 124, in train
    verbose=1 if FLAGS.run_functions_eagerly else 2)
  File "/u/yvu2cv/miniconda3/envs/tf3d/lib/python3.6/site-packages/tensorflow/python/keras/engine/training.py", line 108, in _method_wrapper
    return method(self, *args, **kwargs)
  File "/u/yvu2cv/miniconda3/envs/tf3d/lib/python3.6/site-packages/tensorflow/python/keras/engine/training.py", line 1098, in fit
    tmp_logs = train_function(iterator)
  File "/u/yvu2cv/miniconda3/envs/tf3d/lib/python3.6/site-packages/tensorflow/python/keras/engine/training.py", line 806, in train_function
    return step_function(self, iterator)
  File "/u/yvu2cv/miniconda3/envs/tf3d/lib/python3.6/site-packages/tensorflow/python/keras/engine/training.py", line 795, in step_function
    data = next(iterator)
  File "/u/yvu2cv/miniconda3/envs/tf3d/lib/python3.6/site-packages/tensorflow/python/distribute/input_lib.py", line 649, in __next__
    return self.get_next()
  File "/u/yvu2cv/miniconda3/envs/tf3d/lib/python3.6/site-packages/tensorflow/python/distribute/input_lib.py", line 694, in get_next
    self._iterators[i].get_next_as_list_static_shapes(new_name))
  File "/u/yvu2cv/miniconda3/envs/tf3d/lib/python3.6/site-packages/tensorflow/python/distribute/input_lib.py", line 1474, in get_next_as_list_static_shapes
    return self._iterator.get_next()
  File "/u/yvu2cv/miniconda3/envs/tf3d/lib/python3.6/site-packages/tensorflow/python/data/ops/multi_device_iterator_ops.py", line 581, in get_next
    result.append(self._device_iterators[i].get_next())
  File "/u/yvu2cv/miniconda3/envs/tf3d/lib/python3.6/site-packages/tensorflow/python/data/ops/iterator_ops.py", line 825, in get_next
    return self._next_internal()
  File "/u/yvu2cv/miniconda3/envs/tf3d/lib/python3.6/site-packages/tensorflow/python/data/ops/iterator_ops.py", line 764, in _next_internal
    return structure.from_compatible_tensor_list(self._element_spec, ret)
  File "/u/yvu2cv/miniconda3/envs/tf3d/lib/python3.6/contextlib.py", line 99, in __exit__
    self.gen.throw(type, value, traceback)
  File "/u/yvu2cv/miniconda3/envs/tf3d/lib/python3.6/site-packages/tensorflow/python/eager/context.py", line 2105, in execution_mode
    executor_new.wait()
  File "/u/yvu2cv/miniconda3/envs/tf3d/lib/python3.6/site-packages/tensorflow/python/eager/executor.py", line 67, in wait
    pywrap_tfe.TFE_ExecutorWaitForAllPendingNodes(self._handle)
tensorflow.python.framework.errors_impl.InvalidArgumentError: Need minval < maxval, got 0 >= 0
         [[{{node random_uniform}}]]
         [[MultiDeviceIteratorGetNextFromShard]]
         [[RemoteCall]]
  In call to configurable 'train' (<function train at 0x7f18128d1048>)
```
  
</details>
  
## TypeError: Input 'y' of 'Mul' Op has type float32 that does not match type int32 of argument 'x'.

**Solution** 
  
`tf.cast(y, tf.float32)`
  
## ImportError: cannot import name 'string_int_label_map_pb2'
  
**Solution** 
  
`protoc tf3d/utils/protos/*.proto --python_out=.`
 
 from: https://github.com/tensorflow/models/issues/1595
  
## Convert windows text file to linux
  
**Solution** 
  
 `dos2unix filename`

## ModuleNotFoundError: No module named 'tensorflow_sparse_conv_ops'

**Solution**
  
Change `from tensorflow_sparse_conv_ops import sparse_conv_ops` into `from tf3d.ops.tensorflow_sparse_conv_ops import sparse_conv_ops`
  
## NVIDIA NVML Driver/library version mismatch 
## failed call to cuInit: CUDA_ERROR_NO_DEVICE: no CUDA-capable device is detected
  
When type `nvidia-smi`. Try uninstalling the nvidia GPU driver by running `/usr/bin/nvidia-uninstall`.
  
from: https://stackoverflow.com/questions/43022843/nvidia-nvml-driver-library-version-mismatch and https://partners-intl.aliyun.com/help/doc-detail/109111.htm
  
## [AWS] WARNING: UNPROTECTED PRIVATE KEY FILE! at log in 
  
**Solution** 

`chmod  400 ~/.ssh/id_rsa`
  
## ImportError: libcudnn.so.7: cannot open shared object file: No such file or directory
  
**Solution** 

```bash
export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```
  
## Could not load dynamic library 'libcudart.so.10.1'; dlerror: libcudart.so.10.1: cannot open shared object file: No such file or directory
  
**Solution** 
  
Double check that you have CUDA10.1 installed. 
  
## Python 3.6 not compatible with Tensorflow 2.3.0, need higher python version 
  
**Solution** 
  
`pip install tensorflow-gpu==2.3.0`
  
 



  


