> ![@老徐](http://oyiztpjzn.bkt.clouddn.com/avatar.png)老徐
>
> Thursday, 26 July 2018

#  Tensorflow 1.8 macOS GPU Install

> The Tensorflow team announced that it will stop supporting the tensorflow gpu version of the Mac version after 1.2.
>
> Therefore, there is no way to install directly and only compile with the source code.
>
> Tensorflow 1.8 with CUDA on macOS High Sierra 10.13.6

The CPU running Tensorflow doesn't feel fast enough, I want to try GPU acceleration! Just have a graphics card that supports CUDA.

![cpu-vs-gpu](https://github.com/SixQuant/tensorflow-macos-gpu/raw/master/res/cpu-vs-gpu.jpg)

## Version

> The important thing to say three times: the relevant driver and compiler environment tools must choose the matching version, otherwise the compilation is not successful!

Version:

- TensorFlow r1.8 source code, the latest 1.9 seems to have problems
- macOS 10.13.6, this should not matter much
- 显卡驱动 387.10.10.10.40.105，支持的 CUDA 9.1
- CUDA 9.2, this is the CUDA driver, which can be higher than the CUDA version supported by the above graphics card, which is CUDA Driver 9.2
- cuDNN 7.2, corresponding to the above CUDA, directly install the latest version
- **XCode 8.2.1** , this is the focus, please downgrade to this version, otherwise it will compile error or run-time error `Segmentation Fault`
- **bazel 0.14.0** , this is the point, please downgrade to this version
- **Python** 3.6 , this is the point, don't use the latest version of Python 3.7

## Links

Need to download (some files need to be downloaded, please start downloading before saving, save time):

- Xcode 8.2.1

  https://developer.apple.com/download/more/

  Xcode_8.2.1.xip

- bazel-0.14.0

  https://github.com/bazelbuild/bazel/releases/download/0.14.0/bazel-0.14.0-installer-darwin-x86_64.sh


- CUDA Toolkit 9.2

  https://developer.nvidia.com/cuda-toolkit-archive

- cuDNN v7.2.1

  https://developer.nvidia.com/rdp/cudnn-download

- Tensorflow source code, 333M

  ```Bash
  $ git clone https://github.com/tensorflow/tensorflow -b r1.8
  ```

### Python 3.6.5_1

If you have Python 3.7 installed, you need to downgrade it:

```bash
$ brew unlink python
$ brew install https://raw.githubusercontent.com/Homebrew/homebrew-core/f2a764ef944b1080be64bd88dca9a1d80130c558/Formula/python.rb
$ pip3 install --upgrade pip setuptools wheel
```

> Do not use Python 3.7.0, otherwise there will be problems with compilation

You can switch back after compiling

```bash
$ brew switch python 3.7.0
```

### Xcode 8.2.1

> Need to downgrade Xcode to 8.2.1

Go to the apple developer's official website to download the package, https://developer.apple.com/download/more/

Unzip and copy to `/Applications/Xcode.app` and then point to

```bash
$ sudo xcode-select -s /Applications/Xcode.app
```

Confirm that the installation is accurate

```bash
$ cc -v
Apple LLVM version 8.0.0 (clang-800.0.42.1)
Target: x86_64-apple-darwin17.7.0
Thread model: posix
InstalledDir: /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin
```

> Command Line Tools, cc is clang
>
> This is very important, otherwise the compilation will be successful but the `Segmentation Fault` will occur when running a complicated project.

## Environmental Variable

> Since lib using CUDA is not in the system directory, you need to set environment variables to point to
>
> LD_LIBRARY_PATH is invalid on Mac, using DYLD_LIBRARY_PATH

Configure environment variables to edit `~/.bash_profile` or `~/.zshrc`

```bash
export CUDA_HOME=/usr/local/cuda
export DYLD_LIBRARY_PATH=$CUDA_HOME/lib:$CUDA_HOME/extras/CUPTI/lib
export PATH=$CUDA_HOME/bin:$PATH
```

## Install CUDA

> CUDA is a parallel computing framework for NVIDIA for its own GPUs, which means that CUDA can only run on NVIDIA GPUs, and only when the computational problem to be solved is a large amount of parallel computing can play the role of CUDA.

### Step 1: Confirm if the graphics card supports GPU computing

> Find your graphics card model here to see if it supports
>
> https://developer.nvidia.com/cuda-gpus

My graphics card is the **NVIDIA GeForce GTX 750 Ti:**

| GPU                                                          | Compute Capability |
| ------------------------------------------------------------ | ------------------ |
| [GeForce GTX 750 Ti](http://www.geforce.com/hardware/desktop-gpus/geforce-gtx-750-ti) | 5.0                |

###  Step 2: Install CUDA

If you have other versions of CUDA installed, you need to uninstall and execute

```bash
$ sudo /usr/local/bin/uninstall_cuda_drv.pl
$ sudo /usr/local/cuda/bin/uninstall_cuda_9.1.pl
$ sudo rm -rf /Developer/NVIDIA/CUDA-9.1/
$ sudo rm -rf /Library/Frameworks/CUDA.framework
$ sudo rm -rf /usr/local/cuda/
```

> In order to be foolproof, it is best to restart it.

The first thing to note is that the CUDA Driver and the GPU Driver must be the same version in order for CUDA to find the graphics card.

* GPU Driver is the graphics driver
    * http://www.macvidcards.com/drivers.html

    * My macOS is 10.13.6 corresponding driver has been installed the latest version `387.10.10.10.40.105`

      https://www.nvidia.com/download/driverResults.aspx/136062/en-us

      ```
      Version:	387.10.10.10.40.105
      Release Date:	2018.7.10
      Operating System:	macOS High Sierra 10.13.6
      CUDA Toolkit:	9.1
      ```
* CUDA Driver 
    * http://www.nvidia.com/object/mac-driver-archive.html

    * Install CUDA Driver separately, you can choose the latest version, see his support for graphics card driver

    * cudadriver_396.148_macos.dmg

      ```
      New Release 396.148
      CUDA driver update to support CUDA Toolkit 9.2, macOS 10.13.6 and NVIDIA display driver 387.10.10.10.40.105
      Recommended CUDA version(s): CUDA 9.2
      Supported macOS 10.13
      ```
* CUDA Toolkit
    * https://developer.nvidia.com/cuda-toolkit

    * You can choose the latest version, here choose 9.2

    * cuda_9.2.148_mac.dmg、cuda_9.2.148.1_mac.dmg


Check after installation is complete:

```bash
$ nvcc -V
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2018 NVIDIA Corporation
Built on Tue_Jun_12_23:08:12_CDT_2018
Cuda compilation tools, release 9.2, V9.2.148
```

Confirm that the driver is loaded

```bash
$ kextstat | grep -i cuda.
  149    0 0xffffff7f838d3000 0x2000     0x2000     com.nvidia.CUDA (1.1.0) E13478CB-B251-3C0A-86E9-A6B56F528FE8 <4 1>
```

Test if CUDA is working properly:

```bash
$ cd /usr/local/cuda/samples
$ sudo make -C 1_Utilities/deviceQuery
$ ./bin/x86_64/darwin/release/deviceQuery
./bin/x86_64/darwin/release/deviceQuery Starting...

 CUDA Device Query (Runtime API) version (CUDART static linking)

Detected 1 CUDA Capable device(s)

Device 0: "GeForce GTX 750 Ti"
  CUDA Driver Version / Runtime Version          9.2 / 9.2
  CUDA Capability Major/Minor version number:    5.0
  Total amount of global memory:                 2048 MBytes (2147155968 bytes)
  ( 5) Multiprocessors, (128) CUDA Cores/MP:     640 CUDA Cores
  GPU Max Clock rate:                            1254 MHz (1.25 GHz)
  Memory Clock rate:                             2700 Mhz
  Memory Bus Width:                              128-bit
  L2 Cache Size:                                 2097152 bytes
  Maximum Texture Dimension Size (x,y,z)         1D=(65536), 2D=(65536, 65536), 3D=(4096, 4096, 4096)
  Maximum Layered 1D Texture Size, (num) layers  1D=(16384), 2048 layers
  Maximum Layered 2D Texture Size, (num) layers  2D=(16384, 16384), 2048 layers
  Total amount of constant memory:               65536 bytes
  Total amount of shared memory per block:       49152 bytes
  Total number of registers available per block: 65536
  Warp size:                                     32
  Maximum number of threads per multiprocessor:  2048
  Maximum number of threads per block:           1024
  Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
  Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)
  Maximum memory pitch:                          2147483647 bytes
  Texture alignment:                             512 bytes
  Concurrent copy and kernel execution:          Yes with 1 copy engine(s)
  Run time limit on kernels:                     Yes
  Integrated GPU sharing Host Memory:            No
  Support host page-locked memory mapping:       Yes
  Alignment requirement for Surfaces:            Yes
  Device has ECC support:                        Disabled
  Device supports Unified Addressing (UVA):      Yes
  Supports Cooperative Kernel Launch:            No
  Supports MultiDevice Co-op Kernel Launch:      No
  Device PCI Domain ID / Bus ID / location ID:   0 / 1 / 0
  Compute Mode:
     < Default (multiple host threads can use ::cudaSetDevice() with device simultaneously) >

deviceQuery, CUDA Driver = CUDART, CUDA Driver Version = 9.2, CUDA Runtime Version = 9.2, NumDevs = 1
Result = PASS
```

> If Result = PASS is finally displayed, then CUDA is working properly.

If the following error occurs

```
The version ('9.1') of the host compiler ('Apple clang') is not supported
```

> Explain that the Xcode version is too new and requires downgrading Xcode

### Step 3: Install cuDNN

> CUDNN (CUDA Deep Neural Network library): An accelerated library for deep neural networks created by NVIDIA. It is a GPU acceleration library for deep neural networks. If you want to train the model with the GPU, cuDNN is not required, but this acceleration library is generally used.

cuDNN
- https://developer.nvidia.com/rdp/cudnn-download
- Download the latest version of cuDNN v7.2.1 for CUDA 9.2
- cudnn-9.2-osx-x64-v7.2.1.38.tgz

After the best, merge the decompression directly into the CUDA directory /usr/local/cuda/:

```bash
$ tar -xzvf cudnn-9.2-osx-x64-v7.2.1.38.tgz
$ sudo cp cuda/include/cudnn.h /usr/local/cuda/include
$ sudo cp cuda/lib/libcudnn* /usr/local/cuda/lib
$ sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib/libcudnn*
$ rm -rf cuda
```

### Step 4: Install CUDA-Z

> Used to view CUDA operations

```bash
$ brew cask install cuda-z
```
Then you can run CUDA-Z from the Application to see the CUDA operation.

![CUDA-Z](https://github.com/SixQuant/tensorflow-macos-gpu/raw/master/res/CUDA-Z.png)

## Compile

> If you have a compiled version, you can skip this chapter and go directly to the "Installation" section.

The following is a compilation of the Tensorflow GPU version from source.

### CUDA preparation

> Please refer to the previous section

### Compilation environment preparation

Python

```bash
$ python3 --version
Python 3.6.5
```

> Do not use Python 3.7.0, otherwise there will be problems with compilation

Python dependency

```bash
$ pip3 install six numpy wheel
```

Coreutils，llvm，OpenMP

```bash
$ brew install coreutils llvm cliutils/apple/libomp
```

Bazel

> Note that this must be a 0.14.0 version, both new and old can cause compilation to fail. Download version 0.14.0, [bazel release page](https://github.com/bazelbuild/bazel/releases)

```bash
$ curl -O https://github.com/bazelbuild/bazel/releases/download/0.14.0/bazel-0.14.0-installer-darwin-x86_64.sh
$ chmod +x bazel-0.14.0-installer-darwin-x86_64.sh
$ ./bazel-0.14.0-installer-darwin-x86_64.sh
$ bazel version
Build label: 0.14.0
```

> Too low version may result in no environment variables being found, so Library not loaded

Check the NVIDIA development environment

```bash
$ nvcc -V
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2018 NVIDIA Corporation
Built on Tue_Jun_12_23:08:12_CDT_2018
Cuda compilation tools, release 9.2, V9.2.148
```

Check the clang version

```bash
$ cc -v
Apple LLVM version 8.0.0 (clang-800.0.42.1)
Target: x86_64-apple-darwin17.7.0
Thread model: posix
InstalledDir: /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin
```

### Source Preparation

Pull the TensorFlow source release 1.8 branch and modify it to make it compatible with macOS

Here you can download the modified source directly.

```bash
$ curl -O https://raw.githubusercontent.com/SixQuant/tensorflow-macos-gpu/master/tensorflow-macos-gpu-r1.8-src.tar.gz
```

Or manually modify

```bash
$ git clone https://github.com/tensorflow/tensorflow -b r1.8
$ cd tensorflow
$ curl -O https://raw.githubusercontent.com/SixQuant/tensorflow-macos-gpu/master/patch/tensorflow-macos-gpu-r1.8.patch
$ git apply tensorflow-macos-gpu-r1.8.patch
$ curl -o third_party/nccl/nccl.h https://raw.githubusercontent.com/SixQuant/tensorflow-macos-gpu/master/patch/nccl.h
```

### Build

Configuration

```bash
$ which python3
/usr/local/bin/python3
```

```bash
$ ./configure
```

```
Please specify the location of python. [Default is /usr/local/opt/python@2/bin/python2.7]: /usr/local/bin/python3

Found possible Python library paths:
  /usr/local/Cellar/python/3.6.5_1/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages
Please input the desired Python library path to use.  Default is [/usr/local/Cellar/python/3.6.5_1/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages]

Do you wish to build TensorFlow with Google Cloud Platform support? [Y/n]: n
No Google Cloud Platform support will be enabled for TensorFlow.

Do you wish to build TensorFlow with Hadoop File System support? [Y/n]: n
No Hadoop File System support will be enabled for TensorFlow.

Do you wish to build TensorFlow with Amazon S3 File System support? [Y/n]: n
No Amazon S3 File System support will be enabled for TensorFlow.

Do you wish to build TensorFlow with Apache Kafka Platform support? [Y/n]: n
No Apache Kafka Platform support will be enabled for TensorFlow.

Do you wish to build TensorFlow with XLA JIT support? [y/N]: n
No XLA JIT support will be enabled for TensorFlow.

Do you wish to build TensorFlow with GDR support? [y/N]: n
No GDR support will be enabled for TensorFlow.

Do you wish to build TensorFlow with VERBS support? [y/N]: n
No VERBS support will be enabled for TensorFlow.

Do you wish to build TensorFlow with OpenCL SYCL support? [y/N]: n
No OpenCL SYCL support will be enabled for TensorFlow.

Do you wish to build TensorFlow with CUDA support? [y/N]: y
CUDA support will be enabled for TensorFlow.

Please specify the CUDA SDK version you want to use, e.g. 7.0. [Leave empty to default to CUDA 9.0]: 9.2

Please specify the location where CUDA 9.1 toolkit is installed. Refer to README.md for more details. [Default is /usr/local/cuda]:

Please specify the cuDNN version you want to use. [Leave empty to default to cuDNN 7.0]: 7.2

Please specify the location where cuDNN 7 library is installed. Refer to README.md for more details. [Default is /usr/local/cuda]:

Please specify a list of comma-separated Cuda compute capabilities you want to build with.
You can find the compute capability of your device at: https://developer.nvidia.com/cuda-gpus.
Please note that each additional compute capability significantly increases your build time and binary size. [Default is: 3.5,5.2]3.0,3.5,5.0,5.2,6.0,6.1

Do you want to use clang as CUDA compiler? [y/N]:n
nvcc will be used as CUDA compiler.

Please specify which gcc should be used by nvcc as the host compiler. [Default is /usr/bin/gcc]:

Do you wish to build TensorFlow with MPI support? [y/N]:
No MPI support will be enabled for TensorFlow.

Please specify optimization flags to use during compilation when bazel option "--config=opt" is specified [Default is -march=native]:

Would you like to interactively configure ./WORKSPACE for Android builds? [y/N]:
Not configuring the WORKSPACE for Android builds.

Preconfigured Bazel build configs. You can use any of the below by adding "--config=<>" to your build command. See tools/bazel.rc for more details.
	--config=mkl         	# Build with MKL support.
	--config=monolithic  	# Config for mostly static monolithic build.
Configuration finished
```

> Be sure to enter the correct version
>
> * /usr/local/bin/python3
> * CUDA 9.2
> * cuDNN 7.2
> * Compute capability 3.0, 3.5, 5.0, 5.2, 6.0, 6.1 This must check the version supported by your graphics card, you can enter multiple

The above actually generated the .tf_configure.bazelrc configuration file

Start compiling

```bash
$ bazel clean --expunge
$ bazel build --config=opt --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" --action_env PATH --action_env DYLD_LIBRARY_PATH //tensorflow/tools/pip_package:build_pip_package
```

> Due to network problems during compilation, download failure may occur, and multiple attempts may be made.
>
> If the bazel version is incorrect, it may cause DYLD_LIBRARY_PATH not passed, so Library not loaded

#### Compilation instructions

The meaning of --config=opt should be

```
build:opt --copt=-march=native
build:opt --host_copt=-march=native
build:opt --define with_default_optimizations=true
```

> -march=native means to compile with the optimization instructions supported by the current CPU.

View the current CPU-supported instruction set

```bash
$ sysctl machdep.cpu.features
machdep.cpu.features: FPU VME DE PSE TSC MSR PAE MCE CX8 APIC SEP MTRR PGE MCA CMOV PAT PSE36 CLFSH DS ACPI MMX FXSR SSE SSE2 SS HTT TM PBE SSE3 PCLMULQDQ DTES64 MON DSCPL VMX EST TM2 SSSE3 FMA CX16 TPR PDCM SSE4.1 SSE4.2 x2APIC MOVBE POPCNT AES PCID XSAVE OSXSAVE SEGLIM64 TSCTMR AVX1.0 RDRAND F16C
```

```bash
$ gcc -march=native -dM -E -x c++ /dev/null | egrep "AVX|SSE"

#define __AVX2__ 1
#define __AVX__ 1
#define __SSE2_MATH__ 1
#define __SSE2__ 1
#define __SSE3__ 1
#define __SSE4_1__ 1
#define __SSE4_2__ 1
#define __SSE_MATH__ 1
#define __SSE__ 1
#define __SSSE3__ 1
```

#### Compile error dyld: Library not loaded: @rpath/libcudart.9.2.dylib

```
ERROR: /Users/c/Downloads/tensorflow-macos-gpu-r1.8/src/tensorflow/python/BUILD:1590:1: Executing genrule //tensorflow/python:string_ops_pygenrule failed (Aborted): bash failed: error executing command /bin/bash bazel-out/host/genfiles/tensorflow/python/string_ops_pygenrule.genrule_script.sh
dyld: Library not loaded: @rpath/libcudart.9.2.dylib
  Referenced from: /private/var/tmp/_bazel_c/ea0f1e868907c49391ddb6d2fb9d5630/execroot/org_tensorflow/bazel-out/host/bin/tensorflow/python/gen_string_ops_py_wrappers_cc
  Reason: image not found
```

> The environment variable DYLD_LIBRARY_PATH was not passed in due to a bazel bug.

Solution: Install the correct version of bazel

#### Compile error PyString_AsStringAndSize

```
external/protobuf_archive/python/google/protobuf/pyext/descriptor_pool.cc:169:7: error: assigning to 'char *' from incompatible type 'const char *'
  if (PyString_AsStringAndSize(arg, &name, &name_size) < 0) {
```

> This is because Python3.7 has a bug with protobuf_python. Please recompile after changing to Python3.6.
>
> https://github.com/google/protobuf/issues/4086

Compile time is up to 1.5 hours, please be patient

### Generate a PIP installation package

Recompile and replace _nccl_ops.so

```Bash
$ gcc -march=native -c -fPIC tensorflow/contrib/nccl/kernels/nccl_ops.cc -o _nccl_ops.o
$ gcc _nccl_ops.o -shared -o _nccl_ops.so
$ mv _nccl_ops.so bazel-out/darwin-py3-opt/bin/tensorflow/contrib/nccl/python/ops
$ rm _nccl_ops.o
```

Unpack

```bash
$ bazel-bin/tensorflow/tools/pip_package/build_pip_package ~/Downloads/
```

Clean up

```bash
$ bazel clean --expunge
```

## Installation

```bash
$ pip3 uninstall tensorflow
$ pip3 install ~/Downloads/tensorflow-1.8.0-cp36-cp36m-macosx_10_13_x86_64.whl
```

Can also be installed directly through http

```bash
$ pip3 install https://github.com/SixQuant/tensorflow-macos-gpu/releases/download/v1.8.0/tensorflow-1.8.0-cp36-cp36m-macosx_10_13_x86_64.whl
```

> If it is a direct installation, please make sure that the relevant version is consistent with the compiled or higher.
>
> * cudadriver_396.148_macos.dmg
> * cuda_9.2.148_mac.dmg
> * cuda_9.2.148.1_mac.dmg
> * cudnn-9.2-osx-x64-v7.2.1.38.tgz

## Confirm

> Confirm that the Tensorflow GPU is working properly

### Confirm environment variables

> Confirm that the Python code can read the correct environment variable DYLD_LIBRARY_PATH

```bash
$ nano tensorflow-gpu-01-env.py
```

```python
#!/usr/bin/env python

import os

print(os.environ["DYLD_LIBRARY_PATH"])
```

```bash
$ python3 tensorflow-gpu-01-env.py
/usr/local/cuda/lib:/usr/local/cuda/extras/CUPTI/lib
```

### Confirm if the GPU is enabled

If the TensorFlow instruction has both a CPU and GPU implementation, the GPU device has priority when the instruction is assigned to the device. For example, if matmul both CPU and GPU kernel functions, in systems with both cpu:0 and gpu:0 devices, gpu:0 will be selected to run matmul. To find out which device your instructions and tensors are assigned to, create a session and set the log_device_placement configuration option to True.

```bash
$ nano tensorflow-gpu-02-hello.py
```

```python
#!/usr/bin/env python

import tensorflow as tf

config = tf.ConfigProto()
config.log_device_placement = True

# Creates a graph.
a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
c = tf.matmul(a, b)
# Creates a session with log_device_placement set to True.
with tf.Session(config=config) as sess:
    # Runs the op.
    print(sess.run(c))
```

```bash
$ python3 tensorflow-gpu-02-hello.py
2018-08-26 14:13:45.987276: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1356] Found device 0 with properties:
name: GeForce GTX 750 Ti major: 5 minor: 0 memoryClockRate(GHz): 1.2545
pciBusID: 0000:01:00.0
totalMemory: 2.00GiB freeMemory: 706.66MiB
2018-08-26 14:13:45.987303: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1435] Adding visible gpu devices: 0
2018-08-26 14:13:46.245132: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1053] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 426 MB memory) -> physical GPU (device: 0, name: GeForce GTX 750 Ti, pci bus id: 0000:01:00.0, compute capability: 5.0)
Device mapping:
/job:localhost/replica:0/task:0/device:GPU:0 -> device: 0, name: GeForce GTX 750 Ti, pci bus id: 0000:01:00.0, compute capability: 5.0
2018-08-26 14:13:46.253938: I tensorflow/core/common_runtime/direct_session.cc:284] Device mapping:
/job:localhost/replica:0/task:0/device:GPU:0 -> device: 0, name: GeForce GTX 750 Ti, pci bus id: 0000:01:00.0, compute capability: 5.0

MatMul: (MatMul): /job:localhost/replica:0/task:0/device:GPU:0
2018-08-26 14:13:46.254406: I tensorflow/core/common_runtime/placer.cc:886] MatMul: (MatMul)/job:localhost/replica:0/task:0/device:GPU:0
b: (Const): /job:localhost/replica:0/task:0/device:GPU:0
2018-08-26 14:13:46.254415: I tensorflow/core/common_runtime/placer.cc:886] b: (Const)/job:localhost/replica:0/task:0/device:GPU:0
a: (Const): /job:localhost/replica:0/task:0/device:GPU:0
2018-08-26 14:13:46.254421: I tensorflow/core/common_runtime/placer.cc:886] a: (Const)/job:localhost/replica:0/task:0/device:GPU:0
[[22. 28.]
 [49. 64.]]
```

> Some of the useless log output that seems to be worrying is commented out directly from the source code, for example:
>
> OS X does not support NUMA - returning NUMA node zero
>
> Not found: TF GPU device with id 0 was not registered

### Running a little more complicated

```bash
$ nano tensorflow-gpu-04-cnn-gpu.py
```

```python
#!/usr/bin/env python

from __future__ import absolute_import, division, print_function
import os
import time
import numpy as np
import tflearn
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

from tensorflow.python.client import device_lib
def print_gpu_info():
    for device in device_lib.list_local_devices():
        print(device.name, 'memory_limit', str(round(device.memory_limit/1024/1024))+'M', 
            device.physical_device_desc)
    print('=======================')

print_gpu_info()


DATA_PATH = "/Volumes/Cloud/DataSet"

mnist = tflearn.datasets.mnist.read_data_sets(DATA_PATH+"/mnist", one_hot=True)

config = tf.ConfigProto()
config.log_device_placement = True
config.allow_soft_placement = True

config.gpu_options.allocator_type = 'BFC'
config.gpu_options.allow_growth = True
#config.gpu_options.per_process_gpu_memory_fraction = 0.3

# Building convolutional network
net = tflearn.input_data(shape=[None, 28, 28, 1], name='input') 
net = tflearn.conv_2d(net, 32, 5, weights_init='variance_scaling', activation='relu', regularizer="L2") 
net = tflearn.conv_2d(net, 64, 5, weights_init='variance_scaling', activation='relu', regularizer="L2") 
net = tflearn.fully_connected(net, 10, activation='softmax') 
net = tflearn.regression(net,
                         optimizer='adam',                  
                         learning_rate=0.01,
                         loss='categorical_crossentropy', 
                         name='target')

# Training
model = tflearn.DNN(net, tensorboard_verbose=3)

start_time = time.time()
model.fit(mnist.train.images.reshape([-1, 28, 28, 1]),
          mnist.train.labels.astype(np.int32),
          validation_set=(
              mnist.test.images.reshape([-1, 28, 28, 1]),
              mnist.test.labels.astype(np.int32)
          ),
          n_epoch=1,
          batch_size=128,
          shuffle=True,
          show_metric=True,
          run_id='cnn_mnist_tflearn')

duration = time.time() - start_time
print('Training Duration %.3f sec' % (duration))
```

```bash
$ python3 tensorflow-gpu-04-cnn-gpu.py
2018-08-26 14:11:00.463212: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1356] Found device 0 with properties:
name: GeForce GTX 750 Ti major: 5 minor: 0 memoryClockRate(GHz): 1.2545
pciBusID: 0000:01:00.0
totalMemory: 2.00GiB freeMemory: 258.06MiB
2018-08-26 14:11:00.463235: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1435] Adding visible gpu devices: 0
2018-08-26 14:11:00.717963: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1053] Created TensorFlow device (/device:GPU:0 with 203 MB memory) -> physical GPU (device: 0, name: GeForce GTX 750 Ti, pci bus id: 0000:01:00.0, compute capability: 5.0)
/device:CPU:0 memory_limit 256M
/device:GPU:0 memory_limit 204M device: 0, name: GeForce GTX 750 Ti, pci bus id: 0000:01:00.0, compute capability: 5.0
=======================
Extracting /Volumes/Cloud/DataSet/mnist/train-images-idx3-ubyte.gz
Extracting /Volumes/Cloud/DataSet/mnist/train-labels-idx1-ubyte.gz
Extracting /Volumes/Cloud/DataSet/mnist/t10k-images-idx3-ubyte.gz
Extracting /Volumes/Cloud/DataSet/mnist/t10k-labels-idx1-ubyte.gz
2018-08-26 14:11:01.158727: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1435] Adding visible gpu devices: 0
2018-08-26 14:11:01.158843: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1053] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 203 MB memory) -> physical GPU (device: 0, name: GeForce GTX 750 Ti, pci bus id: 0000:01:00.0, compute capability: 5.0)
2018-08-26 14:11:01.487530: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1435] Adding visible gpu devices: 0
2018-08-26 14:11:01.487630: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1053] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 203 MB memory) -> physical GPU (device: 0, name: GeForce GTX 750 Ti, pci bus id: 0000:01:00.0, compute capability: 5.0)
---------------------------------
Run id: cnn_mnist_tflearn
Log directory: /tmp/tflearn_logs/
---------------------------------
Training samples: 55000
Validation samples: 10000
--
Training Step: 430  | total loss: 0.16522 | time: 45.764s
| Adam | epoch: 001 | loss: 0.16522 - acc: 0.9660 | val_loss: 0.06837 - val_acc: 0.9780 -- iter: 55000/55000
--
Training Duration 45.898 sec
```

> The speed increase is obvious:
>
> CPU version without AVX2 FMA, time: 168.151s
>
> CPU version plus AVX2 FMA, time: 147.697s
>
> GPU version plus AVX2 FMA, time: 45.898s

### cuda-smi

> cuda-smi is used to replace nvidia-smi on Mac

nvidia-smi is used to view GPU memory usage.

After downloading, put it in the /usr/local/bin/ directory.

```bash
$ sudo scp cuda-smi /usr/local/bin/
$ sudo chmod 755 /usr/local/bin/cuda-smi
$ cuda-smi
Device 0 [PCIe 0:1:0.0]: GeForce GTX 750 Ti (CC 5.0): 5.0234 of 2047.7 MB (i.e. 0.245%) Free
```

## Problem

### Error _ncclAllReduce

> Recompile a _nccl_ops.so and copy it in the past.

```bash
$ gcc -c -fPIC tensorflow/contrib/nccl/kernels/nccl_ops.cc -o _nccl_ops.o
$ gcc _nccl_ops.o -shared -o _nccl_ops.so
$ mv _nccl_ops.so /usr/local/lib/python3.6/site-packages/tensorflow/contrib/nccl/python/ops/
$ rm _nccl_ops.o
```

### Library not loaded: @rpath/libcublas.9.2.dylib

> This is because the DYLD_LIBRARY_PATH environment variable is missing from the Jupyter
>
> Or the new version of MacOS prohibits you from making random changes to unsafe factors such as DYLD_LIBRARY_PATH unless you turn off SIP.

Reproduce

```python
import os
os.environ['DYLD_LIBRARY_PATH']
```

> The above code will get an error in Jupyter because the environment variable DYLD_LIBRARY_PATH cannot be modified because of SIP.

Solution: Refer to the previous "Environment Variables" settings section.

### Segmentation Fault

> The so-called segmentation error means that the memory accessed exceeds the memory space given by the system.

Solution: Please re-confirm that the correct version and compilation parameters are used, especially XCode

### Not found: TF GPU device with id 0 was not registered

> Ignore this warning directly

##  Is there a leak in the GPU memory?

I don't know what to solve: (

 