### 概述

要完成TritonJit面向华为昇腾NPU 910B的环境适配，有两个检验指标：

- 一、能成功构建并运行Ascend/triton-ascend（对应适配TritonJit的standalone_compile的Python文件）
- 二、能成功构建并运行Ascend/AscendNPU IR（对应适配TritonJit的wrapper相关的C++文件）
- 三、安装libtorch_npu完成<torch_npu/torch_npu.h>库的导入

### 目录

**一、triton-ascend配置进度：**

- [x] 测试远程连接
- [x] 安装cann软件包
- [x] 安装python软件包
- [x] 安装系统软件包
  - [x] 检查gcc和glibc版本
  - [x] 安装ccache
  - [x] 安装clang和lld
- [x] 安装triton-ascend依赖的triton
- [x] 源码构建llvm-project
- [x] 源码构建TritonAscend
- [x] 运行01-vec-add.py

**二、AscendNPU IR配置进度**

- [x] 测试远程连接和安装cann软件包
- [x] 第一步：安装预编译组件
- [x] 第二步：将BiSheng IR构建为外部LLVM项目
- [x] 第三步：构建llvm
- [x] 第四步：构建端到端样例
- [x] 第五步：运行端到端样例

**三、libtorch_npu 安装及路径设置**

### 一、triton-ascend配置

triton-ascend的gitee链接：https://gitee.com/ascend/triton-ascend
可参考安装步骤：https://gitee.com/ascend/triton-ascend/blob/master/docs/sources/getting-started/installation.md

测试远程连接

```
aTrust登录
用户名:wuwenli
密码：Cjzk@2025

远程链接
ssh -t baai_user@10.166.54.63 'ssh baai_user@124.224.239.245 -p 1158'
	需要依次输入两个密码
	baai_2025
	NB@aP079ghD.

ssh baai-08
sudo su
	再次输入密码：NB@aP079ghD.
```

安装cann软件包

```
0. 配置cann软件包环境变量
$ source /usr/local/Ascend/ascend-toolkit/set_env.sh

1. 检查Toolkit开发套件包安装
$ cat /usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/ascend_toolkit_install.info
打印信息如下：
	package_name=Ascend-cann-toolkit
	version=8.3.RC1.alpha001
	innerversion=V100R001C23B046
	compatible_version=[V100R001C15],[V100R001C18],[V100R001C19],[V100R001C20],[V100R001C21],[V100R001C23]
	arch=aarch64
	os=linux
	path=/usr/local/Ascend/ascend-toolkit/8.3.RC1.alpha001/aarch64-linux

2. 检查Kernels算子包安装
$ cat /usr/local/Ascend/ascend-toolkit/latest/opp_kernel/version.info
打印信息如下：
	Version=8.3.T5.0.B028
	version_dir=8.3.RC1.alpha001
	timestamp=20250807_092944934
	ops_version=8.3.T5.0.B028
	adk_version=8.3.T5.0.B028
	required_package_amct_acl_version=">=7.6, <=8.3"
	required_package_aoe_version=">=7.6, <=8.3"
	required_package_compiler_version=">=7.6, <=8.3"
	required_package_fwkplugin_version=">=7.6, <=8.3"
	required_package_hccl_version=">=7.6, <=8.3"
	required_package_nca_version=">=7.6, <=8.3"
	required_package_ncs_version=">=7.6, <=8.3"
	required_package_opp_version=">=7.6, <=8.3"
	required_package_runtime_version=">=7.6, <=8.3"
	required_package_toolkit_version=">=7.6, <=8.3"
```

安装python软件包

```
安装miniconda，创建py310的新环境并命名为"triton"，换源安装torch_npu和相关库
$ conda activate triton
$ pip config set global.index-url https://mirrors.aliyun.com/pypi/simple
$ pip install attrs==24.2.0 numpy==1.26.4 scipy==1.13.1 decorator==5.1.1 psutil==6.0.0 pytest==8.3.2 pytest-xdist==3.6.1 pyyaml
$ pip install torch_npu==2.6.0
```



检查gcc和glibc版本

```
(triton) [root@worker-13 triton_jit]# gcc --version
gcc (GCC) 10.3.1

(triton) [root@worker-13 triton_jit]# ldd --version
ldd (GNU libc) 2.34
```

安装ccache。一开始直接安装遇到了“Cannot download repodata/repomd.xml: All mirrors were tried”的问题，参考该链接换源：https://www.cnblogs.com/hsh96/p/18156103。采用`sudo yum install ccache`安装成功。

```
(triton) [root@worker-13 build-b5cc2]# ccache --version
ccache version 3.7.12
```

安装版本15以上的clang和lld。直接yum install的版本太低，这里选择基于llvm-project构建

```
cd /data/baai_user_home/triton_jit/llvm-project
git checkout b5cc222d7429fe6f18c787f633d5262fac2e676f
mkdir build-clang
cd build-clang
cmake ../llvm \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_ENABLE_PROJECTS="clang;lld" \
  -DLLVM_TARGETS_TO_BUILD="host"
ninja
sudo ln -s /data/baai_user_home/triton_jit/llvm-project/build-clang/bin/clang /usr/local/bin/clang
sudo ln -s /data/baai_user_home/triton_jit/llvm-project/build-clang/bin/clang++ /usr/local/bin/clang++
```

安装triton-ascend。triton-ascend可从gitee拉取，但是triton-ascend依赖github里的triton，因为不能访问github（或者能访问但很慢），改为clone一个gitee上的triton镜像

```
cd /data/baai_user_home/triton_jit/triton-ascend
cd third_party
rmdir triton # 移除原先submodule的来自github的triton文件夹
git clone https://gitee.com/shijingchang/triton
git submodule update --remote
git checkout 9641643 # triton-ascend依赖的triton对应commit
```

源码构建llvm-project，它在源码构建triton-ascend里的triton时被依赖

```
cd /data/baai_user_home/triton_jit/llvm-project
mkdir build-b5cc2
cd build-b5cc2

cmake ../llvm \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_ENABLE_PROJECTS="mlir;llvm" \
  -DLLVM_TARGETS_TO_BUILD="host;NVPTX;AMDGPU" \
  -DCMAKE_INSTALL_PREFIX=${LLVM_INSTALL_PREFIX} \
  -DCMAKE_C_COMPILER=/data/baai_user_home/triton_jit/llvm-project/build-clang/bin/clang \
  -DCMAKE_CXX_COMPILER=/data/baai_user_home/triton_jit/llvm-project/build-clang/bin/clang++
ninja

export LLVM_INSTALL_PREFIX=/data/baai_user_home/triton_jit/llvm-project/build-b5cc2
export LLVM_INSTALL_PREFIX=/home/zhouxulin/research/llvm-project/build
```

源码构建triton-ascend

```
cd triton-ascend/
LLVM_SYSPATH=${LLVM_INSTALL_PREFIX} \
TRITON_PLUGIN_DIRS=./ascend \
TRITON_BUILD_WITH_CCACHE=true \
TRITON_BUILD_WITH_CLANG_LLD=true \
TRITON_BUILD_PROTON=OFF \
TRITON_WHEEL_NAME="triton-ascend" \
TRITON_APPEND_CMAKE_ARGS="-DTRITON_BUILD_UT=OFF" \
python3 setup.py install
```

运行01-vec-add.py

```bash
conda activate triton
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd /data/baai_user_home/triton_jit/triton-ascend
python3 ./ascend/examples/tutorials/01-vector-add.py
输出结果：
	[W922 16:39:09.197134268 compiler_depend.ts:164] Warning: Warning: Device do not support double dtype now, dtype cast repalce with float. (function operator())
	tensor([0.8329, 1.0024, 1.3639,  ..., 1.0796, 1.0406, 1.5811], device='npu:0')
	tensor([0.8329, 1.0024, 1.3639,  ..., 1.0796, 1.0406, 1.5811], device='npu:0')
```

期间一度遇到`call aclnnAdd failed`的问题，经检查aclnnAdd为cann的算子加速库，相关文档和该算子的端到端运行示例见：https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/index/index.html?sub_id=%2Fzh%2Fcanncommercial%2F82RC1%2FAPI%2Faolapi%2Fcontext%2FaclnnAdd%26aclnnInplaceAdd.md。

搭建端到端运行示例后，排查出原先所安装的算子库面向910，而不是在本机器执行`npu-smi info`所显示的910B3，重新安装面向正确硬件的算子库后解决。

-----

### 二、AscendNPU IR配置

AscendNPU IR的gitee链接：https://gitee.com/ascend/ascendnpu-ir

第一步：从官网下载并安装构建BiSheng IR所需的预编译组件（包括bishengir-compile等）

```
export BISHENG_IR_INSTALL_PATH=/data/baai_user_home/bishengir_aarch64
```

第二步：将BiSheng IR构建为外部LLVM项目。这里说的BiSheng IR就是AscendNPU IR项目

```
cd /data/baai_user_home/llvm-project
git checkout llvmorg-19.1.7
git submodule add https://gitee.com/ascend/ascendnpu-ir.git third-party/bishengir
```

第三步：构建llvm

```
cd build
cmake -G Ninja -DCMAKE_BUILD_TYPE=Release ../llvm \
-DLLVM_ENABLE_PROJECTS="mlir;llvm" \
-DLLVM_EXTERNAL_PROJECTS="bishengir" \
-DLLVM_EXTERNAL_BISHENGIR_SOURCE_DIR=/data/baai_user_home/llvm-project/third-party/bishengir \
-DBISHENG_IR_INSTALL_PATH=${BISHENG_IR_INSTALL_PATH} \
-DBISHENGIR_BUILD_EXAMPLES=ON

cmake --build . --target "check-bishengir"
打印单元测试的信息如下：
Total Discovered Tests: 11
  Unsupported:  1 (9.09%)
  Passed     : 10 (90.91%)
```

第四步：构建端到端样例。该样例位于/data/baai_user_home/llvm-project/third-party/bishengir/examples/HIVM/VecAdd下

```
echo $ASCEND_HOME_PATH # 输出：/usr/local/Ascend/ascend-toolkit/latest/aarch64-linux
cmake --build .
ls ./bin | grep hivm # 输出：hivm-vec-add
```

第五步：运行端到端样例

```
cd bin
bishengir-compile /data/baai_user_home/llvm-project/third-party/bishengir/examples/HIVM/VecAdd/add.mlir -enable-hivm-compile -o kernel.o
./hivm-vec-add
输出信息
	[success] initialize success
	[success] file:./kernel.o read succ!
	[success] register kernel success
	[success] memcpy host to device success
	[success] stream synchronize success
	[success] memcpy device to host success
	i0       Expect: 1                              Result: 1
	i1       Expect: 2                              Result: 2
	i2       Expect: 3                              Result: 3
	i3       Expect: 4                              Result: 4
	i4       Expect: 5                              Result: 5
	i5       Expect: 6                              Result: 6
	i6       Expect: 7                              Result: 7
	i7       Expect: 8                              Result: 8
	i8       Expect: 9                              Result: 9
	i9       Expect: 10                             Result: 10
	i10      Expect: 11                             Result: 11
	i11      Expect: 12                             Result: 12
	i12      Expect: 13                             Result: 13
	i13      Expect: 14                             Result: 14
	i14      Expect: 15                             Result: 15
	i15      Expect: 16                             Result: 16
	[success] compare output success
```

上面打印出的"success"操作对应的是ACL使用接口，可在ACL接口手册中查找到：https://www.hiascend.com/doc_center/source/zh/canncommercial/82RC1/API/appdevgapi/CANN%E5%95%86%E7%94%A8%E7%89%88%208.2.RC1%20%20AscendCL%E5%BA%94%E7%94%A8%E5%BC%80%E5%8F%91%E6%8E%A5%E5%8F%A3%2001.pdf



### 三、Triton_JIT

在配置triton_ascend环境时通过``pip install torch_npu`` 的方式安装的triton_npu包
缺少acl相关文件以及``#include <torch_npu/torch_npu.h>``无法找到对应库



安装libtorch_npu插件，参考下方链接

https://www.hiascend.com/document/detail/zh/Pytorch/60RC2/configandinstg/instg/insg_0007.html



在Triton_JIT的主CMakeLists.txt中

TORCH_NPU_PATH 设置成了我的安装路径/data/baai_user_home/chwork/compile_triton/pytorch/libtorch_npu

Line174 ``set(TORCH_NPU_PATH "/data/baai_user_home/chwork/compile_triton/pytorch/libtorch_npu")``

需要修改成当前环境对应的安装路径即可



主CMakeLists.txt中针对github下载网络问题 添加了USE_CHINA_MIRRORS参数（非必须）

可以使用-DUSE_CHINA_MIRRORS=ON换成镜像源，默认使用原github链接下载

```shell
cmake -S . -B build/ -DPython_ROOT="$(which python)/../.." -DUSE_CHINA_MIRRORS=ON
```

编译指令

```
cmake --build build/ --parallel
```



**编译后运行命令和打印参考**

```shell
(triton) [root@worker-13 triton_jit]# cd build/examples/pointwise/
(triton) [root@worker-13 pointwise]# ./test_add 
Tensor a device: npu:0
Tensor b device: npu:0
First 10 elements of tensor b: 0.0982715 0.179066 0.593264 0.306981 0.553679 0.717337 0.392389 0.926874 0.457076 0.23576 
First 10 elements of tensor a: 0.246928 0.700275 0.672404 0.819334 0.171779 0.517412 0.605406 0.318786 0.99653 0.556897 
cache_manager.cache_dir:  /root/.triton/cache/2Gf3ZRGxBnsaCOz8dlzdoOG8BbsnPtqFY-n5oNhbuoM
0.345199 0.87934 1.26567 1.12631 0.725459 1.23475 0.997795 1.24566 1.45361 0.792657 
```

