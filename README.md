# CODMPC

## 说明

该分支用于CODMPC的开发，与main分支的区别如下：

- 直接使用C++ Pinocchio计算运动学与动力学，不再使用acados生成代码
- 求解器改为hpipm

## Dependencies

系统推荐Ubuntu24.04。Ubuntu20.04经过测试无法安装仿真器环境。

Before proceeding with the installation, ensure that the following dependencies are available on your system:

- CMake
- GCC
- Python 3 (along with development headers)
- Eigen3
- YAML-CPP
- Pybind11
- ndcurves
- hpipm-cpp
- Python package `mcap`

## Installation 

### 1. Clone the Repository

To get started, clone the repository:

```bash
git clone git@github.com:srqqq/DWMPC.git
cd DWMPC
git checkout cod-mpc-develop
```

### 2. Install System Dependencies

```bash
sudo apt-get install -y cmake g++ python3 python3-dev python3-pip libeigen3-dev libyaml-cpp-dev pybind11-dev
python3 -m pip install mcap
```
- Follow the instructions to install the `ndcurves` library from the [official repository](https://github.com/loco-3d/ndcurves)
- install the `gym-quadruped` environment from the the [official repository](https://github.com/iit-DLSLab/gym-quadruped)
- install hpipm-cpp  from the the [official repository](https://github.com/srqqq/hpipm-cpp) （注意：由于hpipm-cpp库长时间未更新，最新的hpipm和blasfeo接口已变更，需要将两个库回退到指定commit才能编译成功，具体见仓库README）

### 3. Build DWMPC
From the main `DWMPC` repository, create a build directory and compile the project:
```bash
mkdir build && cd build
cmake ..
make -j8 && sudo make install
```
Add the `DWMPC` library to your environment:
```bash
export LD_LIBRARY_PATH=/usr/lib/dls2/controllers/dwmpc:$LD_LIBRARY_PATH
export PYTHONPATH=$PYTHONPATH:/usr/lib/dls2/controllers/dwmpc
```
### 4. To run the example

```
cd example
python codmpc_sim.py
```

When logging is enabled, the example writes a compressed MCAP file under
`data/experiment/`. The file name records the robot, trajectory, controller,
and simulation condition. Its dynamic flat schema is compatible with the
`quadruped_mpc` data tools.
