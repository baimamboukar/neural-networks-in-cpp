<div align="center">

# Neural Networks in C++

<p align="center">
  <i>A high-performance, from-scratch implementation of neural networks in modern C++</i>
</p>

[![Project Status: Active](http://www.repostatus.org/badges/latest/active.svg)](http://www.repostatus.org/#active)
[![Build Status](https://travis-ci.org/bsamseth/cpp-project.svg?branch=master)](https://travis-ci.org/bsamseth/cpp-project)
[![Build status](https://ci.appveyor.com/api/projects/status/g9bh9kjl6ocvsvse/branch/master?svg=true)](https://ci.appveyor.com/project/bsamseth/cpp-project/branch/master)
[![Coverage Status](https://coveralls.io/repos/github/bsamseth/cpp-project/badge.svg?branch=master)](https://coveralls.io/github/bsamseth/cpp-project?branch=master)
[![codecov](https://codecov.io/gh/bsamseth/cpp-project/branch/master/graph/badge.svg)](https://codecov.io/gh/bsamseth/cpp-project)
[![License](https://img.shields.io/badge/license-Unlicense-blue.svg)](https://github.com/bsamseth/cpp-project/blob/master/LICENSE)
[![C++](https://img.shields.io/badge/C++-17-blue.svg?style=flat&logo=c%2B%2B)](https://isocpp.org/)
[![CMake](https://img.shields.io/badge/CMake-3.15+-064F8C.svg?style=flat&logo=cmake)](https://cmake.org/)

</div>

## Overview

Neural Networks in C++ is a lightweight implementation of fundamental neural network architectures built from scratch. This project provides clean C++ code for understanding neural networks at a fundamental level.

## Content

- Forward and backward propagation implementation
- Multiple activation functions: Sigmoid, Tanh, ReLU, GeLU, Softmax
- Loss functions: MSE, RMSE, Cross-Entropy
- Layer architectures: Dense and Linear layers
- Modular design for easy extension

## Getting Started

### Prerequisites

- C++ Compiler with C++17 support
- CMake 3.15 or higher

### Build

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make
./main
```

## Usage Example

```cpp
#include "layers.h"
#include "activations.h"
#include "losses.h"

int main() {
    using namespace Layers;
    using namespace ActivationFunctions;
    using namespace LossFunctions;

    // Initialize layers
    Dense layer1;
    ReLU activation1;
    Dense layer2;
    SoftMax activation2;

    // Define loss function
    CrossEntropy loss;

    return 0;
}
```

## Testing

```bash
cd build
make test
```

## Project Structure

```text
neural-networks-in-cpp/
├── CMakeLists.txt
├── README.md
├── LICENSE
│
├── app/
│   └── main.cpp
│
├── include/
│   ├── example.h
│   └── exampleConfig.h.in
│
├── src/
│   ├── layers.h
│   ├── layers.cpp
│   ├── activations.h
│   ├── activations.cpp
│   ├── losses.h
│   ├── losses.cpp
│   └── example.cpp
│
└── tests/
    ├── main.cpp
    └── dummy.cpp
```

## Roadmap

- [ ] Implement Convolutional layers (Conv2D, MaxPool2D)
- [ ] Add optimization algorithms (SGD, Adam, RMSprop)
- [ ] Matrix operations library (or integration with Eigen)
- [ ] GPU acceleration with CUDA/OpenCL
- [ ] Model serialization (save/load trained models)
- [ ] Advanced architectures (RNN, LSTM, Transformer)
- [ ] Data loading utilities
- [ ] Benchmarking suite
