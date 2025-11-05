<div align="center">

# 🧠 Neural Networks in C++

<p align="center">
  <i>A high-performance, from-scratch implementation of neural networks and deep learning algorithms in modern C++</i>
</p>

[![Project Status: Active](http://www.repostatus.org/badges/latest/active.svg)](http://www.repostatus.org/#active)
[![Build Status](https://travis-ci.org/bsamseth/cpp-project.svg?branch=master)](https://travis-ci.org/bsamseth/cpp-project)
[![Build status](https://ci.appveyor.com/api/projects/status/g9bh9kjl6ocvsvse/branch/master?svg=true)](https://ci.appveyor.com/project/bsamseth/cpp-project/branch/master)
[![Coverage Status](https://coveralls.io/repos/github/bsamseth/cpp-project/badge.svg?branch=master)](https://coveralls.io/github/bsamseth/cpp-project?branch=master)
[![codecov](https://codecov.io/gh/bsamseth/cpp-project/branch/master/graph/badge.svg)](https://codecov.io/gh/bsamseth/cpp-project)
[![License](https://img.shields.io/badge/license-Unlicense-blue.svg)](https://github.com/bsamseth/cpp-project/blob/master/LICENSE)
[![Lines of Code](https://tokei.rs/b1/github/bsamseth/cpp-project)](https://github.com/Aaronepower/tokei)
[![C++](https://img.shields.io/badge/C++-17-blue.svg?style=flat&logo=c%2B%2B)](https://isocpp.org/)
[![CMake](https://img.shields.io/badge/CMake-3.15+-064F8C.svg?style=flat&logo=cmake)](https://cmake.org/)

---

</div>

## 🎯 Overview

**Neural Networks in C++** is a lightweight, educational, and performance-oriented implementation of fundamental neural network architectures built entirely from scratch. This project provides a deep dive into the mathematics and mechanics behind modern deep learning, offering a clean C++ codebase for researchers, students, and ML enthusiasts who want to understand neural networks at a fundamental level.

> 💡 **Why C++?** While Python dominates ML research, C++ offers unparalleled performance, memory control, and insight into low-level operations—making it ideal for understanding the computational foundations of neural networks.

<details>
<summary><b>📚 What You'll Learn</b></summary>

- Forward and backward propagation from scratch
- Gradient descent and optimization algorithms
- Activation functions and their derivatives
- Loss functions and backpropagation mathematics
- Layer architectures (Dense, Linear)
- Modern deep learning concepts implemented in pure C++

</details>

---

## 🧮 Mathematical Foundations

### Neural Network Forward Pass

A neural network computes outputs through sequential transformations:

```math
\mathbf{z}^{[l]} = \mathbf{W}^{[l]} \mathbf{a}^{[l-1]} + \mathbf{b}^{[l]}
```

```math
\mathbf{a}^{[l]} = g^{[l]}(\mathbf{z}^{[l]})
```

Where:
- $\mathbf{W}^{[l]}$ is the weight matrix for layer $l$
- $\mathbf{b}^{[l]}$ is the bias vector
- $g^{[l]}$ is the activation function
- $\mathbf{a}^{[l]}$ is the activation output

### Activation Functions

#### Sigmoid
```math
\sigma(x) = \frac{1}{1 + e^{-x}}
```

```math
\sigma'(x) = \sigma(x)(1 - \sigma(x))
```

#### Tanh (Hyperbolic Tangent)
```math
\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
```

```math
\tanh'(x) = 1 - \tanh^2(x)
```

#### ReLU (Rectified Linear Unit)
```math
\text{ReLU}(x) = \max(0, x)
```

```math
\text{ReLU}'(x) = \begin{cases} 1 & \text{if } x > 0 \\ 0 & \text{otherwise} \end{cases}
```

#### GeLU (Gaussian Error Linear Unit)
```math
\text{GeLU}(x) = x \cdot \Phi(x) = x \cdot \frac{1}{2}\left[1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right)\right]
```

#### Softmax
```math
\text{softmax}(\mathbf{z})_i = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}
```

### Loss Functions

#### Mean Squared Error (MSE)
```math
\mathcal{L}_{\text{MSE}} = \frac{1}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i)^2
```

#### Root Mean Squared Error (RMSE)
```math
\mathcal{L}_{\text{RMSE}} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i)^2}
```

#### Cross-Entropy Loss
```math
\mathcal{L}_{\text{CE}} = -\frac{1}{n} \sum_{i=1}^{n} \sum_{j=1}^{C} y_{i,j} \log(\hat{y}_{i,j})
```

### Backpropagation

The gradient of the loss with respect to weights is computed using the chain rule:

```math
\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{[l]}} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}^{[l]}} \cdot \frac{\partial \mathbf{z}^{[l]}}{\partial \mathbf{W}^{[l]}} = \delta^{[l]} \cdot (\mathbf{a}^{[l-1]})^T
```

```math
\frac{\partial \mathcal{L}}{\partial \mathbf{b}^{[l]}} = \delta^{[l]}
```

Where $\delta^{[l]} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}^{[l]}}$ is the error signal.

---

## ✨ Features

- ✅ **Pure C++ Implementation** - No external ML libraries, built from first principles
- ✅ **Multiple Activation Functions** - Sigmoid, Tanh, ReLU, GeLU, Softmax
- ✅ **Flexible Loss Functions** - MSE, RMSE, Cross-Entropy
- ✅ **Layer Architectures** - Dense (Fully Connected) and Linear layers
- ✅ **Modular Design** - Easy to extend with new layers and activation functions
- ✅ **Comprehensive Testing** - Unit tests with doctest framework
- ✅ **Modern C++17** - Clean, readable, and maintainable code
- ✅ **CMake Build System** - Cross-platform compilation support
- ✅ **Documentation** - Doxygen-ready code documentation

---

## 🏗️ Project Structure

```text
neural-networks-in-cpp/
├── CMakeLists.txt           # Build configuration
├── README.md                # This file
├── LICENSE                  # Project license
├── Doxyfile.in             # Documentation config
│
├── app/
│   └── main.cpp            # Main application entry
│
├── include/
│   ├── example.h           # Header files
│   └── exampleConfig.h.in  # CMake config template
│
├── src/
│   ├── layers.h            # Layer implementations (Dense, Linear)
│   ├── layers.cpp
│   ├── activations.h       # Activation functions
│   ├── activations.cpp
│   ├── losses.h            # Loss functions
│   ├── losses.cpp
│   └── example.cpp
│
└── tests/
    ├── main.cpp            # Test runner
    └── dummy.cpp           # Unit tests
```

---

## 🚀 Getting Started

### Prerequisites

- **C++ Compiler** with C++17 support (GCC 7+, Clang 5+, MSVC 2017+)
- **CMake** 3.15 or higher
- **Git** (for cloning)

### Installation

```bash
# Clone the repository
git clone https://github.com/baimamboukar/neural-networks-in-cpp.git
cd neural-networks-in-cpp

# Create build directory
mkdir build && cd build

# Configure with CMake
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build the project
make

# Run the main application
./main
```

### Build Configurations

```bash
# Debug build (with debug symbols)
cmake .. -DCMAKE_BUILD_TYPE=Debug
make

# Release build (optimized)
cmake .. -DCMAKE_BUILD_TYPE=Release
make

# Coverage build (for code coverage analysis)
cmake .. -DCMAKE_BUILD_TYPE=Coverage
make
make coverage
```

### Running Tests

```bash
# Build and run all tests
make test

# Generate code coverage report
make coverage

# Generate HTML documentation
make doc
```

---

## 💻 Usage Example

```cpp
#include "layers.h"
#include "activations.h"
#include "losses.h"
#include <iostream>

int main() {
    // Create a simple neural network
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

    // Forward pass
    // auto output = model.forward(input);

    // Compute loss
    // auto loss_value = loss.compute(output, target);

    // Backward pass
    // auto gradients = model.backward(loss_gradient);

    std::cout << "Neural Network initialized successfully!" << std::endl;

    return 0;
}
```

---

## 🛠️ Components

### Layers

| Layer Type | Description |
|------------|-------------|
| **Dense** | Fully connected layer with weights and biases |
| **Linear** | Linear transformation layer |

### Activation Functions

| Activation | Use Case | Range |
|------------|----------|-------|
| **Sigmoid** | Binary classification, gates in RNNs | $(0, 1)$ |
| **Tanh** | Hidden layers, centered data | $(-1, 1)$ |
| **ReLU** | Most hidden layers, default choice | $[0, \infty)$ |
| **GeLU** | Transformer models, modern architectures | $(-0.17, \infty)$ |
| **Softmax** | Multi-class classification output | $(0, 1)$ with $\sum = 1$ |

### Loss Functions

| Loss | Task Type |
|------|-----------|
| **MSE** | Regression problems |
| **RMSE** | Regression with same units as output |
| **Cross-Entropy** | Classification problems |

---

## 🧪 Testing

This project uses the [doctest](https://github.com/doctest/doctest) testing framework for unit tests.

```bash
# Run all tests
cd build
make test

# Run tests with verbose output
./tests/cpp-test --success
```

---

## 📊 Performance

C++ offers significant performance advantages for neural network computations:

- **Memory Management**: Direct control over memory allocation and layout
- **SIMD Optimization**: Potential for vectorization with modern CPUs
- **Zero-Copy Operations**: Efficient tensor operations without GIL constraints
- **Compile-Time Optimization**: Template metaprogramming for optimal code generation

---

## 🗺️ Roadmap

- [ ] Implement Convolutional layers (Conv2D, MaxPool2D)
- [ ] Add optimization algorithms (SGD, Adam, RMSprop)
- [ ] Matrix operations library (or integration with Eigen)
- [ ] GPU acceleration with CUDA/OpenCL
- [ ] Serialization (save/load trained models)
- [ ] More advanced architectures (RNN, LSTM, Transformer)
- [ ] Data loading utilities
- [ ] Benchmarking suite

---

## 🤝 Contributing

Contributions are welcome! Whether you're fixing bugs, improving documentation, or proposing new features, your help is appreciated.

### How to Contribute

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/AmazingFeature`)
3. **Commit** your changes (`git commit -m 'Add some AmazingFeature'`)
4. **Push** to the branch (`git push origin feature/AmazingFeature`)
5. **Open** a Pull Request

### Guidelines

- Follow the existing code style and conventions
- Write clear, commented code
- Add tests for new features
- Update documentation as needed
- Be respectful and constructive in discussions

---

## 📚 Resources & References

- [Deep Learning Book](https://www.deeplearningbook.org/) by Ian Goodfellow
- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/) by Michael Nielsen
- [CS231n: Convolutional Neural Networks](http://cs231n.stanford.edu/) - Stanford
- [Mathematics for Machine Learning](https://mml-book.github.io/)

---

## 📝 License

This project is licensed under the **Unlicense** - see the [LICENSE](LICENSE) file for details.

This means the code is in the public domain and you can use it however you want, with no restrictions.

---

## 👨‍💻 Author

**Baimam Boukar Jean Jacques**

- 🎓 Master's in IT, Applied Machine Learning @ Carnegie Mellon University Africa
- 🔬 Research interests: ML for Space Systems, Earth Observation, Climate Modeling
- 🌐 GitHub: [@baimamboukar](https://github.com/baimamboukar)
- 💼 LinkedIn: [baimamboukar](https://www.linkedin.com/in/baimamboukar/)
- 📧 Email: [Contact](mailto:your-email@example.com)

---

## 🌟 Acknowledgments

- Inspired by the simplicity of educational ML frameworks
- Built on modern C++ best practices
- Thanks to the open-source community for tools and libraries

---

<div align="center">

**⭐ Star this repo if you find it helpful!**

Made with ❤️ and C++

</div>
