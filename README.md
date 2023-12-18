# Conformer Model Implementation in Kaldi

## Kaldi-conformer

Wenqing Yang (wy2374@columbia.edu)

### Overview

This project presents an implementation of the Conformer model, a state-of-the-art speech recognition architecture, using the Kaldi speech recognition toolkit. The Conformer model combines convolutional neural networks with self-attention mechanisms to achieve superior performance in automatic speech recognition tasks. Currently, the Conformer model stands at the forefront of End-to-End Automatic Speech Recognition (E2E ASR) technology. It's acclaimed for its state-of-the-art architecture, which innovatively merges various modules, delivering outstanding performance in benchmarks. So far, the implementation of the Conformer model has predominantly been in PyTorch, a popular machine learning framework. In contrast, Kaldi as another highly efficient toolkit for speech recognition, offers a different approach. Known for its robustness and versatility in speech recognition tasks, Kaldi has not yet been utilized to implement the Conformer model. This presents a unique opportunity to explore the integration of this advanced model into the Kaldi toolkit. This project aims to bridge this gap by detailing the integration of the Conformer model within the Kaldi framework.

### Features

Implementation of Conformer blocks with multi-head self-attention and convolution modules.
Efficient convolutional subsampling for input data processing.
Integration with Kaldi's robust speech processing and GPU-accelerated computation capabilities.

\subsection{Convolutional Subsampling}

Convolutional subsampling is a critical part of the Conformer model, reducing the temporal resolution of the input while increasing the feature dimension. In Kaldi, this was achieved through a custom Conv2dSubsampling component. This component consists of two consecutive convolutional layers, each followed by ReLU activation. The layers employ a kernel size of 3 and a stride of 2, effectively reducing the input sequence length by a factor of 4. The subsampling layer outputs a transformed feature matrix ready for subsequent processing by the Conformer blocks.

\subsection{Conformer Blocks}

The core of the Conformer model is its blocks, each comprising various specialized components that collectively enhance its ability to process sequential data effectively. In our Kaldi implementation, each Conformer block is meticulously engineered to include a multi-head self-attention module, a convolution module, and two pointwise convolutional layers, all integrated to perform complex transformations on the input data.

\textbf{Feed Forward Module}: The Feed Forward Module in each Conformer block is a critical component that processes the output from the self-attention mechanism. It typically consists of two linear layers with a nonlinear activation function in between. In Kaldi, this module is carefully optimized to handle large-scale matrix operations efficiently. The expansion factor in the feed-forward module increases the dimensionality of the intermediate representation, allowing the network to capture more complex features before projecting back to the original dimension.

\textbf{Multi-Head Self Attention Module}: The Multi-Head Self Attention Module is pivotal for capturing the contextual relationships within the input sequences. This module in Kaldi is designed to parallelize the attention mechanism across multiple 'heads', enabling the model to focus on different parts of the input sequence simultaneously. Each head computes scaled dot-product attention, and their outputs are concatenated and linearly transformed to produce the final output of the module. The implementation ensures that each attention head operates efficiently, particularly in leveraging CUDA optimizations for handling large-scale data.

\textbf{Convolution Module}: The Convolution Module in a Conformer block is essential for capturing local features within the input sequence. This module, as implemented in Kaldi, consists of a depthwise separable convolution layer, which is a computationally efficient alternative to standard convolutions. This layer applies a depthwise convolution followed by a pointwise convolution, allowing the model to integrate local information across the sequence. The module also includes batch normalization and Swish activation to stabilize and enhance the learning process. The depthwise separable convolution is particularly advantageous in Kaldi due to its reduced computational burden and its effectiveness in processing time-series data.

These components work in unison within each Conformer block, contributing to the model's ability to effectively process and understand complex speech patterns. The integration of these modules in Kaldi, with a focus on computational efficiency and gradient flow, ensures that the Conformer model is not only effective in its task but also optimized for performance in large-scale speech recognition applications.

\subsection{Activations and Utils}

Utility functions and classes play a vital role in our implementation. This includes custom activation functions like Swish and Gated Linear Unit (GLU), which were implemented as standalone components in Kaldi. Additionally, functions for weight initialization and normalization were crafted to ensure that the model conforms to the expected behavior of Conformer.

### Structure

conformer % tree
.
├── CMakeLists.txt
├── Makefile    
├── conformer-activation.cc
├── conformer-activation.h
├── conformer-attention.cc
├── conformer-attention.h
├── conformer-convolution.cc
├── conformer-convolution.h
├── conformer-embedding.cc
├── conformer-embedding.h
├── conformer-encoder.cc
├── conformer-encoder.h
├── conformer-feedforward.cc
├── conformer-feedforward.h
├── conformer-models.cc
├── conformer-models.h
├── conformer-utils.cc
├── conformer-utils.h
├── gen_cmakelist.py
├── main.cc
└── main.h

21 files

### Requirements

Kaldi Speech Recognition Toolkit 
C++ Compiler (e.g., g++)
CUDA Toolkit (for GPU acceleration, optional)
Additional dependencies listed in Makefile/CMakeList.txt (aligned with Kaldi/src/.).

### Installation

Clone Kaldi repository:

```
git clone https://github.com/kaldi-asr/kaldi.git
cd kaldi
git remote add upstream https://github.com/kaldi-asr/kaldi.git
```

Build Kaldi repository:

```
cd tools
make
extras/check_dependencies.sh

cd ..
./configure
make depend
make
```

Clone kaldi-conformer repository:
```
git clone https://github.com/Merlin0513/Kaldi-conformer.git
cd Kaldi-conformer
```

### Usage

Run the compiled Conformer model:
```
./main_program
```

### Example (Pending)



