# Conformer Model Implementation in Kaldi

## Kaldi-conformer

Wenqing Yang (wy2374@columbia.edu)

### Overview

This project presents an implementation of the Conformer model, a state-of-the-art speech recognition architecture, using the Kaldi speech recognition toolkit. The Conformer model combines convolutional neural networks with self-attention mechanisms to achieve superior performance in automatic speech recognition tasks. Currently, the Conformer model stands at the forefront of End-to-End Automatic Speech Recognition (E2E ASR) technology. It's acclaimed for its state-of-the-art architecture, which innovatively merges various modules, delivering outstanding performance in benchmarks. So far, the implementation of the Conformer model has predominantly been in PyTorch, a popular machine learning framework. In contrast, Kaldi as another highly efficient toolkit for speech recognition, offers a different approach. Known for its robustness and versatility in speech recognition tasks, Kaldi has not yet been utilized to implement the Conformer model. This presents a unique opportunity to explore the integration of this advanced model into the Kaldi toolkit. This project aims to bridge this gap by detailing the integration of the Conformer model within the Kaldi framework.

### Features

![Conformer Architecture](./img/conformer.png)

Implementation of Conformer blocks with multi-head self-attention and convolution modules.

Efficient convolutional subsampling for input data processing.

Integration with Kaldi's robust speech processing and GPU-accelerated computation capabilities.

- Convolutional Subsampling:

Convolutional subsampling is a critical part of the Conformer model, reducing the temporal resolution of the input while increasing the feature dimension. In Kaldi, this was achieved through a custom Conv2dSubsampling component. This component consists of two consecutive convolutional layers, each followed by ReLU activation. The layers employ a kernel size of 3 and a stride of 2, effectively reducing the input sequence length by a factor of 4. The subsampling layer outputs a transformed feature matrix ready for subsequent processing by the Conformer blocks.

- Conformer Blocks:

The core of the Conformer model is its blocks, each comprising various specialized components that collectively enhance its ability to process sequential data effectively. In our Kaldi implementation, each Conformer block is meticulously engineered to include:

**a). Feed Forward Module:**

![Feed Foward Module](./img/feed_forward.png)

Processes the output from the self-attention mechanism, consisting of two linear layers with a nonlinear activation function in between. Optimized in Kaldi for large-scale matrix operations. The expansion factor increases the dimensionality of the intermediate representation, capturing more complex features.

**b). Multi-Head Self Attention Module:** 

![Multi-Head Self Attention Module](./img/attention.png)

Captures contextual relationships within input sequences. Parallelizes the attention mechanism across multiple heads, each computing scaled dot-product attention. Their outputs are concatenated and linearly transformed. Optimized for efficiency, leveraging CUDA for large-scale data.

**c).Convolution Module:**

![Convolution Module](./img/convolution.png)

Captures local features within the input sequence. Consists of a depthwise separable convolution layer, followed by batch normalization and Swish activation. This structure is computationally efficient and effective for time-series data processing.
These components work together within each Conformer block, contributing to the model's ability to process and understand complex speech patterns effectively.

- Activations and Utils:

a). Utility functions and classes play a vital role in our implementation:

b). Custom Activation Functions: Includes Swish and Gated Linear Unit (GLU), implemented as standalone components in Kaldi.
Weight Initialization and Normalization Functions: Crafted to ensure that the model adheres to the expected behavior of Conformer.


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

### Content of each source and head files:
**conformer-activations:**

a). Swish -- activations implemented by C++ and Kaldi

b). GLU -- activations implemented by C++ and Kaldi

**conformer-attention:**

a). RelativeMultiHeadAttentionComponent -- used to 

b). MultiHeadAttentionModule --

**conformer-convolution:**

a). Conv1d --

b). Conv2d --

c). DepthwiseConv1d --

d). PointwiseConv1d --

e). ConformerConvModule --

f). Conv2dSubampling --

**conformer-embedding:**

a). PositionalEncoding -- 

**conformer-encoder:**

a). ConformerBlock --

b). ConformerEncoder --

**conformer-feedforward:**

a). FeedForwardModule --

**conformer-models:**

a). Conformer --

**conformer-utils:**

a). LayerNorm
b). BatchNorm1d
c). Dropout
d). ReLU
e). Linear
f). CalculateOutputLength
g). ApplyMask
h). AddBias
i). AddMatrices

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



