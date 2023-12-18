# Conformer Model Implementation in Kaldi

## Kaldi-conformer

Wenqing Yang (wy2374@columbia.edu)

### Overview

This project presents an implementation of the Conformer model, a state-of-the-art speech recognition architecture, using the Kaldi speech recognition toolkit. The Conformer model combines convolutional neural networks with self-attention mechanisms to achieve superior performance in automatic speech recognition tasks. Currently, the Conformer model stands at the forefront of End-to-End Automatic Speech Recognition (E2E ASR) technology. It's acclaimed for its state-of-the-art architecture, which innovatively merges various modules, delivering outstanding performance in benchmarks. So far, the implementation of the Conformer model has predominantly been in PyTorch, a popular machine learning framework. In contrast, Kaldi as another highly efficient toolkit for speech recognition, offers a different approach. Known for its robustness and versatility in speech recognition tasks, Kaldi has not yet been utilized to implement the Conformer model. This presents a unique opportunity to explore the integration of this advanced model into the Kaldi toolkit. This project aims to bridge this gap by detailing the integration of the Conformer model within the Kaldi framework.

### Features

Implementation of Conformer blocks with multi-head self-attention and convolution modules.
Efficient convolutional subsampling for input data processing.
Integration with Kaldi's robust speech processing and GPU-accelerated computation capabilities.

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



