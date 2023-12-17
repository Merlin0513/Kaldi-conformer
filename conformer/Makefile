all:


include $(KALDI_ROOT)/kaldi.mk

# Name of the library to be built
LIBNAME = kaldi-conformer

# Source files
SRC = conformer-activation.cc conformer-attention.cc conformer-convolution.cc \
      conformer-embedding.cc conformer-encoder.cc conformer-feedforward.cc \
      conformer-models.cc conformer-utils.cc

# Object files (replace .cc with .o)
OBJFILES = $(SRC:.cc=.o)

LIBNAME = kaldi-conformer

# Additional libraries to link with (from Kaldi)
ADDLIBS = ../matrix/kaldi-matrix.a \
          ../util/kaldi-util.a \
          ../base/kaldi-types.a ../matrix/matrix-lib.a ../cudamatrix/cu-matrix-lib.a ../cudamatrix/cu-matrix.a ../cudamatrix/cu-math.a ../base/kaldi-error.a ../base/kaldi-math.a

# Default target
all: $(LIBNAME)

# Include Kaldi's default rules
include $(KALDI_ROOT)/makefiles/default_rules.mk

