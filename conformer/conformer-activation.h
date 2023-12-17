#ifndef KALDI_CONFORMER_ACTIVATION_H_
#define KALDI_CONFORMER_ACTIVATION_H_

#include "conformer-utils.h"
#include "conformer-embedding.h"
#include "matrix/kaldi-matrix.h"
#include "cudamatrix/cu-matrix-lib.h"
#include "base/kaldi-types.h"
#include "matrix/matrix-lib.h"
#include "cudamatrix/cu-matrix.h"
#include "cudamatrix/cu-math.h"
#include "base/kaldi-math.h"
#include "base/kaldi-error.h"
#include <vector>
#include <cmath>
#include <random>

namespace kaldi {
    namespace conformer {

        void Swish(const CuMatrixBase<BaseFloat>& input, CuMatrixBase<BaseFloat>* output);

        BaseFloat Sigmoid(BaseFloat x);

        void GLU(const CuMatrixBase<BaseFloat>& input, CuMatrixBase<BaseFloat>* output, int dim);

    }
}

#endif