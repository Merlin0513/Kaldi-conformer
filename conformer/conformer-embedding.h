#ifndef KALDI_CONFORMER_POSITIONAL_ENCODING_H_
#define KALDI_CONFORMER_POSITIONAL_ENCODING_H_

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

        class PositionalEncoding {
        public:
            PositionalEncoding(int d_model, int max_len);

            void Forward(int seq_length, kaldi::Matrix<BaseFloat>* pos_encoding);

        private:
            int d_model_;
            kaldi::Matrix<BaseFloat> encoding_;
        };

    }
}

#endif  // KALDI_CONFORMER_POSITIONAL_ENCODING_H_

