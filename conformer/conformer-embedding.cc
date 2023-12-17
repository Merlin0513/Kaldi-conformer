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

        PositionalEncoding::PositionalEncoding(int d_model, int max_len)
                : d_model_(d_model), encoding_(max_len, d_model) {
            double inv_denom = 1.0 / std::pow(10000.0, 2.0 / d_model);
            for (int pos = 0; pos < max_len; ++pos) {
                for (int i = 0; i < d_model; ++i) {
                    double angle = pos * std::pow(inv_denom, i / 2.0);  // Division by 2.0 for alternate sin and cos
                    if (i % 2 == 0) {
                        encoding_(pos, i) = std::sin(angle);
                    } else {
                        encoding_(pos, i) = std::cos(angle);
                    }
                }
            }
        }

        void PositionalEncoding::Forward(int seq_length, kaldi::Matrix<BaseFloat> *pos_encoding) {
            KALDI_ASSERT(seq_length <= encoding_.NumRows());
            pos_encoding->Resize(seq_length, d_model_);
            pos_encoding->CopyFromMat(encoding_.Range(0, seq_length, 0, d_model_));
        }

    }
}