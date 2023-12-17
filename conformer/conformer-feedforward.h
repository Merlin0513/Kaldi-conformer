//
// Created by Wenqing Yang on 12/12/23.
//

#ifndef KALDI_MASTER_CONFORMER_FEEDFORWARD_H
#define KALDI_MASTER_CONFORMER_FEEDFORWARD_H

#include "conformer-utils.h"
#include "conformer-activation.h"
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

        class FeedForwardModule {
        public:
            FeedForwardModule(int input_dim = 512, int expansion_factor = 4, BaseFloat dropout_rate = 0.1);
            void Forward(const Matrix<BaseFloat>& input, Matrix<BaseFloat>* output);
            void Backward(const Matrix<BaseFloat>& input,
                          const Matrix<BaseFloat>& output_grad,
                          Matrix<BaseFloat>* input_grad);
        private:
            int input_dim_;
            int expansion_factor_;
            BaseFloat dropout_rate_;
            int output_dim_;

            LayerNorm layer_norm_;
            Linear linear1_;
            Linear linear2_;
            Dropout dropout_;
        };

    }
}

#endif //KALDI_MASTER_CONFORMER_FEEDFORWARD_H
