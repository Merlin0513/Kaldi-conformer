//
// Created by Wenqing Yang on 12/14/23.
//

#ifndef KALDI_MASTER_CONFORMER_MODELS_H
#define KALDI_MASTER_CONFORMER_MODELS_H

#include "conformer-encoder.h"
#include "conformer-utils.h"
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

        class Conformer {
        public:
            Conformer(int num_classes, int input_dim = 80, int encoder_dim = 512, int num_encoder_layers = 17,
                      int num_attention_heads = 8, int feed_forward_expansion_factor = 4,
                      int conv_expansion_factor = 2, BaseFloat input_dropout_p = 0.1,
                      BaseFloat feed_forward_dropout_p = 0.1, BaseFloat attention_dropout_p = 0.1,
                      BaseFloat conv_dropout_p = 0.1, int conv_kernel_size = 31, bool half_step_residual = true);

            void Forward(const Matrix<BaseFloat>& inputs, const std::vector<int>& input_lengths,
                         Matrix<BaseFloat>* outputs, std::vector<int>* output_lengths);

            void Backward(const Matrix<BaseFloat>& input, const Matrix<BaseFloat>& output_grad,
                          Matrix<BaseFloat>* input_grad, Matrix<BaseFloat>* param_grad);

        private:
            ConformerEncoder encoder_;
            Linear fc_;

            void ApplyLogSoftmax(Matrix<BaseFloat>* linear_outputs);
        };

    }
}

#endif //KALDI_MASTER_CONFORMER_MODELS_H
