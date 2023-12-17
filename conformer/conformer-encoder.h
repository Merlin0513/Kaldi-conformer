//
// Created by Wenqing Yang on 12/13/23.
//

#ifndef KALDI_MASTER_CONFORMER_ENCODER_H
#define KALDI_MASTER_CONFORMER_ENCODER_H

#include "conformer-feedforward.h"
#include "conformer-attention.h"
#include "conformer-convolution.h"
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

        class ConformerBlock {
        public:
            ConformerBlock(int encoder_dim = 512, int num_heads = 8, int feed_forward_expansion_factor = 4,
                           int conv_expansion_factor = 2, BaseFloat feed_forward_dropout_p = 0.1,
                           BaseFloat attention_dropout_p = 0.1, BaseFloat conv_dropout_p = 0.1,
                           int conv_kernel_size = 31, bool half_step_residual = true);
            void Forward(const Matrix<BaseFloat> &input, Matrix<BaseFloat> *output);

        private:
            FeedForwardModule feed_forward_module_1_;
            MultiHeadedSelfAttentionModule attention_module_;
            ConformerConvModule conv_module_;
            FeedForwardModule feed_forward_module_2_;
            LayerNorm layer_norm_;
            BaseFloat feed_forward_residual_factor_;
        };

        class ConformerEncoder {
        public:
            ConformerEncoder(int input_dim = 80, int encoder_dim = 512, int num_layers = 17,
                             int num_attention_heads = 8, int feed_forward_expansion_factor = 4,
                             int conv_expansion_factor = 2, BaseFloat input_dropout_p = 0.1,
                             BaseFloat feed_forward_dropout_p = 0.1, BaseFloat attention_dropout_p = 0.1,
                             BaseFloat conv_dropout_p = 0.1, int conv_kernel_size = 31, bool half_step_residual = true);
            void Forward(const Matrix<BaseFloat>& inputs, int input_length,
                         Matrix<BaseFloat>* outputs, std::vector<int>* output_lengths);

        private:
            Conv2dSubsampling conv_subsample_;
            Linear input_projection_;
            std::vector<ConformerBlock> layers_;
        };

    }
}

#endif //KALDI_MASTER_CONFORMER_ENCODER_H
