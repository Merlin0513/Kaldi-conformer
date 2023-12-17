#include "conformer-encoder.h"
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

        ConformerBlock::ConformerBlock(int encoder_dim = 512, int num_heads = 8, int feed_forward_expansion_factor = 4,
                                       int conv_expansion_factor = 2, BaseFloat feed_forward_dropout_p = 0.1,
                                       BaseFloat attention_dropout_p = 0.1, BaseFloat conv_dropout_p = 0.1,
                                       int conv_kernel_size = 31, bool half_step_residual = true)
                : feed_forward_module_1_(encoder_dim, feed_forward_expansion_factor, feed_forward_dropout_p),
                  attention_module_(encoder_dim, num_heads, attention_dropout_p),
                  conv_module_(encoder_dim, conv_kernel_size, conv_expansion_factor, conv_dropout_p),
                  feed_forward_module_2_(encoder_dim, feed_forward_expansion_factor, feed_forward_dropout_p),
                  layer_norm_(encoder_dim),
                  feed_forward_residual_factor_(half_step_residual ? 0.5 : 1.0) {
        }

        void ConformerBlock::Forward(const Matrix <BaseFloat> &input, Matrix <BaseFloat> *output) {
            CuMatrix<BaseFloat> temp, temp_output;
            feed_forward_module_1_.Forward(input, &temp);
            temp.Scale(feed_forward_residual_factor_);
            temp.AddMat(1.0, input);
            attention_module_.Forward(temp, &temp_output);
            temp_output.AddMat(1.0, temp);
            conv_module_.Forward(temp_output, &temp);
            temp.AddMat(1.0, temp_output);
            feed_forward_module_2_.Forward(temp, &temp_output);
            temp_output.Scale(feed_forward_residual_factor_);
            temp_output.AddMat(1.0, temp);
            layer_norm_.Forward(temp_output, output);
        }

        ConformerEncoder::ConformerEncoder(int input_dim = 80, int encoder_dim = 512, int num_layers = 17,
                                           int num_attention_heads = 8, int feed_forward_expansion_factor = 4,
                                           int conv_expansion_factor = 2, BaseFloat input_dropout_p = 0.1,
                                           BaseFloat feed_forward_dropout_p = 0.1, BaseFloat attention_dropout_p = 0.1,
                                           BaseFloat conv_dropout_p = 0.1, int conv_kernel_size = 31, bool half_step_residual = true)
                : conv_subsample_(1, encoder_dim, 3),
                  input_projection_(encoder_dim * (((input_dim - 1) / 2 - 1) / 2), encoder_dim, input_dropout_p) {
            for (int i = 0; i < num_layers; ++i) {
                layers_.push_back(ConformerBlock(encoder_dim, num_attention_heads,
                                                 feed_forward_expansion_factor, conv_expansion_factor,
                                                 feed_forward_dropout_p, attention_dropout_p,
                                                 conv_dropout_p, conv_kernel_size, half_step_residual));
            }
        }

        void ConformerEncoder::Forward(const Matrix<BaseFloat>& inputs, 1 /* input length */,
                                       Matrix<BaseFloat>* outputs, std::vector<int>* output_lengths) {
            Matrix<BaseFloat> current_outputs;
            conv_subsample_.Forward(inputs, input_lengths, &current_outputs, output_lengths);
            input_projection_.Forward(current_outputs, &current_outputs);
            for (size_t i = 0; i < layers_.size(); ++i) {
                layers_[i].Forward(current_outputs, &current_outputs);
            }
            outputs->CopyFromMat(current_outputs);
        }

    }
}