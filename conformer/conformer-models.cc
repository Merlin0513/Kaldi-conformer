#include "conformer-models.h"
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
    namespace nnet3 {

        Conformer::Conformer(int num_classes, int input_dim = 80, int encoder_dim = 512, int num_encoder_layers = 17,
                             int num_attention_heads = 8, int feed_forward_expansion_factor = 4,
                             int conv_expansion_factor = 2, BaseFloat input_dropout_p = 0.1,
                             BaseFloat feed_forward_dropout_p = 0.1, BaseFloat attention_dropout_p = 0.1,
                             BaseFloat conv_dropout_p = 0.1, int conv_kernel_size = 31, bool half_step_residual = true)
                : encoder_(input_dim, encoder_dim, num_encoder_layers, num_attention_heads,
                           feed_forward_expansion_factor, conv_expansion_factor, input_dropout_p,
                           feed_forward_dropout_p, attention_dropout_p, conv_dropout_p,
                           conv_kernel_size, half_step_residual),
                  fc_(encoder_dim, num_classes, bias = false) {
        }

        void Conformer::Forward(const Matrix <BaseFloat> &inputs, const std::vector<int> &input_lengths,
                                Matrix <BaseFloat> *outputs, std::vector<int> *output_lengths) {
            Matrix <BaseFloat> encoder_outputs;
            encoder_.Forward(inputs, 1 /* input length */, &encoder_outputs, output_lengths);
            Matrix <BaseFloat> linear_outputs(encoder_outputs.NumRows(), fc_.OutputDim());
            fc_.Forward(encoder_outputs, &linear_outputs);
            ApplyLogSoftmax(&linear_outputs);
            outputs->Resize(linear_outputs.NumRows(), linear_outputs.NumCols());
            outputs->CopyFromMat(linear_outputs);
        }

        void Conformer::ApplyLogSoftmax(Matrix<BaseFloat>* linear_outputs) {
            for (int32 r = 0; r < linear_outputs->NumRows(); ++r) {
                BaseFloat row_max = linear_outputs->Row(r).Max();
                BaseFloat sum = 0.0;
                for (int32 c = 0; c < linear_outputs->NumCols(); ++c) {
                    (*linear_outputs)(r, c) = exp((*linear_outputs)(r, c) - row_max);
                    sum += (*linear_outputs)(r, c);
                }
                for (int32 c = 0; c < linear_outputs->NumCols(); ++c) {
                    (*linear_outputs)(r, c) = log((*linear_outputs)(r, c) / sum);
                }
            }
        }

        void Conformer::Backward(const Matrix<BaseFloat> &input, const Matrix<BaseFloat> &output_grad,
                                 Matrix<BaseFloat> *input_grad, Matrix<BaseFloat> *param_grad) {
            Matrix<BaseFloat> encoder_output_grad;
            fc_.Backward(encoder_outputs, output_grad, &encoder_output_grad, param_grad);
            encoder_.Backward(input, encoder_output_grad, input_grad, param_grad);
        }

    }
}