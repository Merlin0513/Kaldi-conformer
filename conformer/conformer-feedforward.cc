#include "conformer-feedforward.h"
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
        FeedForwardModule::FeedForwardModule(int input_dim = 512, int expansion_factor = 4, BaseFloat dropout_rate = 0.1)
                : input_dim_(input_dim),
                  expansion_factor_(expansion_factor),
                  dropout_rate_(dropout_rate),
                  output_dim_(input_dim * expansion_factor),
                  layer_norm_(input_dim),
                  linear1_(input_dim, input_dim * expansion_factor),
                  linear2_(input_dim * expansion_factor, input_dim),
                  dropout_(dropout_rate) {
        }

        void FeedForwardModule::Forward(const Matrix<BaseFloat>& input, Matrix<BaseFloat>* output) {
            KALDI_ASSERT(input.NumCols() == input_dim_);
            Matrix<BaseFloat> temp1(input.NumRows(), input_dim_); // Size as per layer_norm_ output
            Matrix<BaseFloat> temp2(input.NumRows(), output_dim_);
            Matrix<BaseFloat> temp3(input.NumRows(), output_dim_);
            Matrix<BaseFloat> temp4(input.NumRows(), output_dim_);
            Matrix<BaseFloat> temp5(input.NumRows(), input_dim_);  // Size as per linear2_ output
            layer_norm_.Forward(input, &temp1);
            linear1_.Forward(temp1, &temp2);
            Swish(temp2, &temp3);
            Dropout(temp3, &temp4, true);
            linear2_.Forward(temp4, &temp5);
            Dropout(temp5, output, true);
        }

        void FeedForwardModule::Backward(const Matrix<BaseFloat>& input,
                                         const Matrix<BaseFloat>& output_grad,
                                         Matrix<BaseFloat>* input_grad) {
            KALDI_ASSERT(input.NumCols() == input_dim_ && output_grad.NumCols() == output_dim_);
            Matrix<BaseFloat> temp1_grad(output_grad.NumRows(), output_dim_);
            Matrix<BaseFloat> temp2_grad(output_grad.NumRows(), output_dim_);
            Matrix<BaseFloat> temp3_grad(output_grad.NumRows(), output_dim_);
            Matrix<BaseFloat> temp4_grad(output_grad.NumRows(), input_dim_);
            Matrix<BaseFloat> temp5_grad(output_grad.NumRows(), input_dim_);
            dropout_.Backward(output_grad, &temp1_grad);
            linear2_.Backward(temp1_grad, &temp2_grad);
            dropout_.Backward(temp2_grad, &temp3_grad);
            linear1_.Backward(temp3_grad, &temp4_grad);
            layer_norm_.Backward(temp4_grad, input_grad);
        }

    }
}