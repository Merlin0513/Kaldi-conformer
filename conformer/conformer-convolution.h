//
// Created by Wenqing Yang on 12/4/23.
//

#ifndef KALDI_MASTER_CONFORMER_CONVOLUTION_H_
#define KALDI_MASTER_CONFORMER_CONVOLUTION_H_

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

        class Conv1d {
        public:
            Conv1d(int input_channels, int output_channels, int kernel_size,
                   int stride, int padding, int groups, bool bias);
            void Forward(const CuMatrixBase<BaseFloat>& input, CuMatrixBase<BaseFloat>* output);
            void Backward(const CuMatrixBase<BaseFloat>& in, const CuMatrixBase<BaseFloat>& out,
                          const CuMatrixBase<BaseFloat>& out_diff,
                          CuMatrixBase<BaseFloat>* in_diff,
                          CuMatrixBase<BaseFloat>* weights_diff, CuVectorBase<BaseFloat>* bias_diff) const;
        private:
            int input_channels_;
            int output_channels_;
            int kernel_size_;
            int stride_;
            int padding_;
            int groups_;
            bool bias_;

            CuMatrix<BaseFloat> weights_;
            CuVector<BaseFloat> bias_;
        };

        class Conv2d {
        public:
            Conv2d(int in_channels, int out_channels, int kernel_size, int stride);
            void Forward(const CuMatrixBase<BaseFloat>& input, CuMatrixBase<BaseFloat>* output);
            void Backward(const CuMatrixBase<BaseFloat>& input,
                          const CuMatrixBase<BaseFloat>& grad_output,
                          CuMatrixBase<BaseFloat>* grad_input,
                          CuMatrixBase<BaseFloat>* grad_weights);
        private:
            int in_channels_;
            int out_channels_;
            int kernel_size_;
            int stride_;
            CuMatrix<BaseFloat> weights_;
            CuVector<BaseFloat> bias_;
        };

        class DepthwiseConv1d {
        public:
            DepthwiseConv1d(int input_channels, int output_channels, int kernel_size,
                            int stride = 1, int padding = 0, bool bias = false);
            void Forward(const CuMatrixBase<BaseFloat>& input, CuMatrixBase<BaseFloat>* output);
            void backward(const CuMatrixBase<BaseFloat> &in,
                          const CuMatrixBase<BaseFloat> &out_diff,
                          CuMatrixBase<BaseFloat> *in_diff,
                          CuMatrixBase<BaseFloat> *weights_diff,
                          CuVectorBase<BaseFloat> *bias_diff) const;
        private:
            int input_channels_;
            int output_channels_;
            int kernel_size_;
            int stride_;
            int padding_;
            bool bias_;
            Conv1d conv_;
        };

        class PointwiseConv1d {
        public:
            PointwiseConv1d(int input_channels, int output_channels, int stride = 1, int padding = 0, bool bias = true);
            void Forward(const CuMatrixBase<BaseFloat>& input, CuMatrixBase<BaseFloat>* output);
            void Backward(const CuMatrixBase<BaseFloat> &in,
                          const CuMatrixBase<BaseFloat> &out_diff,
                          CuMatrixBase<BaseFloat> *in_diff,
                          CuMatrixBase<BaseFloat> *weights_diff,
                          CuVectorBase<BaseFloat> *bias_diff) const;
        private:
            Conv1d conv_;
        };


        class ConformerConvModule {
        public:
            ConformerConvModule(int in_channels, int kernel_size = 31, int expansion_factor = 2, BaseFloat dropout_rate = 0.1);
            void Forward(const ChunkInfo &in_info, const ChunkInfo &out_info,
                         const CuMatrixBase<BaseFloat> &in, CuMatrixBase<BaseFloat> *out) const;
            void Backward(const ChunkInfo &in_info, const ChunkInfo &out_info,
                          const CuMatrixBase<BaseFloat> &in_value,
                          const CuMatrixBase<BaseFloat> &out_value,
                          const CuMatrixBase<BaseFloat> &out_deriv,
                          Component *to_update, CuMatrixBase<BaseFloat> *in_deriv) const;
        private:
            int in_channels_;
            int kernel_size_;
            int expansion_factor_;
            BaseFloat dropout_rate_;
            LayerNorm layer_norm_;
            PointwiseConv1d pointwise_conv1d_1_;
            DepthwiseConv1d depthwise_conv1d_;
            BatchNorm1d batch_norm_;
            PointwiseConv1d pointwise_conv1d_2_;
            Dropout dropout_;
        };

        class Conv2dSubsampling {
        public:
            Conv2dSubsampling(int in_channels, int out_channels);
            void Forward(const Matrix<BaseFloat>& inputs,
                         const std::vector<int>& input_lengths,
                         Matrix<BaseFloat>& outputs,
                         std::vector<int>& output_lengths);
            void Backward(const Matrix<BaseFloat>& input,
                          const Matrix<BaseFloat>& grad_output,
                          Matrix<BaseFloat>& grad_input);
        private:
            int in_channels_;
            int out_channels_;
            Conv2d conv1_;
            Conv2d conv2_;
            ReLU relu_;
    }
}

#endif
