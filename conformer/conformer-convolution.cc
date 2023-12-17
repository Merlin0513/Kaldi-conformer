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
#include "conformer-convolution.h"

// Conv1d()
// Conv2d
// DepthwiseConv1d()
// PointwiseConv1d()
// ConformerConvModule()
// Conv2dSubampling()

namespace kaldi {
    namespace conformer{

        Conv1d::Conv1d(int input_channels, int output_channels, int kernel_size,
                       int stride, int padding, int groups, bool bias)
                : input_channels_(input_channels), output_channels_(output_channels),
                  kernel_size_(kernel_size), stride_(stride), padding_(padding), groups_(groups) {
            KALDI_ASSERT(input_channels % groups == 0);
            int group_in_channels = input_channels / groups;
            weights_.Resize(output_channels, group_in_channels * kernel_size);
            weights_.SetRandn();
            if (bias) {
                bias_.Resize(output_channels);
                bias_.SetZero();
            }
        }

        void Conv1d::Forward(const CuMatrixBase<BaseFloat>& input, CuMatrixBase<BaseFloat>* output) {
            int num_frames = input.NumRows();
            int num_features = input.NumCols();
            KALDI_ASSERT(num_features == input_channels_);
            int output_length = (num_features - kernel_size_ + 2 * padding_) / stride_ + 1;
            output->Resize(num_frames, output_channels_ * output_length);

            for (int r = 0; r < num_frames; ++r) {
                for (int c = 0; c < output_length; ++c) {
                    for (int d = 0; d < output_channels_; ++d) {
                        BaseFloat sum = 0.0;
                        for (int k = 0; k < kernel_size_; ++k) {
                            int col_idx = c * stride_ - padding_ + k;
                            if (col_idx >= 0 && col_idx < num_features) {
                                for (int ch = 0; ch < input_channels_; ++ch) {
                                    int weight_idx = d * (input_channels_ * kernel_size_) + ch * kernel_size_ + k;
                                    sum += input(r, ch) * weights_(d, weight_idx);
                                }
                            }
                        }
                        if (bias) {
                            (*output)(r, c * output_channels_ + d) = sum + bias_(d);
                        } else {
                            (*output)(r, c * output_channels_ + d) = sum
                        }
                    }
                }
            }
        }

        void Conv1d::backward(const CuMatrixBase<BaseFloat> &in, const CuMatrixBase<BaseFloat> &out,
                              const CuMatrixBase<BaseFloat> &out_diff,
                              CuMatrixBase<BaseFloat> *in_diff,
                              CuMatrixBase<BaseFloat> *weights_diff, CuVectorBase<BaseFloat> *bias_diff) const {
            int num_samples = in.NumRows();
            int group_in_channels = input_channels_ / groups_;
            int signal_length = in.NumCols() / input_channels_;
            int output_length = CalculateOutputLength(signal_length, kernel_size_, stride_, padding_);
            // Initialize gradients to zero
            in_diff->SetZero();
            weights_diff->SetZero();
            bias_diff->SetZero();
            for (int g = 0; g < groups_; ++g) {
                for (int b = 0; b < num_samples; ++b) { // Loop over batches
                    for (int c = 0; c < group_in_channels; ++c) { // Loop over channels within a group
                        int channel_index = g * group_in_channels + c;
                        for (int t = 0; t < output_length; ++t) { // Loop over output time dimension
                            BaseFloat out_diff_val = out_diff(b, t * output_dim_ + channel_index);
                            for (int k = 0; k < kernel_size_; ++k) { // Loop over kernel
                                int input_index = t * stride_ - padding_ + k;
                                if (input_index >= 0 && input_index < signal_length) {
                                    int in_index = input_index * input_channels_ + channel_index;
                                    // Compute gradient for weights
                                    (*weights_diff)(channel_index, k) += out_diff_val * in(b, in_index);
                                    // Compute gradient for input
                                    (*in_diff)(b, in_index) += out_diff_val * weights_(channel_index, k);
                                }
                            }
                            if (bias) {
                                (*bias_diff)(channel_index) += out_diff_val;
                            }
                        }
                    }
                }
            }
        }

        Conv2d::Conv2d(int in_channels, int out_channels, int kernel_size, int stride)
                : in_channels_(in_channels), out_channels_(out_channels),
                  kernel_size_(kernel_size), stride_(stride) {
            int weight_rows = out_channels;
            int weight_cols = in_channels * kernel_size * kernel_size;
            weights_.Resize(weight_rows, weight_cols);
            RandGauss(0.0, 0.1, &weights_);
            if (bias) {
                bias_.Resize(out_channels);
                bias_.SetZero();
            }
        }

        void Conv2d::Forward(const CuMatrixBase<BaseFloat>& input, CuMatrixBase<BaseFloat>* output) {
            int H = input.NumRows(); // Height dimension
            int W = input.NumCols() / in_channels_; // Width dimension, assuming input is flattened
            int H_out = (H - kernel_size_) / stride_ + 1; // Output height
            int W_out = (W - kernel_size_) / stride_ + 1; // Output width
            output->Resize(H_out, W_out * out_channels_);
            for (int h = 0; h < H_out; ++h) {
                for (int w = 0; w < W_out; ++w) {
                    for (int c_out = 0; c_out < out_channels_; ++c_out) {
                        BaseFloat conv_sum = 0.0;
                        for (int c_in = 0; c_in < in_channels_; ++c_in) {
                            for (int kh = 0; kh < kernel_size_; ++kh) {
                                for (int kw = 0; kw < kernel_size_; ++kw) {
                                    int h_idx = h * stride_ + kh;
                                    int w_idx = w * stride_ + kw;
                                    if (h_idx < H && w_idx < W) {
                                        int input_idx = h_idx * (W * in_channels_) + w_idx * in_channels_ + c_in;
                                        int weight_idx = c_out * (in_channels_ * kernel_size_ * kernel_size_) + c_in * (kernel_size_ * kernel_size_) + kh * kernel_size_ + kw;
                                        conv_sum += input.Data()[input_idx] * weights_.Data()[weight_idx];
                                    }
                                }
                            }
                        }
                        if (bias) {
                            conv_sum += bias_(c_out);
                        } else {
                            (*output)(h, w * out_channels_ + c_out) = conv_sum;
                        }
                    }
                }
            }
        }

        void Conv2d::Backward(const CuMatrixBase<BaseFloat>& input,
                              const CuMatrixBase<BaseFloat>& grad_output,
                              CuMatrixBase<BaseFloat>* grad_input,
                              CuMatrixBase<BaseFloat>* grad_weights) {

            int32 num_rows = input.NumRows();
            int32 num_cols = input.NumCols();
            int32 output_dim = (num_cols - kernel_size_) / stride_ + 1;
            grad_input->Resize(num_rows, num_cols);
            grad_weights->Resize(out_channels_, in_channels_ * kernel_size_ * kernel_size_);
            for (int patch_start = 0; patch_start <= num_cols - kernel_size_; patch_start += stride_) {
                // Extract a patch from the input matrix
                CuSubMatrix<BaseFloat> patch(input, 0, num_rows, patch_start, kernel_size_);
                // Compute gradient w.r.t. weights
                // For each output channel
                for (int32 c = 0; c < out_channels_; ++c) {
                    // Gradient for this filter
                    CuSubMatrix<BaseFloat> grad_filter(*grad_weights, c, 1, 0, patch.NumCols());
                    grad_filter.AddMatMat(1.0, grad_output.ColRange(patch_start / stride_ * out_channels_ + c, 1), kNoTrans, patch, kNoTrans, 1.0);
                }
                // Compute gradient w.r.t. input (if needed)
                if (grad_input != NULL) {
                    CuMatrix<BaseFloat> expanded_grad_output;
                    grad_input->ColRange(patch_start, kernel_size_).AddMatMat(1.0, expanded_grad_output, kNoTrans, weights_, kNoTrans, 1.0);
                }
            }
        }

        DepthwiseConv1d::DepthwiseConv1d(int input_channels, int output_channels, int kernel_size,
                                         int stride = 1, int padding = 0, bool bias = false)
                : conv_(input_channels, output_channels, kernel_size, stride, padding, input_channels, bias) {
            KALDI_ASSERT(output_channels % input_channels == 0);
        }

        void DepthwiseConv1d::Forward(const CuMatrixBase<BaseFloat>& input, CuMatrixBase<BaseFloat>* output) {
            conv_.Forward(input, output);
        }

        void DepthwiseConv1d::Backward(const CuMatrixBase<BaseFloat> &in,
                                       const CuMatrixBase<BaseFloat> &out_diff,
                                       CuMatrixBase<BaseFloat> *in_diff,
                                       CuMatrixBase<BaseFloat> *weights_diff,
                                       CuVectorBase<BaseFloat> *bias_diff) const {
            int num_samples = in.NumRows(); // Number of samples (batch size)
            int input_channels = this->input_channels_;
            int signal_length = in.NumCols() / input_channels;
            int kernel_size = this->kernel_size_;
            int stride = this->stride_;
            int padding = this->padding_;
            // Loop over each sample in the batch
            for (int b = 0; b < num_samples; ++b) {
                // Loop over each channel
                for (int c = 0; c < input_channels; ++c) {
                    // Loop over each time step in the output gradient
                    for (int t = 0; t < out_diff.NumCols() / input_channels; ++t) {
                        // Calculate the position in the input
                        int t_in = t * stride - padding;
                        // Gradient w.r.t. bias
                        if (bias_diff != nullptr) {
                            (*bias_diff)(c) += out_diff(b, t * input_channels + c);
                        }
                        // Loop over the kernel
                        for (int k = 0; k < kernel_size; ++k) {
                            if (t_in + k >= 0 && t_in + k < signal_length) {
                                // Gradient w.r.t. input
                                (*in_diff)(b, (t_in + k) * input_channels + c) +=
                                        out_diff(b, t * input_channels + c) * this->weights_(c, k);
                                // Gradient w.r.t. weights
                                (*weights_diff)(c, k) +=
                                        out_diff(b, t * input_channels + c) * in(b, (t_in + k) * input_channels + c);
                            }
                        }
                    }
                }
            }
        }

        PointwiseConv1d::PointwiseConv1d(int input_channels, int output_channels, int stride = 1, int padding = 0, bool bias = true)
                : conv_(in_channels, out_channels, 1, 1, 0, 1, bias) {
        }

        void PointwiseConv1d::Forward(const CuMatrixBase<BaseFloat>& input, CuMatrixBase<BaseFloat>* output) {
            conv_.Forward(input, output);
        }

        void PointwiseConv1d::Backward(const CuMatrixBase<BaseFloat> &in,
                                       const CuMatrixBase<BaseFloat> &out_diff,
                                       CuMatrixBase<BaseFloat> *in_diff,
                                       CuMatrixBase<BaseFloat> *weights_diff,
                                       CuVectorBase<BaseFloat> *bias_diff) const {
            // Since PointwiseConv1d is a special case of Conv1d with kernel_size = 1,
            Conv1d::backward(in, out_diff, in_diff, weights_diff, bias_diff);
        }

        // PyTorch version: Input Dimensions: (batch, time, dim) - This is a 3D tensor where
        // batch is the batch size, time is the temporal dimension, and dim is the feature dimension.
        // Output Dimensions: (batch, time, dim), which are the same as the input dimensions.
        //
        // Kaldi version: (T, F) is the default input in Kaldi-matrix,
        // where T is the number of time frames and F is the number of features per frame.

        ConformerConvModule::ConformerConvModule(int in_channels, int kernel_size = 31, int expansion_factor = 2, BaseFloat dropout_rate = 0.1)
                : in_channels_(in_channels),
                  kernel_size_(kernel_size),
                  expansion_factor_(expansion_factor),
                  dropout_rate_(dropout_rate),
                  layer_norm_(in_channels),
                  pointwise_conv1d_1_(in_channels, in_channels * expansion_factor),
                  depthwise_conv1d_(in_channels * expansion_factor, in_channels * expansion_factor, kernel_size),
                  batch_norm_(in_channels * expansion_factor),
                  pointwise_conv1d_2_(in_channels * expansion_factor, in_channels),
                  dropout_(dropout_rate) {
            KALDI_ASSERT((kernel_size - 1) % 2 == 0);
            KALDI_ASSERT(expansion_factor == 2);
        }

        void ConformerConvModule::Forward(const ChunkInfo &in_info, const ChunkInfo &out_info,
                                          const CuMatrixBase<BaseFloat> &in, CuMatrixBase<BaseFloat> *out) const {
            // Ensure the output matrix is of the correct size
            out->Resize(out_info.NumRows(), out_info.NumCols());
            // Temporary matrices for intermediate results
            CuMatrix<BaseFloat> temp_matrix1, temp_matrix2;
            // Layer Normalization
            layer_norm_.Forward(in, &temp_matrix1);
            // Pointwise Convolution 1
            pointwise_conv1d_1_.Forward(temp_matrix1, &temp_matrix2);
            // Gated Linear Unit (GLU) - Custom implementation required
            GLU(temp_matrix2, &temp_matrix1);
            // Depthwise Convolution
            depthwise_conv1d_.Forward(temp_matrix1, &temp_matrix2);
            // Batch Normalization
            batch_norm_.Forward(temp_matrix2, &temp_matrix1);
            // Swish Activation
            Swish(temp_matrix1, &temp_matrix2);
            // Pointwise Convolution 2
            pointwise_conv1d_2_.Forward(temp_matrix2, &temp_matrix1);
            // Dropout
            dropout_.Forward(temp_matrix1, out, true);
        }

        void ConformerConvModule::Backward(const ChunkInfo &in_info, const ChunkInfo &out_info,
                                           const CuMatrixBase<BaseFloat> &in_value,
                                           const CuMatrixBase<BaseFloat> &out_value,
                                           const CuMatrixBase<BaseFloat> &out_deriv,
                                           Component *to_update, CuMatrixBase<BaseFloat> *in_deriv) const {
            // Initialize gradient matrices for each layer
            CuMatrix<BaseFloat> grad_next_step, grad_layer, grad_params;
            // Start with the output gradients
            grad_next_step = out_deriv;
            // Assuming each layer has a Backprop method implemented.
            // The layers should be processed in reverse order compared to the forward pass.
            // Backprop through Dropout
            grad_input = grad_output.transpose();
            dropout_layer.Backprop(out_value, grad_next_step, &grad_layer, to_update);
            grad_next_step = grad_layer;
            // Backprop through PointwiseConv1d
            pointwise_conv_layer.Backprop(out_value, grad_next_step, &grad_layer, to_update);
            grad_next_step = grad_layer;
            // Backprop through Swish Activation
            swish_layer.Backprop(out_value, grad_next_step, &grad_layer, to_update);
            grad_next_step = grad_layer;
            // Backprop through BatchNorm1d
            batch_norm_layer.Backprop(out_value, grad_next_step, &grad_layer, to_update);
            grad_next_step = grad_layer;
            // Backprop through DepthwiseConv1d
            depthwise_conv_layer.Backprop(out_value, grad_next_step, &grad_layer, to_update);
            grad_next_step = grad_layer;
            // Backprop through GLU
            glu_layer.Backprop(out_value, grad_next_step, &grad_layer, to_update);
            grad_next_step = grad_layer;
            // Backprop through LayerNorm
            layer_norm.Backprop(out_value, grad_next_step, &grad_layer, to_update);
            grad_next_step = grad_layer;

            in_deriv->CopyFromMat(grad_next_step);
        }

        // Input Shape: (time_dim, feature_dim) = (400, 40)
        Conv2dSubsampling::Conv2dSubsampling(int in_channels, int out_channels)
                : in_channels_(in_channels), out_channels_(out_channels) {
            conv1_.Init(in_channels, out_channels, 3 /*kernel_size*/, 2 /*stride*/, 0 /*padding*/);
            conv2_.Init(out_channels, out_channels, 3 /*kernel_size*/, 2 /*stride*/, 0 /*padding*/);
        }

        void Conv2dSubsampling::Forward(const Matrix<BaseFloat>& inputs, const std::vector<int>& input_lengths,
                                        Matrix<BaseFloat>& outputs, std::vector<int>& output_lengths) {
            int time_dim = inputs.NumRows();
            int feature_dim = inputs.NumCols();
            CuMatrix<BaseFloat> intermediate;
            CuMatrix<BaseFloat> output;
            conv1_.Forward(input, &intermediate);
            // (199, 19*64)
            relu_.Forward(intermediate, &intermediate);
            conv2_.Forward(intermediate, &intermediate);
            // (99, 8)
            relu_.Forward(intermediate, &intermediate);

            int output_time_dim = intermediate.NumRows();
            int output_feature_dim = intermediate.NumCols();
            outputs.Resize(output_time_dim, output_feature_dim);
            outputs.CopyFromMat(intermediate);
            // Output length calculation
            output_lengths = time_dim/4 - 1;
        }

        void Conv2dSubsampling::Backward(const Matrix<BaseFloat>& input,
                                         const Matrix<BaseFloat>& grad_output,
                                         Matrix<BaseFloat>& grad_input) {
            CuMatrix<BaseFloat> grad_intermediate(input.NumRows(), input.NumCols());
            CuMatrix<BaseFloat> grad_output_conv2(grad_output.NumRows(), grad_output.NumCols());
            CuMatrix<BaseFloat> grad_output_conv1(grad_output.NumRows(), grad_output.NumCols());
            CuMatrix<BaseFloat> reshaped_grad_output(a, output.NumCols());
            for (int t = 0; t < a; ++t) {
                for (int c = 0; c < out_channels_; ++c) {
                    for (int f = 0; f < output.NumCols(); ++f) {
                        int grad_output_index = c * output.NumCols() + f;
                        reshaped_grad_output(t, f) = grad_output(t, grad_output_index);
                    }
                }
            }
            ReLU::Backward(output, reshaped_grad_output, grad_output_conv2);
            conv2_.Backward(intermediate, grad_output_conv2, grad_intermediate);
            ReLU::Backward(intermediate, grad_intermediate, grad_output_conv1);
            conv1_.Backward(input, grad_output_conv1, grad_input);
        }

    }
}