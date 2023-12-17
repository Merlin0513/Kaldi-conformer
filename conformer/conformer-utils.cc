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

// ResidualConnectionModule
// Transpose12
// LayerNorm
// BatchNorm1d
// Dropout
// ReLU
// Linear

namespace kaldi {
    namespace conformer {

        /*ResidualConnectionModule::ResidualConnectionModule(float module_factor, float input_factor, Module *module)
                : module_factor_(module_factor), input_factor_(input_factor), module_(module) {
        }

        ResidualConnectionModule(Module *module, float module_factor = 1.0, float input_factor = 1.0)
        : module_(module), module_factor_(module_factor), input_factor_(input_factor) {}

        void ResidualConnectionModule::Forward(const MatrixBase<BaseFloat> &input,
                                               MatrixBase<BaseFloat> *output) {
            KALDI_ASSERT(module_ != nullptr);
            KALDI_ASSERT(input.NumCols() == module_->InputDim());
            Matrix<BaseFloat> module_output(input.NumRows(), module_->OutputDim());
            module_->Forward(input, &module_output);
            output->Resize(input.NumRows(), input.NumCols());
            output->CopyFromMat(input, kNoTrans);
            output->Scale(input_factor_);
            output->AddMat(module_factor_, module_output, kNoTrans);
        }

        void ResidualConnectionModule::Backward(const Matrix<BaseFloat>& input,
                                                const Matrix<BaseFloat>& grad_output,
                                                Matrix<BaseFloat>* grad_input) {
            KALDI_ASSERT(module_ != nullptr);
            KALDI_ASSERT(input.NumCols() == module_->InputDim());
            KALDI_ASSERT(grad_output.NumRows() == input.NumRows());
            KALDI_ASSERT(grad_output.NumCols() == input.NumCols());
            if (grad_input != nullptr) {
                grad_input->Resize(input.NumRows(), input.NumCols());
                grad_input->SetZero();
            }
            Matrix<BaseFloat> grad_module_output;
            module_->Backward(input, grad_output, &grad_module_output);
            if (grad_input != nullptr) {
                grad_input->AddMat(input_factor_, grad_output);
                grad_input->AddMat(module_factor_, grad_module_output);
            }
        }
*/
        void Transpose12(CuArray<CuMatrixBase<BaseFloat>>& input_tensor, CuArray<CuMatrixBase<BaseFloat>>& output_tensor) {
            int num_matrices = input_tensor.Dim();
            int rows = input_tensor[0].NumRows();
            int cols = input_tensor[0].NumCols();
            output_tensor.Resize(num_matrices, CuMatrix<BaseFloat>(cols, rows));

            for (int i = 0; i < num_matrices; ++i) {
                output_tensor[i].Transpose(input_tensor[i]);
            }
        }

        LayerNorm::LayerNorm(int dim, BaseFloat epsilon = 1e-5)
            : dim_(dim), epsilon_(epsilon), gain_(dim), bias_(dim) {
                gain_.Set(1.0);
                bias_.Set(0.0);
            }

        void LayerNorm::Forward(const kaldi::MatrixBase<BaseFloat> &input,
                                kaldi::MatrixBase<BaseFloat> *output) {
            KALDI_ASSERT(input.NumCols() == dim_);
            CuVector<BaseFloat> mean(dim_);
            mean.AddRowSumMat(1.0 / input.NumRows(), input, 0.0);
            CuVector<BaseFloat> var(dim_);
            CuMatrix<BaseFloat> input_squared_diff = input;
            input_squared_diff.AddVecToRows(-1.0, mean, 1.0);  // input - mean
            input_squared_diff.MulElements(input_squared_diff); // (input - mean)^2
            var.AddRowSumMat(1.0 / input.NumRows(), input_squared_diff, 0.0);
            var.Add(epsilon_);
            var.ApplyPow(-0.5); // var = 1 / sqrt(var + epsilon)
            output->MulColsVec(var);
            output->MulColsVec(gain_);
            output->AddVecToRows(1.0, bias_);
        }

        void LayerNorm::Backward(const CuMatrixBase<BaseFloat>& in,
                                 const CuMatrixBase<BaseFloat>& out_diff,
                                 CuMatrixBase<BaseFloat>* in_diff,
                                 CuVectorBase<BaseFloat>* gain_diff,
                                 CuVectorBase<BaseFloat>* bias_diff) const {
            int num_rows = in.NumRows();
            int num_cols = in.NumCols();
            CuVector<BaseFloat> mean(num_rows);
            CuVector<BaseFloat> var(num_rows);
            mean.AddColSumMat(1.0 / num_cols, in, 0.0);
            var.CopyFromMat(in);
            var.AddVecToRows(-1.0, mean);
            var.ApplyPow(2.0);
            var.AddColSumMat(1.0 / num_cols, var, 0.0);

            CuMatrix<BaseFloat> normalized(num_rows, num_cols);
            normalized.CopyFromMat(in);
            normalized.AddVecToRows(-1.0, mean);
            var.Add(epsilon_);
            var.ApplyPow(-0.5); // var = 1 / sqrt(var + epsilon)
            normalized.MulColsVec(var);

            var.ApplyPow(-2.0); // var = 1 / (var + epsilon)^2
            CuMatrix<BaseFloat> temp(in);
            temp.AddVecToRows(-1.0, mean);
            temp.MulColsVec(var);
            temp.MulElements(out_diff);

            CuVector<BaseFloat> sum(num_rows);
            sum.AddColSumMat(1.0, temp, 0.0);
            in_diff->CopyFromMat(out_diff);
            in_diff->AddVecToRows(-1.0 / num_cols, sum);
            in_diff->MulColsVec(var);

            if (gain_diff != nullptr && bias_diff != nullptr) {
                gain_diff->AddRowSumMat(1.0, CuMatrix<BaseFloat>(normalized.MulElements(out_diff)), 0.0);
                bias_diff->AddColSumMat(1.0, out_diff, 0.0);
            }
        }


        BatchNorm1d::BatchNorm1d(int in_channels, BaseFloat epsilon)
                : in_channels_(in_channels), epsilon_(epsilon),
                  gamma_(in_channels), beta_(in_channels) {
            gamma_.Set(1.0); // Initialize gamma to 1
            beta_.Set(0.0);  // Initialize beta to 0
        }

        void BatchNorm1d::Forward(const CuMatrixBase<BaseFloat>& input, CuMatrixBase<BaseFloat>* output) {
            // Check dimensions
            KALDI_ASSERT(input.NumCols() == in_channels_);

            // Compute mean and variance
            CuVector<BaseFloat> mean(in_channels_);
            CuVector<BaseFloat> var(in_channels_);
            mean.AddColSumMat(1.0 / input.NumRows(), input, 0.0);

            CuMatrix<BaseFloat> temp(input);
            temp.AddVecToRows(-1.0, mean);
            var.AddDiagMat2(1.0 / input.NumRows(), temp, kNoTrans, 0.0);

            // Normalize
            var.ApplyFloor(epsilon_);
            var.ApplyPow(-0.5); // var = 1 / sqrt(var + epsilon)
            output->CopyFromMat(input);
            output->AddVecToRows(-1.0, mean);
            output->MulColsVec(var);

            // Apply gamma and beta
            output->MulColsVec(gamma_);
            output->AddVecToRows(1.0, beta_);
        }

        Dropout::Dropout(BaseFloat dropout_rate) : dropout_rate_(dropout_rate) {
            KALDI_ASSERT(dropout_rate_ >= 0.0 && dropout_rate_ <= 1.0);
        }

        void Dropout::Forward(const CuMatrixBase<BaseFloat>& input,
                              CuMatrixBase<BaseFloat>* output, bool is_training = true) {
            if (!is_training) {
                output->CopyFromMat(input);
                return;
            }
            CuMatrix<BaseFloat> mask(input.NumRows(), input.NumCols());
            // Randomly initialize the mask with values 0 or 1 based on dropout_rate_
            mask.SetRandUniform();
            mask.ApplyFloor(1.0 - dropout_rate_);
            output->Resize(input.NumRows(), input.NumCols());
            output->CopyFromMat(input);
            output->MulElements(mask);
        }

        void Dropout::Backward(const CuMatrixBase<BaseFloat>& in_diff,
                               const CuMatrixBase<BaseFloat>& mask,  // Use stored mask
                               CuMatrixBase<BaseFloat>* out_diff) {
            out_diff->CopyFromMat(in_diff);
            out_diff->MulElements(mask);  // Filter gradients with mask
        }

        void ReLU::Forward(const CuMatrixBase<BaseFloat>& input, CuMatrixBase<BaseFloat>* output) const {
            output->Resize(input.NumRows(), input.NumCols());
            output->CopyFromMat(input);
            output->ApplyFloor(0.0);
        }

        void ReLU::Backward(const CuMatrixBase<BaseFloat>& out_diff,
                            const CuMatrixBase<BaseFloat>& input,
                            CuMatrixBase<BaseFloat>* in_diff) const {
            in_diff->Resize(input.NumRows(), input.NumCols());

            for (int r = 0; r < input.NumRows(); ++r) {
                for (int c = 0; c < input.NumCols(); ++c) {
                    (*in_diff)(r, c) = (input(r, c) > 0.0) ? out_diff(r, c) : 0.0;
                }
            }
        }

        Linear::Linear(int input_dim, int output_dim, bool bias = true) {
            weights_.Resize(output_dim, input_dim);
            if (bias) {
                bias_.Resize(output_dim);
                bias_.SetZero();
            }
        }

        void Linear::Forward(const kaldi::MatrixBase<BaseFloat> &input, kaldi::Matrix<BaseFloat> *output) {
            KALDI_ASSERT(input.NumCols() == weights_.NumCols());
            // Perform the linear transformation: output = input * weights^T + bias
            output->Resize(input.NumRows(), weights_.NumRows());
            output->AddMatMat(1.0, input, kaldi::kNoTrans, weights_, kaldi::kTrans, 0.0);
            if (bias_.NumRows() > 0) {
                output->AddVecToRows(1.0, bias_);
            }
        }

        int CalculateOutputLength(int signal_length, int kernel_size, int stride, int padding) {
            int output_length = (signal_length + 2 * padding - kernel_size) / stride + 1;
            return output_length;
        }

        void ApplyMask(CuMatrixBase<BaseFloat>* matrix, const CuMatrixBase<BaseFloat>& mask, BaseFloat mask_value = -1e9) {
            KALDI_ASSERT(matrix != nullptr);
            KALDI_ASSERT(matrix->NumRows() == mask.NumRows() && matrix->NumCols() == mask.NumCols());

            for (int32 r = 0; r < matrix->NumRows(); ++r) {
                for (int32 c = 0; c < matrix->NumCols(); ++c) {
                    if (mask(r, c) == 0) {
                        (*matrix)(r, c) = mask_value;
                    }
                }
            }
        }

        void AddBias(const CuMatrixBase<BaseFloat>& input_matrix,
                     const CuVector<BaseFloat>& bias,
                     CuMatrix<BaseFloat>* output_matrix) {
            KALDI_ASSERT(input_matrix.NumCols() == bias.Dim());
            KALDI_ASSERT(output_matrix != nullptr);

            // Resize the output matrix to match the input matrix dimensions
            output_matrix->Resize(input_matrix.NumRows(), input_matrix.NumCols());

            // Copy the input matrix to the output matrix
            output_matrix->CopyFromMat(input_matrix);

            // Add the bias to each row of the output matrix
            output_matrix->AddVecToRows(1.0, bias);
        }

        void AddMatrices(const CuMatrixBase<BaseFloat>& mat1,
                         const CuMatrixBase<BaseFloat>& mat2,
                         CuMatrix<BaseFloat>* result) {
            KALDI_ASSERT(mat1.NumRows() == mat2.NumRows());
            KALDI_ASSERT(mat1.NumCols() == mat2.NumCols());
            KALDI_ASSERT(result != nullptr);
            // Resize result matrix to match the dimensions of the input matrices
            result->Resize(mat1.NumRows(), mat1.NumCols());
            // Add mat1 to result
            result->AddMat(1.0, mat1);
            // Add mat2 to result
            result->AddMat(1.0, mat2);
        }




    }  // namespace nnet3
}  // namespace kaldi