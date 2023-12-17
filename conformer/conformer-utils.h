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

        void Transpose12(CuArray<CuMatrixBase<BaseFloat>>& input_tensor, CuArray<CuMatrixBase<BaseFloat>>& output_tensor);

        class LayerNorm {
        public:
            LayerNorm(int dim, BaseFloat epsilon = 1e-5);
            void Forward(const kaldi::MatrixBase<BaseFloat> &input,
                         kaldi::MatrixBase<BaseFloat> *output);
            void backward(const CuMatrixBase<BaseFloat>& in,
                          const CuMatrixBase<BaseFloat>& out_diff,
                          CuMatrixBase<BaseFloat>* in_diff,
                          CuVectorBase<BaseFloat>* gain_diff,
                          CuVectorBase<BaseFloat>* bias_diff) const;
        private:
            int dim_;
            BaseFloat epsilon_;
            CuVector<BaseFloat> gain_;
            CuVector<BaseFloat> bias_;
        };

        class BatchNorm1d {
        public:
            BatchNorm1d(int in_channels, BaseFloat epsilon);
            void Forward(const CuMatrixBase<BaseFloat>& input, CuMatrixBase<BaseFloat>* output);
        private:
            int in_channels_;
            BaseFloat epsilon_;
            CuVector<BaseFloat> gamma_;
            CuVector<BaseFloat> beta_;
        };

        class Dropout {
        public:
            explicit Dropout(BaseFloat dropout_rate);
            void Forward(const CuMatrixBase<BaseFloat>& input, CuMatrixBase<BaseFloat>* output, bool is_training = true);
            void Backward(const CuMatrixBase<BaseFloat>& in_diff, const CuMatrixBase<BaseFloat>& mask, CuMatrixBase<BaseFloat>* out_diff);
        private:
            BaseFloat dropout_rate_;
        };

        class ReLU {
        public:
            void Forward(const CuMatrixBase<BaseFloat>& input, CuMatrixBase<BaseFloat>* output) const;
            void Backward(const CuMatrixBase<BaseFloat>& out_diff,
                          const CuMatrixBase<BaseFloat>& input,
                          CuMatrixBase<BaseFloat>* in_diff) const;
        };

        class Linear {
        public:
            Linear(int input_dim, int output_dim, bool bias = true);
            void Forward(const kaldi::MatrixBase<BaseFloat> &input, kaldi::Matrix<BaseFloat> *output);
        private:
            kaldi::Matrix<BaseFloat> weights_;
            kaldi::Vector<BaseFloat> bias_;
        };

        int CalculateOutputLength(int signal_length, int kernel_size, int stride, int padding);

        void ApplyMask(CuMatrixBase<BaseFloat>* matrix, const CuMatrixBase<BaseFloat>& mask, BaseFloat mask_value = -1e9);

        void AddBias(const CuMatrixBase<BaseFloat>& input_matrix,
                     const CuVector<BaseFloat>& bias,
                     CuMatrix<BaseFloat>* output_matrix);

        void AddMatrices(const CuMatrixBase<BaseFloat>& mat1,
                         const CuMatrixBase<BaseFloat>& mat2,
                         CuMatrix<BaseFloat>* result);
    }
}