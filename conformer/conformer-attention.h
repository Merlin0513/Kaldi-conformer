#ifndef KALDI_CONFORMER_ATTENTION_H_
#define KALDI_CONFORMER_ATTENTION_H_

#include "conformer-utils.h"
#include "conformer-embedding.h"
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

        class RelativeMultiHeadAttentionComponent {
        public:
            RelativeMultiHeadAttentionComponent(int input_dim, int num_heads, BaseFloat dropout_rate);
            void Forward(const CuMatrixBase<BaseFloat>& query,
                         const CuMatrixBase<BaseFloat>& key,
                         const CuMatrixBase<BaseFloat>& value,
                         const CuMatrixBase<BaseFloat>& pos_embedding,
                         const CuMatrixBase<BaseFloat>* mask,
                         CuMatrixBase<BaseFloat>* output);

        private:
            void InitializeWeights(CuMatrix<BaseFloat>* weights, int input_dim, int output_dim);
            void XavierInitialization(CuMatrix<BaseFloat>* weights, int input_dim, int output_dim);
            void Init(int input_dim, int num_heads, BaseFloat dropout_rate);
            void RelativeShift(const CuMatrixBase<BaseFloat>& pos_score, CuMatrixBase<BaseFloat>* shifted_score);

            int input_dim_;
            int num_heads_;
            BaseFloat dropout_rate_;
            int d_head_;
            BaseFloat sqrt_d_model_;
            Linear query_linear_;
            Linear key_linear_;
            Linear value_linear_;
            Linear pos_linear_;
            Linear output_linear_;

            // Bias terms
            CuVector<BaseFloat> u_bias_;
            CuVector<BaseFloat> v_bias_;
            CuVector<BaseFloat> query_bias_;
            CuVector<BaseFloat> key_bias_;
            CuVector<BaseFloat> value_bias_;
            CuVector<BaseFloat> pos_bias_;
            CuVector<BaseFloat> out_bias_;

            Dropout dropout_;
        };

        class MultiHeadedSelfAttentionModule {
        public:
            MultiHeadedSelfAttentionModule(int d_model, int num_heads, BaseFloat dropout_rate = 0.1);
            void Forward(const Matrix<BaseFloat>& input,
                         Matrix<BaseFloat>* output,
                         const std::vector<bool>* mask);

        private:
            int d_model_;
            int num_heads_;
            BaseFloat dropout_rate_;
            PositionalEncoding positional_encoding_;
            LayerNorm layer_norm_;
            RelativeMultiHeadAttentionComponent attention_;
            Dropout dropout_;
        };
    }
}

#endif