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
#include "conformer-attention.h"

// RelativeMultiHeadAttentionComponent
// MultiHeadedSelfAttentionModule

namespace kaldi {
    namespace conformer {
        RelativeMultiHeadAttentionComponent::RelativeMultiHeadAttentionComponent(int input_dim, int num_heads, BaseFloat dropout_rate)
                : input_dim_(input_dim),
                    num_heads_(num_heads),
                    dropout_rate_(dropout_rate),
                    d_head_(input_dim / num_heads),
                    sqrt_d_model_(std::sqrt(input_dim)),
                    query_linear_(input_dim, input_dim),
                    key_linear_(input_dim, input_dim),
                    value_linear_(input_dim, input_dim),
                    pos_linear_(input_dim, input_dim),
                    output_linear_(input_dim, input_dim),
                    dropout_(dropout_rate) {
            KALDI_ASSERT(input_dim % num_heads == 0);
            Init(input_dim, num_heads, dropout_rate);
        }

        void RelativeMultiHeadAttentionComponent::InitializeWeights(CuMatrix<BaseFloat>* weights, int input_dim, int output_dim) {
            weights->Resize(output_dim, input_dim);
            XavierInitialization(weights, input_dim, output_dim);
        }

        void RelativeMultiHeadAttentionComponent::XavierInitialization(CuMatrix<BaseFloat>* weights, int input_dim, int output_dim) {
            BaseFloat limit = std::sqrt(6.0 / (input_dim + output_dim));
            std::default_random_engine generator;
            std::uniform_real_distribution<BaseFloat> distribution(-1, 1);
            for (int r = 0; r < weights->NumRows(); ++r) {
                for (int c = 0; c < weights->NumCols(); ++c) {
                    (*weights)(r, c) = distribution(generator);
                }
            }
        }

        void RelativeMultiHeadAttentionComponent::Init(int input_dim, int num_heads, BaseFloat dropout_rate) {
            input_dim_ = input_dim;
            num_heads_ = num_heads;
            dropout_rate_ = dropout_rate;
            d_head_ = input_dim / num_heads;
            sqrt_d_model_ = std::sqrt(input_dim);

            InitializeWeights(&query_linear.weights_, input_dim, input_dim);
            InitializeWeights(&key_linear.weights_, input_dim, input_dim);
            InitializeWeights(&value_linear.weights_, input_dim, input_dim);
            InitializeWeights(&pos_linear.weights_, input_dim, input_dim);
            InitializeWeights(&output_linear.weights_, input_dim, input_dim);

            u_bias_.Resize(num_heads, d_head_);
            v_bias_.Resize(num_heads, d_head_);
            query_bias_.Resize(input_dim);
            key_bias_.Resize(input_dim);
            value_bias_.Resize(input_dim);
            pos_bias_.Resize(input_dim);
            out_bias_.Resize(input_dim);

            std::default_random_engine generator;
            std::uniform_real_distribution<BaseFloat> distribution(-0.1, 0.1);

            for (int i = 0; i < u_bias_.Dim(); ++i) {
                u_bias_(i) = distribution(generator);
            }
            for (int i = 0; i < v_bias_.Dim(); ++i) {
                v_bias_(i) = distribution(generator);
            }

            query_bias_.SetZero();
            key_bias_.SetZero();
            value_bias_.SetZero();
            pos_bias_.SetZero();
            out_bias_.SetZero();
        }

        void RelativeMultiHeadAttentionComponent::Forward(
                const CuMatrixBase<BaseFloat>& query,
                const CuMatrixBase<BaseFloat>& key,
                const CuMatrixBase<BaseFloat>& value,
                const CuMatrixBase<BaseFloat>& pos_embedding,
                const CuMatrixBase<BaseFloat>* mask,
                CuMatrixBase<BaseFloat>* output) {

            CuMatrix<BaseFloat> projected_query;
            CuMatrix<BaseFloat> projected_key;
            CuMatrix<BaseFloat> projected_value;
            CuMatrix<BaseFloat> &projected_pos;

            query_linear.Forward(query, &projected_query);
            key_linear.Forward(key, &projected_key);
            value_linear.Forward(value, &projected_value);
            pos_linear.Forward(pos_embedding, &projected_pos)

            CuMatrix<BaseFloat> query_with_u_bias = AddBias(projected_query, u_bias_, &query_with_u_bias);
            CuMatrix<BaseFloat> query_with_v_bias = AddBias(projected_query, v_bias_, &query_with_v_bias);

            CuMatrix<BaseFloat> content_score = MatMul(query_with_u_bias, Transpose(projected_key));
            CuMatrix<BaseFloat> pos_score = MatMul(query_with_v_bias, Transpose(projected_pos_embedding));

            CuMatrix<BaseFloat> shifted_pos_score;
            RelativeShift(pos_score, &shifted_pos_score);
            CuMatrix<BaseFloat> combined_score = AddMatrices(content_score, shifted_pos_score, &combined_score);
            combined_score.Scale(1.0 / sqrt_d_model_);
            if (mask != nullptr) {
                ApplyMask(&combined_score, *mask, mask_value);
            }
            CuMatrix<BaseFloat> attn(combined_score.NumRows(), combined_score.NumCols());
            attn.SoftMaxPerRow(combined_score);
            CuMatrix<BaseFloat> attn_dropped;
            dropout.forward(attn, &attn_dropped, true);
            CuMatrix<BaseFloat> context = MatMul(attn, projected_value);
            CuMatrix<BaseFloat> linear_output;
            output_linear.Forward(context, &linear_output);
            output->CopyFromMat(linear_output);
        }

        void RelativeMultiHeadAttentionComponent::RelativeShift(
                const CuMatrixBase<BaseFloat>& pos_score,
                CuMatrixBase<BaseFloat>* shifted_score) {
            int num_heads = pos_score.NumCols() / d_head_;
            int seq_length = pos_score.NumRows();
            CuMatrix<BaseFloat> padded_pos_score(num_heads, seq_length + 1, kSetZero);
            for (int32 h = 0; h < num_heads; ++h) {
                padded_pos_score.Row(h).ColRange(1, seq_length).CopyFromMat(
                        pos_score.Row(h).ColRange(0, seq_length)
                );
            }
            shifted_score->Resize(num_heads, seq_length, kSetZero);
            for (int32 h = 0; h < num_heads; ++h) {
                shifted_score->Row(h).CopyFromMat(
                        padded_pos_score.Row(h).ColRange(1, seq_length)
                );
            }
        }


        MultiHeadedSelfAttentionModule::MultiHeadedSelfAttentionModule(int d_model, int num_heads, BaseFloat dropout_rate = 0.1)
                : positional_encoding_(d_model),
                  layer_norm_(d_model),
                  attention_(d_model, num_heads, dropout_rate),
                  dropout_(dropout_rate) {
        }

        void MultiHeadedSelfAttentionModule::Forward(const Matrix<BaseFloat>& input, Matrix<BaseFloat>* output, const std::vector<bool>* mask) {
            int seq_length = input.NumRows();
            int feature_dim = input.NumCols();
            Matrix<BaseFloat> pos_embedding;
            positional_encoding_.Forward(seq_length, &pos_embedding);
            Matrix<BaseFloat> norm_input;
            layer_norm_.Forward(input, &norm_input);
            Matrix<BaseFloat> attention_output;
            attention_.Forward(norm_input, norm_input, norm_input, pos_embedding, mask, &attention_output);
            dropout_.Forward(attention_output, output, true);
        }

    }
}
