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
#include "conformer-activation.h"

// Swish()
// GLU()

namespace kaldi {
    namespace conformer {

        // Swish Activation Function
        void Swish(const CuMatrixBase<BaseFloat>& input, CuMatrixBase<BaseFloat>* output) {
            // Swish: x * sigmoid(x)
            output->Sigmoid(input);
            output->MulElements(input);
        }

        // Sigmoid Function
        BaseFloat Sigmoid(BaseFloat x) {
            return 1.0 / (1.0 + Exp(-x));
        }

        // Gated Linear Unit (GLU) Activation Function
        void GLU(const CuMatrixBase<BaseFloat>& input, CuMatrixBase<BaseFloat>* output, int32 dim) {
            KALDI_ASSERT(input.NumCols() == 2 * dim);
            CuMatrix<BaseFloat> input_part1(input.NumRows(), dim);
            CuMatrix<BaseFloat> input_part2(input.NumRows(), dim);
            input_part1.CopyColsFromMat(input.ColRange(0, dim));
            input_part2.CopyColsFromMat(input.ColRange(dim, dim));
            input_part2.Sigmoid(input_part2);
            input_part1.MulElements(input_part2);
            output->CopyFromMat(input_part1);
        }

    }
}
