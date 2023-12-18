#include "main.h"
#include <vector>
#include "conformer-models.h"

// ./main_program

int main() {
    try {
        int num_classes = 1000;
        int input_dim = 80;      // 80 for MFCC or filterbank features
        int encoder_dim = 512;
        int num_encoder_layers = 17;
        int num_attention_heads = 8;
        int feed_forward_expansion_factor = 4;
        int conv_expansion_factor = 2;
        BaseFloat input_dropout_p = 0.1;
        BaseFloat feed_forward_dropout_p = 0.1;
        BaseFloat attention_dropout_p = 0.1;
        BaseFloat conv_dropout_p = 0.1;
        int conv_kernel_size = 31;
        bool half_step_residual = true;

        Conformer model(num_classes, input_dim, encoder_dim, num_encoder_layers,
                        num_attention_heads, feed_forward_expansion_factor,
                        conv_expansion_factor, input_dropout_p,
                        feed_forward_dropout_p, attention_dropout_p,
                        conv_dropout_p, conv_kernel_size, half_step_residual);

        Matrix<BaseFloat> inputs;
        std::vector<int> input_lengths;

        Matrix<BaseFloat> outputs;
        std::vector<int> output_lengths;

        model.Forward(inputs, input_lengths, &outputs, &output_lengths);

    } catch(const std::exception& e) {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}