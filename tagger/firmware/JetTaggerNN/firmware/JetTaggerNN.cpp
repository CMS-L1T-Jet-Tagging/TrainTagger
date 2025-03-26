#include <iostream>

#include "JetTaggerNN.h"
#include "parameters.h"


void JetTaggerNN(
    input_t model_input[N_INPUT_1_1*N_INPUT_2_1],
    layer22_t layer22_out[N_LAYER_19], pT_output_result_t layer23_out[N_LAYER_23]
) {

    // hls-fpga-machine-learning insert IO
    #pragma HLS ARRAY_RESHAPE variable=model_input complete dim=0
    #pragma HLS ARRAY_PARTITION variable=layer22_out complete dim=0
    #pragma HLS ARRAY_PARTITION variable=layer23_out complete dim=0
    #pragma HLS INTERFACE ap_vld port=model_input,layer22_out,layer23_out 
    #pragma HLS DATAFLOW

    // hls-fpga-machine-learning insert load weights
#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        nnet::load_weights_from_txt<norm_input_scale_t, 20>(s2, "s2.txt");
        nnet::load_weights_from_txt<norm_input_bias_t, 20>(b2, "b2.txt");
        nnet::load_weights_from_txt<weight25_t, 200>(w25, "w25.txt");
        nnet::load_weights_from_txt<bias25_t, 10>(b25, "b25.txt");
        nnet::load_weights_from_txt<weight26_t, 100>(w26, "w26.txt");
        nnet::load_weights_from_txt<bias26_t, 10>(b26, "b26.txt");
        nnet::load_weights_from_txt<weight11_t, 320>(w11, "w11.txt");
        nnet::load_weights_from_txt<bias11_t, 32>(b11, "b11.txt");
        nnet::load_weights_from_txt<weight14_t, 512>(w14, "w14.txt");
        nnet::load_weights_from_txt<bias14_t, 16>(b14, "b14.txt");
        nnet::load_weights_from_txt<weight17_t, 100>(w17, "w17.txt");
        nnet::load_weights_from_txt<bias17_t, 10>(b17, "b17.txt");
        nnet::load_weights_from_txt<weight19_t, 128>(w19, "w19.txt");
        nnet::load_weights_from_txt<bias19_t, 8>(b19, "b19.txt");
        nnet::load_weights_from_txt<weight23_t, 10>(w23, "w23.txt");
        nnet::load_weights_from_txt<bias23_t, 1>(b23, "b23.txt");
        loaded_weights = true;    }
#endif
    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning insert layers

    norm_input_result_t layer2_out[N_INPUT_1_1*N_INPUT_2_1];
    #pragma HLS ARRAY_PARTITION variable=layer2_out complete dim=0
    nnet::normalize<input_t, norm_input_result_t, config2>(model_input, layer2_out, s2, b2); // norm_input

    Conv1D_1_result_t layer25_out[N_OUTPUTS_25*N_FILT_25];
    #pragma HLS ARRAY_PARTITION variable=layer25_out complete dim=0
    nnet::pointwise_conv_1d_cl<norm_input_result_t, Conv1D_1_result_t, config25>(layer2_out, layer25_out, w25, b25); // Conv1D_1

    layer5_t layer5_out[N_OUTPUTS_3*N_FILT_3];
    #pragma HLS ARRAY_PARTITION variable=layer5_out complete dim=0
    nnet::relu<Conv1D_1_result_t, layer5_t, relu_config5>(layer25_out, layer5_out); // relu_1

    Conv1D_2_result_t layer26_out[N_OUTPUTS_26*N_FILT_26];
    #pragma HLS ARRAY_PARTITION variable=layer26_out complete dim=0
    nnet::pointwise_conv_1d_cl<layer5_t, Conv1D_2_result_t, config26>(layer5_out, layer26_out, w26, b26); // Conv1D_2

    layer8_t layer8_out[N_OUTPUTS_6*N_FILT_6];
    #pragma HLS ARRAY_PARTITION variable=layer8_out complete dim=0
    nnet::relu<Conv1D_2_result_t, layer8_t, relu_config8>(layer26_out, layer8_out); // relu_2

    layer9_t layer9_out[N_OUTPUTS_6*N_FILT_6];
    #pragma HLS ARRAY_PARTITION variable=layer9_out complete dim=0
    nnet::linear<layer8_t, layer9_t, linear_config9>(layer8_out, layer9_out); // act_pool

    layer10_t layer10_out[N_FILT_10];
    #pragma HLS ARRAY_PARTITION variable=layer10_out complete dim=0
    nnet::global_pooling1d_cl<layer9_t, layer10_t, config10>(layer9_out, layer10_out); // avgpool

    Dense_1_jetID_result_t layer11_out[N_LAYER_11];
    #pragma HLS ARRAY_PARTITION variable=layer11_out complete dim=0
    nnet::dense<layer10_t, Dense_1_jetID_result_t, config11>(layer10_out, layer11_out, w11, b11); // Dense_1_jetID

    layer13_t layer13_out[N_LAYER_11];
    #pragma HLS ARRAY_PARTITION variable=layer13_out complete dim=0
    nnet::relu<Dense_1_jetID_result_t, layer13_t, relu_config13>(layer11_out, layer13_out); // relu_1_jetID

    Dense_2_jetID_result_t layer14_out[N_LAYER_14];
    #pragma HLS ARRAY_PARTITION variable=layer14_out complete dim=0
    nnet::dense<layer13_t, Dense_2_jetID_result_t, config14>(layer13_out, layer14_out, w14, b14); // Dense_2_jetID

    layer16_t layer16_out[N_LAYER_14];
    #pragma HLS ARRAY_PARTITION variable=layer16_out complete dim=0
    nnet::relu<Dense_2_jetID_result_t, layer16_t, relu_config16>(layer14_out, layer16_out); // relu_2_jetID

    Dense_1_pT_result_t layer17_out[N_LAYER_17];
    #pragma HLS ARRAY_PARTITION variable=layer17_out complete dim=0
    nnet::dense<layer10_t, Dense_1_pT_result_t, config17>(layer10_out, layer17_out, w17, b17); // Dense_1_pT

    Dense_3_jetID_result_t layer19_out[N_LAYER_19];
    #pragma HLS ARRAY_PARTITION variable=layer19_out complete dim=0
    nnet::dense<layer16_t, Dense_3_jetID_result_t, config19>(layer16_out, layer19_out, w19, b19); // Dense_3_jetID

    layer21_t layer21_out[N_LAYER_17];
    #pragma HLS ARRAY_PARTITION variable=layer21_out complete dim=0
    nnet::relu<Dense_1_pT_result_t, layer21_t, relu_config21>(layer17_out, layer21_out); // relu_1_pt

    nnet::softmax<Dense_3_jetID_result_t, layer22_t, softmax_config22>(layer19_out, layer22_out); // jet_id_output

    nnet::dense<layer21_t, pT_output_result_t, config23>(layer21_out, layer23_out, w23, b23); // pT_output

}

