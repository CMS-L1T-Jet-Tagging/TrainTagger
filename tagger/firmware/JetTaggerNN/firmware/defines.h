#ifndef DEFINES_H_
#define DEFINES_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "nnet_utils/nnet_types.h"
#include <cstddef>
#include <cstdio>

// hls-fpga-machine-learning insert numbers
#define N_INPUT_1_1 16
#define N_INPUT_2_1 20
#define N_INPUT_1_1 16
#define N_INPUT_2_1 20
#define N_OUTPUTS_25 16
#define N_FILT_25 10
#define N_OUTPUTS_3 16
#define N_FILT_3 10
#define N_OUTPUTS_26 16
#define N_FILT_26 10
#define N_OUTPUTS_6 16
#define N_FILT_6 10
#define N_OUTPUTS_6 16
#define N_FILT_6 10
#define N_FILT_10 10
#define N_LAYER_11 32
#define N_LAYER_11 32
#define N_LAYER_14 16
#define N_LAYER_14 16
#define N_LAYER_17 10
#define N_LAYER_19 8
#define N_LAYER_17 10
#define N_LAYER_19 8
#define N_LAYER_23 1


// hls-fpga-machine-learning insert layer-precision
typedef ap_fixed<16,6> input_t;
typedef ap_fixed<33,13> norm_input_result_t;
typedef ap_fixed<16,6> norm_input_scale_t;
typedef ap_fixed<16,6> norm_input_bias_t;
typedef ap_fixed<16,6> model_default_t;
typedef ap_fixed<48,22> Conv1D_1_result_t;
typedef ap_fixed<9,3> weight25_t;
typedef ap_fixed<9,3> bias25_t;
typedef ap_ufixed<9,0,AP_RND_CONV,AP_SAT,0> layer5_t;
typedef ap_fixed<18,8> relu_1_table_t;
typedef ap_fixed<23,8> Conv1D_2_result_t;
typedef ap_fixed<9,3> weight26_t;
typedef ap_fixed<9,3> bias26_t;
typedef ap_ufixed<9,0,AP_RND_CONV,AP_SAT,0> layer8_t;
typedef ap_fixed<18,8> relu_2_table_t;
typedef ap_fixed<18,9,AP_RND_CONV,AP_SAT,0> layer9_t;
typedef ap_fixed<18,8> act_pool_table_t;
typedef ap_fixed<16,6> layer10_t;
typedef ap_fixed<30,14> Dense_1_jetID_result_t;
typedef ap_fixed<9,3> weight11_t;
typedef ap_fixed<9,3> bias11_t;
typedef ap_uint<1> layer11_index;
typedef ap_ufixed<9,0,AP_RND_CONV,AP_SAT,0> layer13_t;
typedef ap_fixed<18,8> relu_1_jetID_table_t;
typedef ap_fixed<24,9> Dense_2_jetID_result_t;
typedef ap_fixed<9,3> weight14_t;
typedef ap_fixed<9,3> bias14_t;
typedef ap_uint<1> layer14_index;
typedef ap_ufixed<9,0,AP_RND_CONV,AP_SAT,0> layer16_t;
typedef ap_fixed<18,8> relu_2_jetID_table_t;
typedef ap_fixed<30,14> Dense_1_pT_result_t;
typedef ap_fixed<9,3> weight17_t;
typedef ap_fixed<9,3> bias17_t;
typedef ap_uint<1> layer17_index;
typedef ap_fixed<23,8> Dense_3_jetID_result_t;
typedef ap_fixed<9,3> weight19_t;
typedef ap_fixed<9,3> bias19_t;
typedef ap_uint<1> layer19_index;
typedef ap_ufixed<9,0,AP_RND_CONV,AP_SAT,0> layer21_t;
typedef ap_fixed<18,8> relu_1_pt_table_t;
typedef ap_fixed<16,6> layer22_t;
typedef ap_fixed<18,8> jet_id_output_table_t;
typedef ap_fixed<18,8,AP_RND,AP_SAT,0> jet_id_output_exp_table_t;
typedef ap_fixed<18,8,AP_RND,AP_SAT,0> jet_id_output_inv_table_t;
typedef ap_fixed<30,12> pT_output_result_t;
typedef ap_fixed<16,7> weight23_t;
typedef ap_fixed<16,7> bias23_t;
typedef ap_uint<1> layer23_index;


#endif
