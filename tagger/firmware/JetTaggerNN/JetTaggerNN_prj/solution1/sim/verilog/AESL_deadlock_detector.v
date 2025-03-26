`timescale 1 ns / 1 ps

module AESL_deadlock_detector (
    input dl_reset,
    input all_finish,
    input dl_clock);

    wire [0:0] proc_0_data_FIFO_blk;
    wire [0:0] proc_0_data_PIPO_blk;
    wire [0:0] proc_0_start_FIFO_blk;
    wire [0:0] proc_0_TLF_FIFO_blk;
    wire [0:0] proc_0_input_sync_blk;
    wire [0:0] proc_0_output_sync_blk;
    wire [0:0] proc_dep_vld_vec_0;
    reg [0:0] proc_dep_vld_vec_0_reg;
    wire [0:0] in_chan_dep_vld_vec_0;
    wire [1:0] in_chan_dep_data_vec_0;
    wire [0:0] token_in_vec_0;
    wire [0:0] out_chan_dep_vld_vec_0;
    wire [1:0] out_chan_dep_data_0;
    wire [0:0] token_out_vec_0;
    wire dl_detect_out_0;
    wire dep_chan_vld_1_0;
    wire [1:0] dep_chan_data_1_0;
    wire token_1_0;
    wire [0:0] proc_1_data_FIFO_blk;
    wire [0:0] proc_1_data_PIPO_blk;
    wire [0:0] proc_1_start_FIFO_blk;
    wire [0:0] proc_1_TLF_FIFO_blk;
    wire [0:0] proc_1_input_sync_blk;
    wire [0:0] proc_1_output_sync_blk;
    wire [0:0] proc_dep_vld_vec_1;
    reg [0:0] proc_dep_vld_vec_1_reg;
    wire [0:0] in_chan_dep_vld_vec_1;
    wire [1:0] in_chan_dep_data_vec_1;
    wire [0:0] token_in_vec_1;
    wire [0:0] out_chan_dep_vld_vec_1;
    wire [1:0] out_chan_dep_data_1;
    wire [0:0] token_out_vec_1;
    wire dl_detect_out_1;
    wire dep_chan_vld_0_1;
    wire [1:0] dep_chan_data_0_1;
    wire token_0_1;
    wire [1:0] dl_in_vec;
    wire dl_detect_out;
    wire token_clear;
    wire [1:0] origin;

    reg ap_done_reg_0;// for module AESL_inst_JetTaggerNN.pointwise_conv_1d_cl_ap_fixed_ap_fixed_48_22_5_3_0_config25_U0
    always @ (negedge dl_reset or posedge dl_clock) begin
        if (~dl_reset) begin
            ap_done_reg_0 <= 'b0;
        end
        else begin
            ap_done_reg_0 <= AESL_inst_JetTaggerNN.pointwise_conv_1d_cl_ap_fixed_ap_fixed_48_22_5_3_0_config25_U0.ap_done & ~AESL_inst_JetTaggerNN.pointwise_conv_1d_cl_ap_fixed_ap_fixed_48_22_5_3_0_config25_U0.ap_continue;
        end
    end

    reg ap_done_reg_1;// for module AESL_inst_JetTaggerNN.relu_ap_fixed_48_22_5_3_0_ap_ufixed_9_0_4_0_0_relu_config5_U0
    always @ (negedge dl_reset or posedge dl_clock) begin
        if (~dl_reset) begin
            ap_done_reg_1 <= 'b0;
        end
        else begin
            ap_done_reg_1 <= AESL_inst_JetTaggerNN.relu_ap_fixed_48_22_5_3_0_ap_ufixed_9_0_4_0_0_relu_config5_U0.ap_done & ~AESL_inst_JetTaggerNN.relu_ap_fixed_48_22_5_3_0_ap_ufixed_9_0_4_0_0_relu_config5_U0.ap_continue;
        end
    end

    reg ap_done_reg_2;// for module AESL_inst_JetTaggerNN.pointwise_conv_1d_cl_ap_ufixed_ap_fixed_23_8_5_3_0_config26_U0
    always @ (negedge dl_reset or posedge dl_clock) begin
        if (~dl_reset) begin
            ap_done_reg_2 <= 'b0;
        end
        else begin
            ap_done_reg_2 <= AESL_inst_JetTaggerNN.pointwise_conv_1d_cl_ap_ufixed_ap_fixed_23_8_5_3_0_config26_U0.ap_done & ~AESL_inst_JetTaggerNN.pointwise_conv_1d_cl_ap_ufixed_ap_fixed_23_8_5_3_0_config26_U0.ap_continue;
        end
    end

    reg ap_done_reg_3;// for module AESL_inst_JetTaggerNN.relu_ap_fixed_23_8_5_3_0_ap_ufixed_9_0_4_0_0_relu_config8_U0
    always @ (negedge dl_reset or posedge dl_clock) begin
        if (~dl_reset) begin
            ap_done_reg_3 <= 'b0;
        end
        else begin
            ap_done_reg_3 <= AESL_inst_JetTaggerNN.relu_ap_fixed_23_8_5_3_0_ap_ufixed_9_0_4_0_0_relu_config8_U0.ap_done & ~AESL_inst_JetTaggerNN.relu_ap_fixed_23_8_5_3_0_ap_ufixed_9_0_4_0_0_relu_config8_U0.ap_continue;
        end
    end

    reg ap_done_reg_4;// for module AESL_inst_JetTaggerNN.linear_ap_ufixed_9_0_4_0_0_ap_fixed_18_9_4_0_0_linear_config9_U0
    always @ (negedge dl_reset or posedge dl_clock) begin
        if (~dl_reset) begin
            ap_done_reg_4 <= 'b0;
        end
        else begin
            ap_done_reg_4 <= AESL_inst_JetTaggerNN.linear_ap_ufixed_9_0_4_0_0_ap_fixed_18_9_4_0_0_linear_config9_U0.ap_done & ~AESL_inst_JetTaggerNN.linear_ap_ufixed_9_0_4_0_0_ap_fixed_18_9_4_0_0_linear_config9_U0.ap_continue;
        end
    end

    reg ap_done_reg_5;// for module AESL_inst_JetTaggerNN.global_pooling1d_cl_ap_fixed_ap_fixed_16_6_5_3_0_config10_U0
    always @ (negedge dl_reset or posedge dl_clock) begin
        if (~dl_reset) begin
            ap_done_reg_5 <= 'b0;
        end
        else begin
            ap_done_reg_5 <= AESL_inst_JetTaggerNN.global_pooling1d_cl_ap_fixed_ap_fixed_16_6_5_3_0_config10_U0.ap_done & ~AESL_inst_JetTaggerNN.global_pooling1d_cl_ap_fixed_ap_fixed_16_6_5_3_0_config10_U0.ap_continue;
        end
    end

    reg ap_done_reg_6;// for module AESL_inst_JetTaggerNN.dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_30_14_5_3_0_config11_U0
    always @ (negedge dl_reset or posedge dl_clock) begin
        if (~dl_reset) begin
            ap_done_reg_6 <= 'b0;
        end
        else begin
            ap_done_reg_6 <= AESL_inst_JetTaggerNN.dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_30_14_5_3_0_config11_U0.ap_done & ~AESL_inst_JetTaggerNN.dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_30_14_5_3_0_config11_U0.ap_continue;
        end
    end

    reg ap_done_reg_7;// for module AESL_inst_JetTaggerNN.relu_ap_fixed_30_14_5_3_0_ap_ufixed_9_0_4_0_0_relu_config13_U0
    always @ (negedge dl_reset or posedge dl_clock) begin
        if (~dl_reset) begin
            ap_done_reg_7 <= 'b0;
        end
        else begin
            ap_done_reg_7 <= AESL_inst_JetTaggerNN.relu_ap_fixed_30_14_5_3_0_ap_ufixed_9_0_4_0_0_relu_config13_U0.ap_done & ~AESL_inst_JetTaggerNN.relu_ap_fixed_30_14_5_3_0_ap_ufixed_9_0_4_0_0_relu_config13_U0.ap_continue;
        end
    end

    reg ap_done_reg_8;// for module AESL_inst_JetTaggerNN.dense_latency_ap_ufixed_9_0_4_0_0_ap_fixed_24_9_5_3_0_config14_U0
    always @ (negedge dl_reset or posedge dl_clock) begin
        if (~dl_reset) begin
            ap_done_reg_8 <= 'b0;
        end
        else begin
            ap_done_reg_8 <= AESL_inst_JetTaggerNN.dense_latency_ap_ufixed_9_0_4_0_0_ap_fixed_24_9_5_3_0_config14_U0.ap_done & ~AESL_inst_JetTaggerNN.dense_latency_ap_ufixed_9_0_4_0_0_ap_fixed_24_9_5_3_0_config14_U0.ap_continue;
        end
    end

    reg ap_done_reg_9;// for module AESL_inst_JetTaggerNN.relu_ap_fixed_24_9_5_3_0_ap_ufixed_9_0_4_0_0_relu_config16_U0
    always @ (negedge dl_reset or posedge dl_clock) begin
        if (~dl_reset) begin
            ap_done_reg_9 <= 'b0;
        end
        else begin
            ap_done_reg_9 <= AESL_inst_JetTaggerNN.relu_ap_fixed_24_9_5_3_0_ap_ufixed_9_0_4_0_0_relu_config16_U0.ap_done & ~AESL_inst_JetTaggerNN.relu_ap_fixed_24_9_5_3_0_ap_ufixed_9_0_4_0_0_relu_config16_U0.ap_continue;
        end
    end

    reg ap_done_reg_10;// for module AESL_inst_JetTaggerNN.dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_30_14_5_3_0_config17_U0
    always @ (negedge dl_reset or posedge dl_clock) begin
        if (~dl_reset) begin
            ap_done_reg_10 <= 'b0;
        end
        else begin
            ap_done_reg_10 <= AESL_inst_JetTaggerNN.dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_30_14_5_3_0_config17_U0.ap_done & ~AESL_inst_JetTaggerNN.dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_30_14_5_3_0_config17_U0.ap_continue;
        end
    end

    reg ap_done_reg_11;// for module AESL_inst_JetTaggerNN.dense_latency_ap_ufixed_9_0_4_0_0_ap_fixed_23_8_5_3_0_config19_U0
    always @ (negedge dl_reset or posedge dl_clock) begin
        if (~dl_reset) begin
            ap_done_reg_11 <= 'b0;
        end
        else begin
            ap_done_reg_11 <= AESL_inst_JetTaggerNN.dense_latency_ap_ufixed_9_0_4_0_0_ap_fixed_23_8_5_3_0_config19_U0.ap_done & ~AESL_inst_JetTaggerNN.dense_latency_ap_ufixed_9_0_4_0_0_ap_fixed_23_8_5_3_0_config19_U0.ap_continue;
        end
    end

    reg ap_done_reg_12;// for module AESL_inst_JetTaggerNN.relu_ap_fixed_30_14_5_3_0_ap_ufixed_9_0_4_0_0_relu_config21_U0
    always @ (negedge dl_reset or posedge dl_clock) begin
        if (~dl_reset) begin
            ap_done_reg_12 <= 'b0;
        end
        else begin
            ap_done_reg_12 <= AESL_inst_JetTaggerNN.relu_ap_fixed_30_14_5_3_0_ap_ufixed_9_0_4_0_0_relu_config21_U0.ap_done & ~AESL_inst_JetTaggerNN.relu_ap_fixed_30_14_5_3_0_ap_ufixed_9_0_4_0_0_relu_config21_U0.ap_continue;
        end
    end

    reg ap_done_reg_13;// for module AESL_inst_JetTaggerNN.softmax_latency_ap_fixed_ap_fixed_16_6_5_3_0_softmax_config22_U0
    always @ (negedge dl_reset or posedge dl_clock) begin
        if (~dl_reset) begin
            ap_done_reg_13 <= 'b0;
        end
        else begin
            ap_done_reg_13 <= AESL_inst_JetTaggerNN.softmax_latency_ap_fixed_ap_fixed_16_6_5_3_0_softmax_config22_U0.ap_done & ~AESL_inst_JetTaggerNN.softmax_latency_ap_fixed_ap_fixed_16_6_5_3_0_softmax_config22_U0.ap_continue;
        end
    end

    reg ap_done_reg_14;// for module AESL_inst_JetTaggerNN.dense_latency_ap_ufixed_9_0_4_0_0_ap_fixed_30_12_5_3_0_config23_U0
    always @ (negedge dl_reset or posedge dl_clock) begin
        if (~dl_reset) begin
            ap_done_reg_14 <= 'b0;
        end
        else begin
            ap_done_reg_14 <= AESL_inst_JetTaggerNN.dense_latency_ap_ufixed_9_0_4_0_0_ap_fixed_30_12_5_3_0_config23_U0.ap_done & ~AESL_inst_JetTaggerNN.dense_latency_ap_ufixed_9_0_4_0_0_ap_fixed_30_12_5_3_0_config23_U0.ap_continue;
        end
    end

    // Process: AESL_inst_JetTaggerNN.softmax_latency_ap_fixed_ap_fixed_16_6_5_3_0_softmax_config22_U0
    AESL_deadlock_detect_unit #(2, 0, 1, 1) AESL_deadlock_detect_unit_0 (
        .reset(dl_reset),
        .clock(dl_clock),
        .proc_dep_vld_vec(proc_dep_vld_vec_0),
        .in_chan_dep_vld_vec(in_chan_dep_vld_vec_0),
        .in_chan_dep_data_vec(in_chan_dep_data_vec_0),
        .token_in_vec(token_in_vec_0),
        .dl_detect_in(dl_detect_out),
        .origin(origin[0]),
        .token_clear(token_clear),
        .out_chan_dep_vld_vec(out_chan_dep_vld_vec_0),
        .out_chan_dep_data(out_chan_dep_data_0),
        .token_out_vec(token_out_vec_0),
        .dl_detect_out(dl_in_vec[0]));

    assign proc_0_data_FIFO_blk[0] = 1'b0;
    assign proc_0_data_PIPO_blk[0] = 1'b0;
    assign proc_0_start_FIFO_blk[0] = 1'b0;
    assign proc_0_TLF_FIFO_blk[0] = 1'b0;
    assign proc_0_input_sync_blk[0] = 1'b0;
    assign proc_0_output_sync_blk[0] = 1'b0 | (ap_done_reg_13 & AESL_inst_JetTaggerNN.softmax_latency_ap_fixed_ap_fixed_16_6_5_3_0_softmax_config22_U0.ap_done & ~AESL_inst_JetTaggerNN.dense_latency_ap_ufixed_9_0_4_0_0_ap_fixed_30_12_5_3_0_config23_U0.ap_done);
    assign proc_dep_vld_vec_0[0] = dl_detect_out ? proc_dep_vld_vec_0_reg[0] : (proc_0_data_FIFO_blk[0] | proc_0_data_PIPO_blk[0] | proc_0_start_FIFO_blk[0] | proc_0_TLF_FIFO_blk[0] | proc_0_input_sync_blk[0] | proc_0_output_sync_blk[0]);
    always @ (negedge dl_reset or posedge dl_clock) begin
        if (~dl_reset) begin
            proc_dep_vld_vec_0_reg <= 'b0;
        end
        else begin
            proc_dep_vld_vec_0_reg <= proc_dep_vld_vec_0;
        end
    end
    assign in_chan_dep_vld_vec_0[0] = dep_chan_vld_1_0;
    assign in_chan_dep_data_vec_0[1 : 0] = dep_chan_data_1_0;
    assign token_in_vec_0[0] = token_1_0;
    assign dep_chan_vld_0_1 = out_chan_dep_vld_vec_0[0];
    assign dep_chan_data_0_1 = out_chan_dep_data_0;
    assign token_0_1 = token_out_vec_0[0];

    // Process: AESL_inst_JetTaggerNN.dense_latency_ap_ufixed_9_0_4_0_0_ap_fixed_30_12_5_3_0_config23_U0
    AESL_deadlock_detect_unit #(2, 1, 1, 1) AESL_deadlock_detect_unit_1 (
        .reset(dl_reset),
        .clock(dl_clock),
        .proc_dep_vld_vec(proc_dep_vld_vec_1),
        .in_chan_dep_vld_vec(in_chan_dep_vld_vec_1),
        .in_chan_dep_data_vec(in_chan_dep_data_vec_1),
        .token_in_vec(token_in_vec_1),
        .dl_detect_in(dl_detect_out),
        .origin(origin[1]),
        .token_clear(token_clear),
        .out_chan_dep_vld_vec(out_chan_dep_vld_vec_1),
        .out_chan_dep_data(out_chan_dep_data_1),
        .token_out_vec(token_out_vec_1),
        .dl_detect_out(dl_in_vec[1]));

    assign proc_1_data_FIFO_blk[0] = 1'b0;
    assign proc_1_data_PIPO_blk[0] = 1'b0;
    assign proc_1_start_FIFO_blk[0] = 1'b0;
    assign proc_1_TLF_FIFO_blk[0] = 1'b0;
    assign proc_1_input_sync_blk[0] = 1'b0;
    assign proc_1_output_sync_blk[0] = 1'b0 | (ap_done_reg_14 & AESL_inst_JetTaggerNN.dense_latency_ap_ufixed_9_0_4_0_0_ap_fixed_30_12_5_3_0_config23_U0.ap_done & ~AESL_inst_JetTaggerNN.softmax_latency_ap_fixed_ap_fixed_16_6_5_3_0_softmax_config22_U0.ap_done);
    assign proc_dep_vld_vec_1[0] = dl_detect_out ? proc_dep_vld_vec_1_reg[0] : (proc_1_data_FIFO_blk[0] | proc_1_data_PIPO_blk[0] | proc_1_start_FIFO_blk[0] | proc_1_TLF_FIFO_blk[0] | proc_1_input_sync_blk[0] | proc_1_output_sync_blk[0]);
    always @ (negedge dl_reset or posedge dl_clock) begin
        if (~dl_reset) begin
            proc_dep_vld_vec_1_reg <= 'b0;
        end
        else begin
            proc_dep_vld_vec_1_reg <= proc_dep_vld_vec_1;
        end
    end
    assign in_chan_dep_vld_vec_1[0] = dep_chan_vld_0_1;
    assign in_chan_dep_data_vec_1[1 : 0] = dep_chan_data_0_1;
    assign token_in_vec_1[0] = token_0_1;
    assign dep_chan_vld_1_0 = out_chan_dep_vld_vec_1[0];
    assign dep_chan_data_1_0 = out_chan_dep_data_1;
    assign token_1_0 = token_out_vec_1[0];


    wire [1:0] dl_in_vec_comb = dl_in_vec & ~{1{all_finish}};
    AESL_deadlock_report_unit #(2) AESL_deadlock_report_unit_inst (
        .dl_reset(dl_reset),
        .dl_clock(dl_clock),
        .dl_in_vec(dl_in_vec_comb),
        .ap_done_reg_0(ap_done_reg_0),
        .ap_done_reg_1(ap_done_reg_1),
        .ap_done_reg_2(ap_done_reg_2),
        .ap_done_reg_3(ap_done_reg_3),
        .ap_done_reg_4(ap_done_reg_4),
        .ap_done_reg_5(ap_done_reg_5),
        .ap_done_reg_6(ap_done_reg_6),
        .ap_done_reg_7(ap_done_reg_7),
        .ap_done_reg_8(ap_done_reg_8),
        .ap_done_reg_9(ap_done_reg_9),
        .ap_done_reg_10(ap_done_reg_10),
        .ap_done_reg_11(ap_done_reg_11),
        .ap_done_reg_12(ap_done_reg_12),
        .ap_done_reg_13(ap_done_reg_13),
        .ap_done_reg_14(ap_done_reg_14),
        .dl_detect_out(dl_detect_out),
        .origin(origin),
        .token_clear(token_clear));

endmodule
