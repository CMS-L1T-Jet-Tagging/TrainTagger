// ==============================================================
// RTL generated by Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2022.2 (64-bit)
// Version: 2022.2
// Copyright (C) Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// 
// ===========================================================

`timescale 1 ns / 1 ps 

module JetTaggerNN_dense_latency_ap_ufixed_9_0_4_0_0_ap_fixed_30_12_5_3_0_config23_s (
        ap_clk,
        ap_rst,
        ap_start,
        ap_done,
        ap_continue,
        ap_idle,
        ap_ready,
        p_read,
        p_read1,
        p_read2,
        p_read3,
        p_read4,
        p_read5,
        p_read6,
        p_read7,
        p_read8,
        layer23_out,
        layer23_out_ap_vld
);

parameter    ap_ST_fsm_pp0_stage0 = 1'd1;

input   ap_clk;
input   ap_rst;
input   ap_start;
output   ap_done;
input   ap_continue;
output   ap_idle;
output   ap_ready;
input  [8:0] p_read;
input  [8:0] p_read1;
input  [8:0] p_read2;
input  [8:0] p_read3;
input  [8:0] p_read4;
input  [8:0] p_read5;
input  [8:0] p_read6;
input  [8:0] p_read7;
input  [8:0] p_read8;
output  [29:0] layer23_out;
output   layer23_out_ap_vld;

reg ap_done;
reg ap_idle;
reg ap_ready;
reg[29:0] layer23_out;
reg layer23_out_ap_vld;

(* fsm_encoding = "none" *) reg   [0:0] ap_CS_fsm;
wire    ap_CS_fsm_pp0_stage0;
wire    ap_enable_reg_pp0_iter0;
reg    ap_enable_reg_pp0_iter1;
reg    ap_enable_reg_pp0_iter2;
reg    ap_enable_reg_pp0_iter3;
reg    ap_idle_pp0;
reg    ap_done_reg;
reg    ap_block_state1_pp0_stage0_iter0;
wire    ap_block_state2_pp0_stage0_iter1;
wire    ap_block_state3_pp0_stage0_iter2;
wire    ap_block_state4_pp0_stage0_iter3;
reg    ap_block_pp0_stage0_subdone;
reg   [8:0] p_read817_reg_681;
reg    ap_block_pp0_stage0_11001;
reg   [8:0] p_read716_reg_686;
reg   [8:0] p_read615_reg_691;
reg   [8:0] p_read514_reg_696;
reg   [8:0] p_read413_reg_701;
reg   [8:0] p_read211_reg_707;
reg   [8:0] p_read_1472_reg_713;
reg   [6:0] trunc_ln818_s_reg_718;
reg   [6:0] lshr_ln818_s_reg_723;
reg   [11:0] trunc_ln_reg_728;
reg   [9:0] trunc_ln818_3221_reg_733;
reg   [8:0] tmp_reg_738;
reg   [9:0] trunc_ln818_3222_reg_743;
reg   [9:0] tmp_s_reg_748;
reg   [9:0] trunc_ln818_3223_reg_753;
reg   [7:0] trunc_ln6_reg_758;
wire   [8:0] add_ln813_4339_fu_586_p2;
reg   [8:0] add_ln813_4339_reg_763;
wire   [12:0] add_ln813_4336_fu_631_p2;
reg   [12:0] add_ln813_4336_reg_768;
wire   [11:0] add_ln813_4340_fu_650_p2;
reg   [11:0] add_ln813_4340_reg_773;
wire  signed [29:0] sext_ln66_fu_676_p1;
reg   [29:0] layer23_out_preg;
reg    ap_block_pp0_stage0_01001;
wire   [8:0] r_V_fu_163_p0;
wire  signed [10:0] r_V_fu_163_p1;
wire    ap_block_pp0_stage0;
wire   [8:0] r_V_3537_fu_164_p0;
wire  signed [8:0] r_V_3537_fu_164_p1;
wire   [8:0] r_V_3536_fu_167_p0;
wire   [9:0] r_V_3536_fu_167_p1;
wire   [8:0] mul_ln818_fu_168_p0;
wire   [7:0] mul_ln818_fu_168_p1;
wire   [8:0] r_V_3535_fu_169_p0;
wire   [9:0] r_V_3535_fu_169_p1;
wire   [13:0] shl_ln_fu_386_p3;
wire   [14:0] zext_ln1273_fu_394_p1;
wire   [14:0] r_V_3532_fu_398_p2;
wire   [19:0] r_V_fu_163_p2;
wire   [16:0] shl_ln1273_s_fu_441_p3;
wire   [10:0] shl_ln1273_1419_fu_452_p3;
wire   [17:0] zext_ln1273_823_fu_448_p1;
wire   [17:0] zext_ln1273_824_fu_459_p1;
wire   [17:0] r_V_3533_fu_463_p2;
wire   [15:0] shl_ln1273_1420_fu_482_p3;
wire   [12:0] shl_ln1273_1421_fu_493_p3;
wire   [16:0] zext_ln1273_825_fu_489_p1;
wire   [16:0] zext_ln1273_826_fu_500_p1;
wire   [16:0] r_V_3534_fu_504_p2;
wire   [17:0] r_V_3535_fu_169_p2;
wire   [17:0] r_V_3536_fu_167_p2;
wire   [17:0] r_V_3537_fu_164_p2;
wire   [15:0] mul_ln818_fu_168_p2;
wire  signed [7:0] sext_ln70_64_fu_438_p1;
wire   [7:0] add_ln813_4338_fu_576_p2;
wire   [8:0] zext_ln813_396_fu_582_p1;
wire   [8:0] zext_ln818_1072_fu_479_p1;
wire   [9:0] zext_ln70_96_fu_598_p1;
wire   [9:0] add_ln813_fu_610_p2;
wire   [12:0] zext_ln813_fu_607_p1;
wire  signed [12:0] sext_ln70_fu_592_p1;
wire   [12:0] add_ln813_4334_fu_619_p2;
wire   [12:0] zext_ln70_99_fu_601_p1;
wire   [12:0] add_ln813_4335_fu_625_p2;
wire   [12:0] zext_ln813_395_fu_615_p1;
wire  signed [10:0] sext_ln70_65_fu_595_p1;
wire  signed [10:0] sext_ln818_fu_604_p1;
wire   [10:0] add_ln813_4337_fu_637_p2;
wire   [11:0] zext_ln813_397_fu_647_p1;
wire  signed [11:0] sext_ln813_629_fu_643_p1;
wire  signed [13:0] sext_ln813_630_fu_659_p1;
wire  signed [13:0] sext_ln813_fu_656_p1;
wire   [13:0] x_V_fu_662_p2;
wire   [21:0] tmp_423_fu_668_p3;
reg   [0:0] ap_NS_fsm;
reg    ap_idle_pp0_0to2;
reg    ap_reset_idle_pp0;
wire    ap_enable_pp0;
wire   [15:0] mul_ln818_fu_168_p00;
wire   [17:0] r_V_3535_fu_169_p00;
wire   [17:0] r_V_3536_fu_167_p00;
wire   [17:0] r_V_3537_fu_164_p00;
wire   [19:0] r_V_fu_163_p00;
wire    ap_ce_reg;

// power-on initialization
initial begin
#0 ap_CS_fsm = 1'd1;
#0 ap_enable_reg_pp0_iter1 = 1'b0;
#0 ap_enable_reg_pp0_iter2 = 1'b0;
#0 ap_enable_reg_pp0_iter3 = 1'b0;
#0 ap_done_reg = 1'b0;
#0 layer23_out_preg = 30'd0;
end

JetTaggerNN_mul_9ns_11s_20_1_1 #(
    .ID( 1 ),
    .NUM_STAGE( 1 ),
    .din0_WIDTH( 9 ),
    .din1_WIDTH( 11 ),
    .dout_WIDTH( 20 ))
mul_9ns_11s_20_1_1_U3716(
    .din0(r_V_fu_163_p0),
    .din1(r_V_fu_163_p1),
    .dout(r_V_fu_163_p2)
);

JetTaggerNN_mul_9ns_9s_18_1_1 #(
    .ID( 1 ),
    .NUM_STAGE( 1 ),
    .din0_WIDTH( 9 ),
    .din1_WIDTH( 9 ),
    .dout_WIDTH( 18 ))
mul_9ns_9s_18_1_1_U3717(
    .din0(r_V_3537_fu_164_p0),
    .din1(r_V_3537_fu_164_p1),
    .dout(r_V_3537_fu_164_p2)
);

JetTaggerNN_mul_9ns_10ns_18_1_1 #(
    .ID( 1 ),
    .NUM_STAGE( 1 ),
    .din0_WIDTH( 9 ),
    .din1_WIDTH( 10 ),
    .dout_WIDTH( 18 ))
mul_9ns_10ns_18_1_1_U3718(
    .din0(r_V_3536_fu_167_p0),
    .din1(r_V_3536_fu_167_p1),
    .dout(r_V_3536_fu_167_p2)
);

JetTaggerNN_mul_9ns_8ns_16_1_1 #(
    .ID( 1 ),
    .NUM_STAGE( 1 ),
    .din0_WIDTH( 9 ),
    .din1_WIDTH( 8 ),
    .dout_WIDTH( 16 ))
mul_9ns_8ns_16_1_1_U3719(
    .din0(mul_ln818_fu_168_p0),
    .din1(mul_ln818_fu_168_p1),
    .dout(mul_ln818_fu_168_p2)
);

JetTaggerNN_mul_9ns_10ns_18_1_1 #(
    .ID( 1 ),
    .NUM_STAGE( 1 ),
    .din0_WIDTH( 9 ),
    .din1_WIDTH( 10 ),
    .dout_WIDTH( 18 ))
mul_9ns_10ns_18_1_1_U3720(
    .din0(r_V_3535_fu_169_p0),
    .din1(r_V_3535_fu_169_p1),
    .dout(r_V_3535_fu_169_p2)
);

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_CS_fsm <= ap_ST_fsm_pp0_stage0;
    end else begin
        ap_CS_fsm <= ap_NS_fsm;
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_done_reg <= 1'b0;
    end else begin
        if ((ap_continue == 1'b1)) begin
            ap_done_reg <= 1'b0;
        end else if (((1'b0 == ap_block_pp0_stage0_subdone) & (ap_enable_reg_pp0_iter3 == 1'b1))) begin
            ap_done_reg <= 1'b1;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_enable_reg_pp0_iter1 <= 1'b0;
    end else begin
        if (((1'b0 == ap_block_pp0_stage0_subdone) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
            ap_enable_reg_pp0_iter1 <= ap_start;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_enable_reg_pp0_iter2 <= 1'b0;
    end else begin
        if ((1'b0 == ap_block_pp0_stage0_subdone)) begin
            ap_enable_reg_pp0_iter2 <= ap_enable_reg_pp0_iter1;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_enable_reg_pp0_iter3 <= 1'b0;
    end else begin
        if ((1'b0 == ap_block_pp0_stage0_subdone)) begin
            ap_enable_reg_pp0_iter3 <= ap_enable_reg_pp0_iter2;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
                layer23_out_preg[8] <= 1'b0;
        layer23_out_preg[9] <= 1'b0;
        layer23_out_preg[10] <= 1'b0;
        layer23_out_preg[11] <= 1'b0;
        layer23_out_preg[12] <= 1'b0;
        layer23_out_preg[13] <= 1'b0;
        layer23_out_preg[14] <= 1'b0;
        layer23_out_preg[15] <= 1'b0;
        layer23_out_preg[16] <= 1'b0;
        layer23_out_preg[17] <= 1'b0;
        layer23_out_preg[18] <= 1'b0;
        layer23_out_preg[19] <= 1'b0;
        layer23_out_preg[20] <= 1'b0;
        layer23_out_preg[21] <= 1'b0;
        layer23_out_preg[22] <= 1'b0;
        layer23_out_preg[23] <= 1'b0;
        layer23_out_preg[24] <= 1'b0;
        layer23_out_preg[25] <= 1'b0;
        layer23_out_preg[26] <= 1'b0;
        layer23_out_preg[27] <= 1'b0;
        layer23_out_preg[28] <= 1'b0;
        layer23_out_preg[29] <= 1'b0;
    end else begin
        if (((1'b0 == ap_block_pp0_stage0_01001) & (ap_enable_reg_pp0_iter3 == 1'b1))) begin
                        layer23_out_preg[29 : 8] <= sext_ln66_fu_676_p1[29 : 8];
        end
    end
end

always @ (posedge ap_clk) begin
    if ((1'b0 == ap_block_pp0_stage0_11001)) begin
        add_ln813_4336_reg_768 <= add_ln813_4336_fu_631_p2;
        add_ln813_4340_reg_773 <= add_ln813_4340_fu_650_p2;
    end
end

always @ (posedge ap_clk) begin
    if (((1'b0 == ap_block_pp0_stage0_11001) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        add_ln813_4339_reg_763 <= add_ln813_4339_fu_586_p2;
        lshr_ln818_s_reg_723 <= {{p_read3[8:2]}};
        p_read211_reg_707 <= p_read2;
        p_read413_reg_701 <= p_read4;
        p_read514_reg_696 <= p_read5;
        p_read615_reg_691 <= p_read6;
        p_read716_reg_686 <= p_read7;
        p_read817_reg_681 <= p_read8;
        p_read_1472_reg_713 <= p_read;
        tmp_reg_738 <= {{r_V_3534_fu_504_p2[16:8]}};
        tmp_s_reg_748 <= {{r_V_3536_fu_167_p2[17:8]}};
        trunc_ln6_reg_758 <= {{mul_ln818_fu_168_p2[15:8]}};
        trunc_ln818_3221_reg_733 <= {{r_V_3533_fu_463_p2[17:8]}};
        trunc_ln818_3222_reg_743 <= {{r_V_3535_fu_169_p2[17:8]}};
        trunc_ln818_3223_reg_753 <= {{r_V_3537_fu_164_p2[17:8]}};
        trunc_ln818_s_reg_718 <= {{r_V_3532_fu_398_p2[14:8]}};
        trunc_ln_reg_728 <= {{r_V_fu_163_p2[19:8]}};
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_pp0_stage0_subdone) & (ap_enable_reg_pp0_iter3 == 1'b1))) begin
        ap_done = 1'b1;
    end else begin
        ap_done = ap_done_reg;
    end
end

always @ (*) begin
    if (((ap_start == 1'b0) & (ap_idle_pp0 == 1'b1) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        ap_idle = 1'b1;
    end else begin
        ap_idle = 1'b0;
    end
end

always @ (*) begin
    if (((ap_enable_reg_pp0_iter3 == 1'b0) & (ap_enable_reg_pp0_iter2 == 1'b0) & (ap_enable_reg_pp0_iter1 == 1'b0) & (ap_enable_reg_pp0_iter0 == 1'b0))) begin
        ap_idle_pp0 = 1'b1;
    end else begin
        ap_idle_pp0 = 1'b0;
    end
end

always @ (*) begin
    if (((ap_enable_reg_pp0_iter2 == 1'b0) & (ap_enable_reg_pp0_iter1 == 1'b0) & (ap_enable_reg_pp0_iter0 == 1'b0))) begin
        ap_idle_pp0_0to2 = 1'b1;
    end else begin
        ap_idle_pp0_0to2 = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_pp0_stage0_subdone) & (ap_start == 1'b1) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        ap_ready = 1'b1;
    end else begin
        ap_ready = 1'b0;
    end
end

always @ (*) begin
    if (((ap_start == 1'b0) & (ap_idle_pp0_0to2 == 1'b1))) begin
        ap_reset_idle_pp0 = 1'b1;
    end else begin
        ap_reset_idle_pp0 = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_pp0_stage0_01001) & (ap_enable_reg_pp0_iter3 == 1'b1))) begin
        layer23_out = sext_ln66_fu_676_p1;
    end else begin
        layer23_out = layer23_out_preg;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_pp0_stage0_11001) & (ap_enable_reg_pp0_iter3 == 1'b1))) begin
        layer23_out_ap_vld = 1'b1;
    end else begin
        layer23_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    case (ap_CS_fsm)
        ap_ST_fsm_pp0_stage0 : begin
            ap_NS_fsm = ap_ST_fsm_pp0_stage0;
        end
        default : begin
            ap_NS_fsm = 'bx;
        end
    endcase
end

assign add_ln813_4334_fu_619_p2 = ($signed(zext_ln813_fu_607_p1) + $signed(sext_ln70_fu_592_p1));

assign add_ln813_4335_fu_625_p2 = (add_ln813_4334_fu_619_p2 + zext_ln70_99_fu_601_p1);

assign add_ln813_4336_fu_631_p2 = (add_ln813_4335_fu_625_p2 + zext_ln813_395_fu_615_p1);

assign add_ln813_4337_fu_637_p2 = ($signed(sext_ln70_65_fu_595_p1) + $signed(sext_ln818_fu_604_p1));

assign add_ln813_4338_fu_576_p2 = ($signed(sext_ln70_64_fu_438_p1) + $signed(8'd108));

assign add_ln813_4339_fu_586_p2 = (zext_ln813_396_fu_582_p1 + zext_ln818_1072_fu_479_p1);

assign add_ln813_4340_fu_650_p2 = ($signed(zext_ln813_397_fu_647_p1) + $signed(sext_ln813_629_fu_643_p1));

assign add_ln813_fu_610_p2 = (zext_ln70_96_fu_598_p1 + trunc_ln818_3222_reg_743);

assign ap_CS_fsm_pp0_stage0 = ap_CS_fsm[32'd0];

assign ap_block_pp0_stage0 = ~(1'b1 == 1'b1);

always @ (*) begin
    ap_block_pp0_stage0_01001 = ((ap_done_reg == 1'b1) | ((ap_done_reg == 1'b1) & (ap_start == 1'b1)));
end

always @ (*) begin
    ap_block_pp0_stage0_11001 = ((ap_done_reg == 1'b1) | ((ap_done_reg == 1'b1) & (ap_start == 1'b1)));
end

always @ (*) begin
    ap_block_pp0_stage0_subdone = ((ap_done_reg == 1'b1) | ((ap_done_reg == 1'b1) & (ap_start == 1'b1)));
end

always @ (*) begin
    ap_block_state1_pp0_stage0_iter0 = (ap_done_reg == 1'b1);
end

assign ap_block_state2_pp0_stage0_iter1 = ~(1'b1 == 1'b1);

assign ap_block_state3_pp0_stage0_iter2 = ~(1'b1 == 1'b1);

assign ap_block_state4_pp0_stage0_iter3 = ~(1'b1 == 1'b1);

assign ap_enable_pp0 = (ap_idle_pp0 ^ 1'b1);

assign ap_enable_reg_pp0_iter0 = ap_start;

assign mul_ln818_fu_168_p0 = mul_ln818_fu_168_p00;

assign mul_ln818_fu_168_p00 = p_read817_reg_681;

assign mul_ln818_fu_168_p1 = 16'd113;

assign r_V_3532_fu_398_p2 = (15'd0 - zext_ln1273_fu_394_p1);

assign r_V_3533_fu_463_p2 = (zext_ln1273_823_fu_448_p1 - zext_ln1273_824_fu_459_p1);

assign r_V_3534_fu_504_p2 = (zext_ln1273_825_fu_489_p1 + zext_ln1273_826_fu_500_p1);

assign r_V_3535_fu_169_p0 = r_V_3535_fu_169_p00;

assign r_V_3535_fu_169_p00 = p_read514_reg_696;

assign r_V_3535_fu_169_p1 = 18'd269;

assign r_V_3536_fu_167_p0 = r_V_3536_fu_167_p00;

assign r_V_3536_fu_167_p00 = p_read615_reg_691;

assign r_V_3536_fu_167_p1 = 18'd308;

assign r_V_3537_fu_164_p0 = r_V_3537_fu_164_p00;

assign r_V_3537_fu_164_p00 = p_read716_reg_686;

assign r_V_3537_fu_164_p1 = 18'd261932;

assign r_V_fu_163_p0 = r_V_fu_163_p00;

assign r_V_fu_163_p00 = p_read_1472_reg_713;

assign r_V_fu_163_p1 = 20'd1047761;

assign sext_ln66_fu_676_p1 = $signed(tmp_423_fu_668_p3);

assign sext_ln70_64_fu_438_p1 = $signed(trunc_ln818_s_reg_718);

assign sext_ln70_65_fu_595_p1 = $signed(trunc_ln818_3221_reg_733);

assign sext_ln70_fu_592_p1 = $signed(trunc_ln_reg_728);

assign sext_ln813_629_fu_643_p1 = $signed(add_ln813_4337_fu_637_p2);

assign sext_ln813_630_fu_659_p1 = $signed(add_ln813_4340_reg_773);

assign sext_ln813_fu_656_p1 = $signed(add_ln813_4336_reg_768);

assign sext_ln818_fu_604_p1 = $signed(trunc_ln818_3223_reg_753);

assign shl_ln1273_1419_fu_452_p3 = {{p_read211_reg_707}, {2'd0}};

assign shl_ln1273_1420_fu_482_p3 = {{p_read413_reg_701}, {7'd0}};

assign shl_ln1273_1421_fu_493_p3 = {{p_read413_reg_701}, {4'd0}};

assign shl_ln1273_s_fu_441_p3 = {{p_read211_reg_707}, {8'd0}};

assign shl_ln_fu_386_p3 = {{p_read1}, {5'd0}};

assign tmp_423_fu_668_p3 = {{x_V_fu_662_p2}, {8'd0}};

assign x_V_fu_662_p2 = ($signed(sext_ln813_630_fu_659_p1) + $signed(sext_ln813_fu_656_p1));

assign zext_ln1273_823_fu_448_p1 = shl_ln1273_s_fu_441_p3;

assign zext_ln1273_824_fu_459_p1 = shl_ln1273_1419_fu_452_p3;

assign zext_ln1273_825_fu_489_p1 = shl_ln1273_1420_fu_482_p3;

assign zext_ln1273_826_fu_500_p1 = shl_ln1273_1421_fu_493_p3;

assign zext_ln1273_fu_394_p1 = shl_ln_fu_386_p3;

assign zext_ln70_96_fu_598_p1 = tmp_reg_738;

assign zext_ln70_99_fu_601_p1 = tmp_s_reg_748;

assign zext_ln813_395_fu_615_p1 = add_ln813_fu_610_p2;

assign zext_ln813_396_fu_582_p1 = add_ln813_4338_fu_576_p2;

assign zext_ln813_397_fu_647_p1 = add_ln813_4339_reg_763;

assign zext_ln813_fu_607_p1 = trunc_ln6_reg_758;

assign zext_ln818_1072_fu_479_p1 = lshr_ln818_s_reg_723;

always @ (posedge ap_clk) begin
    layer23_out_preg[7:0] <= 8'b00000000;
end

endmodule //JetTaggerNN_dense_latency_ap_ufixed_9_0_4_0_0_ap_fixed_30_12_5_3_0_config23_s
