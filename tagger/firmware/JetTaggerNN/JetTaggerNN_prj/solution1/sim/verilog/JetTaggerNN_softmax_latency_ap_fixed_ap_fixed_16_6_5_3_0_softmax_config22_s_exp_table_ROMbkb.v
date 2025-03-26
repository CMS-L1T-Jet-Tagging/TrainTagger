// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2022.2 (64-bit)
// Version: 2022.2
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// ==============================================================
`timescale 1 ns / 1 ps
module JetTaggerNN_softmax_latency_ap_fixed_ap_fixed_16_6_5_3_0_softmax_config22_s_exp_table_ROMbkb (
    address0, ce0, q0, 
    address1, ce1, q1, 
    address2, ce2, q2, 
    address3, ce3, q3, 
    address4, ce4, q4, 
    address5, ce5, q5, 
    address6, ce6, q6, 
    address7, ce7, q7, 
    reset, clk);

parameter DataWidth = 17;
parameter AddressWidth = 10;
parameter AddressRange = 1024;
 
input[AddressWidth-1:0] address0;
input ce0;
output reg[DataWidth-1:0] q0;
 
input[AddressWidth-1:0] address1;
input ce1;
output reg[DataWidth-1:0] q1;
 
input[AddressWidth-1:0] address2;
input ce2;
output reg[DataWidth-1:0] q2;
 
input[AddressWidth-1:0] address3;
input ce3;
output reg[DataWidth-1:0] q3;
 
input[AddressWidth-1:0] address4;
input ce4;
output reg[DataWidth-1:0] q4;
 
input[AddressWidth-1:0] address5;
input ce5;
output reg[DataWidth-1:0] q5;
 
input[AddressWidth-1:0] address6;
input ce6;
output reg[DataWidth-1:0] q6;
 
input[AddressWidth-1:0] address7;
input ce7;
output reg[DataWidth-1:0] q7;

input reset;
input clk;

 
reg [DataWidth-1:0] rom0[0:AddressRange-1];
 
reg [DataWidth-1:0] rom1[0:AddressRange-1];
 
reg [DataWidth-1:0] rom2[0:AddressRange-1];
 
reg [DataWidth-1:0] rom3[0:AddressRange-1];


initial begin
     
    $readmemh("./JetTaggerNN_softmax_latency_ap_fixed_ap_fixed_16_6_5_3_0_softmax_config22_s_exp_table_ROMbkb.dat", rom0); 
    $readmemh("./JetTaggerNN_softmax_latency_ap_fixed_ap_fixed_16_6_5_3_0_softmax_config22_s_exp_table_ROMbkb.dat", rom1); 
    $readmemh("./JetTaggerNN_softmax_latency_ap_fixed_ap_fixed_16_6_5_3_0_softmax_config22_s_exp_table_ROMbkb.dat", rom2); 
    $readmemh("./JetTaggerNN_softmax_latency_ap_fixed_ap_fixed_16_6_5_3_0_softmax_config22_s_exp_table_ROMbkb.dat", rom3);
end

  
always @(posedge clk) 
begin 
    if (ce0) 
    begin
        q0 <= rom0[address0];
    end
end
  
always @(posedge clk) 
begin 
    if (ce1) 
    begin
        q1 <= rom0[address1];
    end
end
  
always @(posedge clk) 
begin 
    if (ce2) 
    begin
        q2 <= rom1[address2];
    end
end
  
always @(posedge clk) 
begin 
    if (ce3) 
    begin
        q3 <= rom1[address3];
    end
end
  
always @(posedge clk) 
begin 
    if (ce4) 
    begin
        q4 <= rom2[address4];
    end
end
  
always @(posedge clk) 
begin 
    if (ce5) 
    begin
        q5 <= rom2[address5];
    end
end
  
always @(posedge clk) 
begin 
    if (ce6) 
    begin
        q6 <= rom3[address6];
    end
end
  
always @(posedge clk) 
begin 
    if (ce7) 
    begin
        q7 <= rom3[address7];
    end
end


endmodule

