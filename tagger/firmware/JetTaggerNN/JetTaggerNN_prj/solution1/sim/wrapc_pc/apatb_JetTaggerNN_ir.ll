; ModuleID = '/software/bm24156/TrainTagger2/TrainTagger/tagger/firmware/JetTaggerNN/JetTaggerNN_prj/solution1/.autopilot/db/a.g.ld.5.gdce.bc'
source_filename = "llvm-link"
target datalayout = "e-m:e-i64:64-i128:128-i256:256-i512:512-i1024:1024-i2048:2048-i4096:4096-n8:16:32:64-S128-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "fpga64-xilinx-none"

%"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>" = type { %"struct.ap_fixed_base<16, 6, true, AP_TRN, AP_WRAP, 0>" }
%"struct.ap_fixed_base<16, 6, true, AP_TRN, AP_WRAP, 0>" = type { %"struct.ssdm_int<16, true>" }
%"struct.ssdm_int<16, true>" = type { i16 }
%"struct.ap_fixed<30, 12, AP_TRN, AP_WRAP, 0>" = type { %"struct.ap_fixed_base<30, 12, true, AP_TRN, AP_WRAP, 0>" }
%"struct.ap_fixed_base<30, 12, true, AP_TRN, AP_WRAP, 0>" = type { %"struct.ssdm_int<30, true>" }
%"struct.ssdm_int<30, true>" = type { i30 }

; Function Attrs: noinline
define void @apatb_JetTaggerNN_ir(%"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"* noalias nocapture nonnull readonly "fpga.decayed.dim.hint"="320" %model_input, %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"* noalias nocapture nonnull "fpga.decayed.dim.hint"="8" %layer22_out, %"struct.ap_fixed<30, 12, AP_TRN, AP_WRAP, 0>"* noalias nocapture nonnull "fpga.decayed.dim.hint"="1" %layer23_out) local_unnamed_addr #0 {
entry:
  %model_input_copy20 = alloca i5120, align 512
  %layer22_out_copy_0 = alloca i16, align 512
  %layer22_out_copy_1 = alloca i16, align 512
  %layer22_out_copy_2 = alloca i16, align 512
  %layer22_out_copy_3 = alloca i16, align 512
  %layer22_out_copy_4 = alloca i16, align 512
  %layer22_out_copy_5 = alloca i16, align 512
  %layer22_out_copy_6 = alloca i16, align 512
  %layer22_out_copy_7 = alloca i16, align 512
  %layer23_out_copy19 = alloca i30, align 512
  %0 = bitcast %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"* %model_input to [320 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]*
  %1 = bitcast %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"* %layer22_out to [8 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]*
  %2 = bitcast %"struct.ap_fixed<30, 12, AP_TRN, AP_WRAP, 0>"* %layer23_out to [1 x %"struct.ap_fixed<30, 12, AP_TRN, AP_WRAP, 0>"]*
  call void @copy_in([320 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* nonnull %0, i5120* nonnull align 512 %model_input_copy20, [8 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* nonnull %1, i16* nonnull align 512 %layer22_out_copy_0, i16* nonnull align 512 %layer22_out_copy_1, i16* nonnull align 512 %layer22_out_copy_2, i16* nonnull align 512 %layer22_out_copy_3, i16* nonnull align 512 %layer22_out_copy_4, i16* nonnull align 512 %layer22_out_copy_5, i16* nonnull align 512 %layer22_out_copy_6, i16* nonnull align 512 %layer22_out_copy_7, [1 x %"struct.ap_fixed<30, 12, AP_TRN, AP_WRAP, 0>"]* nonnull %2, i30* nonnull align 512 %layer23_out_copy19)
  call void @apatb_JetTaggerNN_hw(i5120* %model_input_copy20, i16* %layer22_out_copy_0, i16* %layer22_out_copy_1, i16* %layer22_out_copy_2, i16* %layer22_out_copy_3, i16* %layer22_out_copy_4, i16* %layer22_out_copy_5, i16* %layer22_out_copy_6, i16* %layer22_out_copy_7, i30* %layer23_out_copy19)
  call void @copy_back([320 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* %0, i5120* %model_input_copy20, [8 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* %1, i16* %layer22_out_copy_0, i16* %layer22_out_copy_1, i16* %layer22_out_copy_2, i16* %layer22_out_copy_3, i16* %layer22_out_copy_4, i16* %layer22_out_copy_5, i16* %layer22_out_copy_6, i16* %layer22_out_copy_7, [1 x %"struct.ap_fixed<30, 12, AP_TRN, AP_WRAP, 0>"]* %2, i30* %layer23_out_copy19)
  ret void
}

; Function Attrs: argmemonly noinline norecurse
define internal void @"onebyonecpy_hls.p0a8struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>.388"(i16* noalias nocapture align 512 "fpga.caller.interfaces"="layout_transformed" "orig.arg.no"="0" "unpacked"="0.0.0.0.0" %_0, i16* noalias nocapture align 512 "fpga.caller.interfaces"="layout_transformed" "orig.arg.no"="0" "unpacked"="0.0.0.0.1" %_1, i16* noalias nocapture align 512 "fpga.caller.interfaces"="layout_transformed" "orig.arg.no"="0" "unpacked"="0.0.0.0.2" %_2, i16* noalias nocapture align 512 "fpga.caller.interfaces"="layout_transformed" "orig.arg.no"="0" "unpacked"="0.0.0.0.3" %_3, i16* noalias nocapture align 512 "fpga.caller.interfaces"="layout_transformed" "orig.arg.no"="0" "unpacked"="0.0.0.0.4" %_4, i16* noalias nocapture align 512 "fpga.caller.interfaces"="layout_transformed" "orig.arg.no"="0" "unpacked"="0.0.0.0.5" %_5, i16* noalias nocapture align 512 "fpga.caller.interfaces"="layout_transformed" "orig.arg.no"="0" "unpacked"="0.0.0.0.6" %_6, i16* noalias nocapture align 512 "fpga.caller.interfaces"="layout_transformed" "orig.arg.no"="0" "unpacked"="0.0.0.0.7" %_7, [8 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* noalias readonly "orig.arg.no"="1" "unpacked"="1") #1 {
entry:
  %1 = icmp eq [8 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* %0, null
  br i1 %1, label %ret, label %copy

copy:                                             ; preds = %entry
  br label %for.loop

for.loop:                                         ; preds = %dst.addr.0.0.06.exit, %copy
  %for.loop.idx1 = phi i64 [ 0, %copy ], [ %for.loop.idx.next, %dst.addr.0.0.06.exit ]
  %src.addr.0.0.05 = getelementptr [8 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"], [8 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* %0, i64 0, i64 %for.loop.idx1, i32 0, i32 0, i32 0
  %2 = load i16, i16* %src.addr.0.0.05, align 2
  %3 = trunc i64 %for.loop.idx1 to i3
  switch i3 %3, label %dst.addr.0.0.06.case.7 [
    i3 0, label %dst.addr.0.0.06.case.0
    i3 1, label %dst.addr.0.0.06.case.1
    i3 2, label %dst.addr.0.0.06.case.2
    i3 3, label %dst.addr.0.0.06.case.3
    i3 -4, label %dst.addr.0.0.06.case.4
    i3 -3, label %dst.addr.0.0.06.case.5
    i3 -2, label %dst.addr.0.0.06.case.6
  ]

dst.addr.0.0.06.case.0:                           ; preds = %for.loop
  store i16 %2, i16* %_0, align 512
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.1:                           ; preds = %for.loop
  store i16 %2, i16* %_1, align 512
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.2:                           ; preds = %for.loop
  store i16 %2, i16* %_2, align 512
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.3:                           ; preds = %for.loop
  store i16 %2, i16* %_3, align 512
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.4:                           ; preds = %for.loop
  store i16 %2, i16* %_4, align 512
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.5:                           ; preds = %for.loop
  store i16 %2, i16* %_5, align 512
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.6:                           ; preds = %for.loop
  store i16 %2, i16* %_6, align 512
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.7:                           ; preds = %for.loop
  store i16 %2, i16* %_7, align 512
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.exit:                             ; preds = %dst.addr.0.0.06.case.7, %dst.addr.0.0.06.case.6, %dst.addr.0.0.06.case.5, %dst.addr.0.0.06.case.4, %dst.addr.0.0.06.case.3, %dst.addr.0.0.06.case.2, %dst.addr.0.0.06.case.1, %dst.addr.0.0.06.case.0
  %for.loop.idx.next = add nuw nsw i64 %for.loop.idx1, 1
  %exitcond = icmp ne i64 %for.loop.idx.next, 8
  br i1 %exitcond, label %for.loop, label %ret

ret:                                              ; preds = %dst.addr.0.0.06.exit, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse
define internal void @"onebyonecpy_hls.p0a1struct.ap_fixed<30, 12, AP_TRN, AP_WRAP, 0>"(i30* noalias nocapture align 512 "fpga.caller.interfaces"="layout_transformed" "orig.arg.no"="0" "unpacked"="0.0.0.0", [1 x %"struct.ap_fixed<30, 12, AP_TRN, AP_WRAP, 0>"]* noalias readonly "orig.arg.no"="1" "unpacked"="1") #1 {
entry:
  %2 = icmp eq [1 x %"struct.ap_fixed<30, 12, AP_TRN, AP_WRAP, 0>"]* %1, null
  br i1 %2, label %ret, label %ret.loopexit

ret.loopexit:                                     ; preds = %entry
  %src.addr.0.0.05 = getelementptr [1 x %"struct.ap_fixed<30, 12, AP_TRN, AP_WRAP, 0>"], [1 x %"struct.ap_fixed<30, 12, AP_TRN, AP_WRAP, 0>"]* %1, i64 0, i64 0, i32 0, i32 0, i32 0
  %3 = bitcast i30* %src.addr.0.0.05 to i32*
  %4 = load i32, i32* %3
  %5 = trunc i32 %4 to i30
  store i30 %5, i30* %0, align 512
  br label %ret

ret:                                              ; preds = %ret.loopexit, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse
define internal void @"onebyonecpy_hls.p0a8struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"([8 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* noalias "orig.arg.no"="0" "unpacked"="0", i16* noalias nocapture readonly align 512 "fpga.caller.interfaces"="layout_transformed" "orig.arg.no"="1" "unpacked"="1.0.0.0.0" %_0, i16* noalias nocapture readonly align 512 "fpga.caller.interfaces"="layout_transformed" "orig.arg.no"="1" "unpacked"="1.0.0.0.1" %_1, i16* noalias nocapture readonly align 512 "fpga.caller.interfaces"="layout_transformed" "orig.arg.no"="1" "unpacked"="1.0.0.0.2" %_2, i16* noalias nocapture readonly align 512 "fpga.caller.interfaces"="layout_transformed" "orig.arg.no"="1" "unpacked"="1.0.0.0.3" %_3, i16* noalias nocapture readonly align 512 "fpga.caller.interfaces"="layout_transformed" "orig.arg.no"="1" "unpacked"="1.0.0.0.4" %_4, i16* noalias nocapture readonly align 512 "fpga.caller.interfaces"="layout_transformed" "orig.arg.no"="1" "unpacked"="1.0.0.0.5" %_5, i16* noalias nocapture readonly align 512 "fpga.caller.interfaces"="layout_transformed" "orig.arg.no"="1" "unpacked"="1.0.0.0.6" %_6, i16* noalias nocapture readonly align 512 "fpga.caller.interfaces"="layout_transformed" "orig.arg.no"="1" "unpacked"="1.0.0.0.7" %_7) #1 {
entry:
  %1 = icmp eq [8 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* %0, null
  br i1 %1, label %ret, label %copy

copy:                                             ; preds = %entry
  br label %for.loop

for.loop:                                         ; preds = %src.addr.0.0.05.exit, %copy
  %for.loop.idx1 = phi i64 [ 0, %copy ], [ %for.loop.idx.next, %src.addr.0.0.05.exit ]
  %dst.addr.0.0.06 = getelementptr [8 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"], [8 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* %0, i64 0, i64 %for.loop.idx1, i32 0, i32 0, i32 0
  %2 = trunc i64 %for.loop.idx1 to i3
  switch i3 %2, label %src.addr.0.0.05.case.7 [
    i3 0, label %src.addr.0.0.05.case.0
    i3 1, label %src.addr.0.0.05.case.1
    i3 2, label %src.addr.0.0.05.case.2
    i3 3, label %src.addr.0.0.05.case.3
    i3 -4, label %src.addr.0.0.05.case.4
    i3 -3, label %src.addr.0.0.05.case.5
    i3 -2, label %src.addr.0.0.05.case.6
  ]

src.addr.0.0.05.case.0:                           ; preds = %for.loop
  %_01 = load i16, i16* %_0, align 512
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.1:                           ; preds = %for.loop
  %_12 = load i16, i16* %_1, align 512
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.2:                           ; preds = %for.loop
  %_23 = load i16, i16* %_2, align 512
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.3:                           ; preds = %for.loop
  %_34 = load i16, i16* %_3, align 512
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.4:                           ; preds = %for.loop
  %_45 = load i16, i16* %_4, align 512
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.5:                           ; preds = %for.loop
  %_56 = load i16, i16* %_5, align 512
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.6:                           ; preds = %for.loop
  %_67 = load i16, i16* %_6, align 512
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.7:                           ; preds = %for.loop
  %_78 = load i16, i16* %_7, align 512
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.exit:                             ; preds = %src.addr.0.0.05.case.7, %src.addr.0.0.05.case.6, %src.addr.0.0.05.case.5, %src.addr.0.0.05.case.4, %src.addr.0.0.05.case.3, %src.addr.0.0.05.case.2, %src.addr.0.0.05.case.1, %src.addr.0.0.05.case.0
  %3 = phi i16 [ %_01, %src.addr.0.0.05.case.0 ], [ %_12, %src.addr.0.0.05.case.1 ], [ %_23, %src.addr.0.0.05.case.2 ], [ %_34, %src.addr.0.0.05.case.3 ], [ %_45, %src.addr.0.0.05.case.4 ], [ %_56, %src.addr.0.0.05.case.5 ], [ %_67, %src.addr.0.0.05.case.6 ], [ %_78, %src.addr.0.0.05.case.7 ]
  store i16 %3, i16* %dst.addr.0.0.06, align 2
  %for.loop.idx.next = add nuw nsw i64 %for.loop.idx1, 1
  %exitcond = icmp ne i64 %for.loop.idx.next, 8
  br i1 %exitcond, label %for.loop, label %ret

ret:                                              ; preds = %src.addr.0.0.05.exit, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse
define internal void @"onebyonecpy_hls.p0a1struct.ap_fixed<30, 12, AP_TRN, AP_WRAP, 0>.362"([1 x %"struct.ap_fixed<30, 12, AP_TRN, AP_WRAP, 0>"]* noalias "orig.arg.no"="0" "unpacked"="0", i30* noalias nocapture readonly align 512 "fpga.caller.interfaces"="layout_transformed" "orig.arg.no"="1" "unpacked"="1.0.0.0") #1 {
entry:
  %2 = icmp eq [1 x %"struct.ap_fixed<30, 12, AP_TRN, AP_WRAP, 0>"]* %0, null
  br i1 %2, label %ret, label %ret.loopexit

ret.loopexit:                                     ; preds = %entry
  %dst.addr.0.0.06 = getelementptr [1 x %"struct.ap_fixed<30, 12, AP_TRN, AP_WRAP, 0>"], [1 x %"struct.ap_fixed<30, 12, AP_TRN, AP_WRAP, 0>"]* %0, i64 0, i64 0, i32 0, i32 0, i32 0
  %3 = bitcast i30* %1 to i32*
  %4 = load i32, i32* %3
  %5 = trunc i32 %4 to i30
  store i30 %5, i30* %dst.addr.0.0.06, align 4
  br label %ret

ret:                                              ; preds = %ret.loopexit, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse
define internal void @"onebyonecpy_hls.p0a320struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"(i5120* noalias nocapture align 512 "fpga.caller.interfaces"="layout_transformed" "orig.arg.no"="0" "unpacked"="0.0.0.0", [320 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* noalias readonly "orig.arg.no"="1" "unpacked"="1") #1 {
entry:
  %2 = icmp eq [320 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* %1, null
  br i1 %2, label %ret, label %copy

copy:                                             ; preds = %entry
  %.promoted = load i5120, i5120* %0, align 512
  br label %for.loop

for.loop:                                         ; preds = %for.loop, %copy
  %3 = phi i5120 [ %.promoted, %copy ], [ %12, %for.loop ]
  %for.loop.idx1 = phi i64 [ 0, %copy ], [ %for.loop.idx.next, %for.loop ]
  %src.addr.0.0.05 = getelementptr [320 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"], [320 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* %1, i64 0, i64 %for.loop.idx1, i32 0, i32 0, i32 0
  %4 = mul nuw nsw i64 16, %for.loop.idx1
  %5 = load i16, i16* %src.addr.0.0.05, align 2
  %6 = zext i64 %4 to i5120
  %7 = shl i5120 65535, %6
  %8 = zext i16 %5 to i5120
  %9 = shl i5120 %8, %6
  %10 = xor i5120 %7, -1
  %11 = and i5120 %3, %10
  %12 = or i5120 %11, %9
  %for.loop.idx.next = add nuw nsw i64 %for.loop.idx1, 1
  %exitcond = icmp ne i64 %for.loop.idx.next, 320
  br i1 %exitcond, label %for.loop, label %ret.loopexit

ret.loopexit:                                     ; preds = %for.loop
  store i5120 %12, i5120* %0, align 512
  br label %ret

ret:                                              ; preds = %ret.loopexit, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse
define internal void @copy_in([320 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* noalias readonly "orig.arg.no"="0" "unpacked"="0", i5120* noalias nocapture align 512 "fpga.caller.interfaces"="layout_transformed" "orig.arg.no"="1" "unpacked"="1.0.0.0", [8 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* noalias readonly "orig.arg.no"="2" "unpacked"="2", i16* noalias nocapture align 512 "fpga.caller.interfaces"="layout_transformed" "orig.arg.no"="3" "unpacked"="3.0.0.0.0" %_0, i16* noalias nocapture align 512 "fpga.caller.interfaces"="layout_transformed" "orig.arg.no"="3" "unpacked"="3.0.0.0.1" %_1, i16* noalias nocapture align 512 "fpga.caller.interfaces"="layout_transformed" "orig.arg.no"="3" "unpacked"="3.0.0.0.2" %_2, i16* noalias nocapture align 512 "fpga.caller.interfaces"="layout_transformed" "orig.arg.no"="3" "unpacked"="3.0.0.0.3" %_3, i16* noalias nocapture align 512 "fpga.caller.interfaces"="layout_transformed" "orig.arg.no"="3" "unpacked"="3.0.0.0.4" %_4, i16* noalias nocapture align 512 "fpga.caller.interfaces"="layout_transformed" "orig.arg.no"="3" "unpacked"="3.0.0.0.5" %_5, i16* noalias nocapture align 512 "fpga.caller.interfaces"="layout_transformed" "orig.arg.no"="3" "unpacked"="3.0.0.0.6" %_6, i16* noalias nocapture align 512 "fpga.caller.interfaces"="layout_transformed" "orig.arg.no"="3" "unpacked"="3.0.0.0.7" %_7, [1 x %"struct.ap_fixed<30, 12, AP_TRN, AP_WRAP, 0>"]* noalias readonly "orig.arg.no"="4" "unpacked"="4", i30* noalias nocapture align 512 "fpga.caller.interfaces"="layout_transformed" "orig.arg.no"="5" "unpacked"="5.0.0.0") #2 {
entry:
  call void @"onebyonecpy_hls.p0a320struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"(i5120* align 512 %1, [320 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* %0)
  call void @"onebyonecpy_hls.p0a8struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>.388"(i16* align 512 %_0, i16* align 512 %_1, i16* align 512 %_2, i16* align 512 %_3, i16* align 512 %_4, i16* align 512 %_5, i16* align 512 %_6, i16* align 512 %_7, [8 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* %2)
  call void @"onebyonecpy_hls.p0a1struct.ap_fixed<30, 12, AP_TRN, AP_WRAP, 0>"(i30* align 512 %4, [1 x %"struct.ap_fixed<30, 12, AP_TRN, AP_WRAP, 0>"]* %3)
  ret void
}

; Function Attrs: argmemonly noinline norecurse
define internal void @"onebyonecpy_hls.p0a320struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>.416"([320 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* noalias "orig.arg.no"="0" "unpacked"="0", i5120* noalias nocapture readonly align 512 "fpga.caller.interfaces"="layout_transformed" "orig.arg.no"="1" "unpacked"="1.0.0.0") #1 {
entry:
  %2 = icmp eq [320 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* %0, null
  br i1 %2, label %ret, label %copy

copy:                                             ; preds = %entry
  %3 = load i5120, i5120* %1, align 512
  br label %for.loop

for.loop:                                         ; preds = %for.loop, %copy
  %for.loop.idx1 = phi i64 [ 0, %copy ], [ %for.loop.idx.next, %for.loop ]
  %4 = mul nuw nsw i64 16, %for.loop.idx1
  %dst.addr.0.0.06 = getelementptr [320 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"], [320 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* %0, i64 0, i64 %for.loop.idx1, i32 0, i32 0, i32 0
  %5 = zext i64 %4 to i5120
  %6 = lshr i5120 %3, %5
  %7 = trunc i5120 %6 to i16
  store i16 %7, i16* %dst.addr.0.0.06, align 2
  %for.loop.idx.next = add nuw nsw i64 %for.loop.idx1, 1
  %exitcond = icmp ne i64 %for.loop.idx.next, 320
  br i1 %exitcond, label %for.loop, label %ret

ret:                                              ; preds = %for.loop, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse
define internal void @copy_out([320 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* noalias "orig.arg.no"="0" "unpacked"="0", i5120* noalias nocapture readonly align 512 "fpga.caller.interfaces"="layout_transformed" "orig.arg.no"="1" "unpacked"="1.0.0.0", [8 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* noalias "orig.arg.no"="2" "unpacked"="2", i16* noalias nocapture readonly align 512 "fpga.caller.interfaces"="layout_transformed" "orig.arg.no"="3" "unpacked"="3.0.0.0.0" %_0, i16* noalias nocapture readonly align 512 "fpga.caller.interfaces"="layout_transformed" "orig.arg.no"="3" "unpacked"="3.0.0.0.1" %_1, i16* noalias nocapture readonly align 512 "fpga.caller.interfaces"="layout_transformed" "orig.arg.no"="3" "unpacked"="3.0.0.0.2" %_2, i16* noalias nocapture readonly align 512 "fpga.caller.interfaces"="layout_transformed" "orig.arg.no"="3" "unpacked"="3.0.0.0.3" %_3, i16* noalias nocapture readonly align 512 "fpga.caller.interfaces"="layout_transformed" "orig.arg.no"="3" "unpacked"="3.0.0.0.4" %_4, i16* noalias nocapture readonly align 512 "fpga.caller.interfaces"="layout_transformed" "orig.arg.no"="3" "unpacked"="3.0.0.0.5" %_5, i16* noalias nocapture readonly align 512 "fpga.caller.interfaces"="layout_transformed" "orig.arg.no"="3" "unpacked"="3.0.0.0.6" %_6, i16* noalias nocapture readonly align 512 "fpga.caller.interfaces"="layout_transformed" "orig.arg.no"="3" "unpacked"="3.0.0.0.7" %_7, [1 x %"struct.ap_fixed<30, 12, AP_TRN, AP_WRAP, 0>"]* noalias "orig.arg.no"="4" "unpacked"="4", i30* noalias nocapture readonly align 512 "fpga.caller.interfaces"="layout_transformed" "orig.arg.no"="5" "unpacked"="5.0.0.0") #3 {
entry:
  call void @"onebyonecpy_hls.p0a320struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>.416"([320 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* %0, i5120* align 512 %1)
  call void @"onebyonecpy_hls.p0a8struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"([8 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* %2, i16* align 512 %_0, i16* align 512 %_1, i16* align 512 %_2, i16* align 512 %_3, i16* align 512 %_4, i16* align 512 %_5, i16* align 512 %_6, i16* align 512 %_7)
  call void @"onebyonecpy_hls.p0a1struct.ap_fixed<30, 12, AP_TRN, AP_WRAP, 0>.362"([1 x %"struct.ap_fixed<30, 12, AP_TRN, AP_WRAP, 0>"]* %3, i30* align 512 %4)
  ret void
}

declare void @apatb_JetTaggerNN_hw(i5120*, i16*, i16*, i16*, i16*, i16*, i16*, i16*, i16*, i30*)

; Function Attrs: argmemonly noinline norecurse
define internal void @copy_back([320 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* noalias "orig.arg.no"="0" "unpacked"="0", i5120* noalias nocapture readonly align 512 "fpga.caller.interfaces"="layout_transformed" "orig.arg.no"="1" "unpacked"="1.0.0.0", [8 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* noalias "orig.arg.no"="2" "unpacked"="2", i16* noalias nocapture readonly align 512 "fpga.caller.interfaces"="layout_transformed" "orig.arg.no"="3" "unpacked"="3.0.0.0.0" %_0, i16* noalias nocapture readonly align 512 "fpga.caller.interfaces"="layout_transformed" "orig.arg.no"="3" "unpacked"="3.0.0.0.1" %_1, i16* noalias nocapture readonly align 512 "fpga.caller.interfaces"="layout_transformed" "orig.arg.no"="3" "unpacked"="3.0.0.0.2" %_2, i16* noalias nocapture readonly align 512 "fpga.caller.interfaces"="layout_transformed" "orig.arg.no"="3" "unpacked"="3.0.0.0.3" %_3, i16* noalias nocapture readonly align 512 "fpga.caller.interfaces"="layout_transformed" "orig.arg.no"="3" "unpacked"="3.0.0.0.4" %_4, i16* noalias nocapture readonly align 512 "fpga.caller.interfaces"="layout_transformed" "orig.arg.no"="3" "unpacked"="3.0.0.0.5" %_5, i16* noalias nocapture readonly align 512 "fpga.caller.interfaces"="layout_transformed" "orig.arg.no"="3" "unpacked"="3.0.0.0.6" %_6, i16* noalias nocapture readonly align 512 "fpga.caller.interfaces"="layout_transformed" "orig.arg.no"="3" "unpacked"="3.0.0.0.7" %_7, [1 x %"struct.ap_fixed<30, 12, AP_TRN, AP_WRAP, 0>"]* noalias "orig.arg.no"="4" "unpacked"="4", i30* noalias nocapture readonly align 512 "fpga.caller.interfaces"="layout_transformed" "orig.arg.no"="5" "unpacked"="5.0.0.0") #3 {
entry:
  call void @"onebyonecpy_hls.p0a8struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"([8 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* %2, i16* align 512 %_0, i16* align 512 %_1, i16* align 512 %_2, i16* align 512 %_3, i16* align 512 %_4, i16* align 512 %_5, i16* align 512 %_6, i16* align 512 %_7)
  call void @"onebyonecpy_hls.p0a1struct.ap_fixed<30, 12, AP_TRN, AP_WRAP, 0>.362"([1 x %"struct.ap_fixed<30, 12, AP_TRN, AP_WRAP, 0>"]* %3, i30* align 512 %4)
  ret void
}

define void @JetTaggerNN_hw_stub_wrapper(i5120*, i16*, i16*, i16*, i16*, i16*, i16*, i16*, i16*, i30*) #4 {
entry:
  %10 = alloca [320 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]
  %11 = alloca [8 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]
  %12 = alloca [1 x %"struct.ap_fixed<30, 12, AP_TRN, AP_WRAP, 0>"]
  call void @copy_out([320 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* %10, i5120* %0, [8 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* %11, i16* %1, i16* %2, i16* %3, i16* %4, i16* %5, i16* %6, i16* %7, i16* %8, [1 x %"struct.ap_fixed<30, 12, AP_TRN, AP_WRAP, 0>"]* %12, i30* %9)
  %13 = bitcast [320 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* %10 to %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"*
  %14 = bitcast [8 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* %11 to %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"*
  %15 = bitcast [1 x %"struct.ap_fixed<30, 12, AP_TRN, AP_WRAP, 0>"]* %12 to %"struct.ap_fixed<30, 12, AP_TRN, AP_WRAP, 0>"*
  call void @JetTaggerNN_hw_stub(%"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"* %13, %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"* %14, %"struct.ap_fixed<30, 12, AP_TRN, AP_WRAP, 0>"* %15)
  call void @copy_in([320 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* %10, i5120* %0, [8 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* %11, i16* %1, i16* %2, i16* %3, i16* %4, i16* %5, i16* %6, i16* %7, i16* %8, [1 x %"struct.ap_fixed<30, 12, AP_TRN, AP_WRAP, 0>"]* %12, i30* %9)
  ret void
}

declare void @JetTaggerNN_hw_stub(%"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"*, %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"*, %"struct.ap_fixed<30, 12, AP_TRN, AP_WRAP, 0>"*)

attributes #0 = { noinline "fpga.wrapper.func"="wrapper" }
attributes #1 = { argmemonly noinline norecurse "fpga.wrapper.func"="onebyonecpy_hls" }
attributes #2 = { argmemonly noinline norecurse "fpga.wrapper.func"="copyin" }
attributes #3 = { argmemonly noinline norecurse "fpga.wrapper.func"="copyout" }
attributes #4 = { "fpga.wrapper.func"="stub" }

!llvm.dbg.cu = !{}
!llvm.ident = !{!0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0}
!llvm.module.flags = !{!1, !2, !3}
!blackbox_cfg = !{!4}
!datalayout.transforms.on.top = !{!5, !19}

!0 = !{!"clang version 7.0.0 "}
!1 = !{i32 2, !"Dwarf Version", i32 4}
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !{i32 1, !"wchar_size", i32 4}
!4 = !{}
!5 = !{!6, !8, !10}
!6 = !{!7}
!7 = !{!"1.0.0.0", [8 x i16]* null}
!8 = !{!9}
!9 = !{!"array_partition", !"type=Complete", !"dim=1"}
!10 = !{!11, !12, !13, !14, !15, !16, !17, !18}
!11 = !{!"1.0.0.0.0", i16* null}
!12 = !{!"1.0.0.0.1", i16* null}
!13 = !{!"1.0.0.0.2", i16* null}
!14 = !{!"1.0.0.0.3", i16* null}
!15 = !{!"1.0.0.0.4", i16* null}
!16 = !{!"1.0.0.0.5", i16* null}
!17 = !{!"1.0.0.0.6", i16* null}
!18 = !{!"1.0.0.0.7", i16* null}
!19 = !{!20, !8, !22}
!20 = !{!21}
!21 = !{!"2.0.0.0", [1 x i30]* null}
!22 = !{!23}
!23 = !{!"2.0.0.0", i30* null}
