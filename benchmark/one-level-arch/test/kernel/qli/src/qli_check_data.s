.section .data
.global _binary_srcq_data_start
_binary_srcq_data_start:
.incbin "/home/z00947698/superScalar/SuperNPUBench/benchmark/one-level-arch/test/kernel/qli/compare/sq4_skv128_topk8/srcq.bin"
.global _binary_srcq_data_end
_binary_srcq_data_end:
.global _binary_srcq_data_size
.equ _binary_srcq_data_size, .-_binary_srcq_data_start

.section .data
.global _binary_srck_data_start
_binary_srck_data_start:
.incbin "/home/z00947698/superScalar/SuperNPUBench/benchmark/one-level-arch/test/kernel/qli/compare/sq4_skv128_topk8/srck.bin"
.global _binary_srck_data_end
_binary_srck_data_end:
.global _binary_srck_data_size
.equ _binary_srck_data_size, .-_binary_srck_data_start

.section .data
.global _binary_srcw_data_start
_binary_srcw_data_start:
.incbin "/home/z00947698/superScalar/SuperNPUBench/benchmark/one-level-arch/test/kernel/qli/compare/sq4_skv128_topk8/srcw.bin"
.global _binary_srcw_data_end
_binary_srcw_data_end:
.global _binary_srcw_data_size
.equ _binary_srcw_data_size, .-_binary_srcw_data_start

.section .data
.global _binary_srcsq_data_start
_binary_srcsq_data_start:
.incbin "/home/z00947698/superScalar/SuperNPUBench/benchmark/one-level-arch/test/kernel/qli/compare/sq4_skv128_topk8/srcsq.bin"
.global _binary_srcsq_data_end
_binary_srcsq_data_end:
.global _binary_srcsq_data_size
.equ _binary_srcsq_data_size, .-_binary_srcq_data_start

.section .data
.global _binary_srcsk_data_start
_binary_srcsk_data_start:
.incbin "/home/z00947698/superScalar/SuperNPUBench/benchmark/one-level-arch/test/kernel/qli/compare/sq4_skv128_topk8/srcsk.bin"
.global _binary_srcsk_data_end
_binary_srcsk_data_end:
.global _binary_srcsk_data_size
.equ _binary_srcsk_data_size, .-_binary_srcsk_data_start
