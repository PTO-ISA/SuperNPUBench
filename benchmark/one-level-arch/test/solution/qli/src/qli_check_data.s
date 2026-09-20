// QLI 输入数据嵌入：由 compile.all 用 gen_qli_golden.py 按当前配置生成到
// compare/current/，再 .incbin 进 .data 段。路径相对 make 的工作目录
// （test/solution/qli，汇编命令 CWD 即此处），不再硬编码任何绝对路径。
.section .data
.global _binary_srcq_data_start
_binary_srcq_data_start:
.incbin "compare/current/srcq.bin"
.global _binary_srcq_data_end
_binary_srcq_data_end:
.global _binary_srcq_data_size
.equ _binary_srcq_data_size, .-_binary_srcq_data_start

.section .data
.global _binary_srck_data_start
_binary_srck_data_start:
.incbin "compare/current/srck.bin"
.global _binary_srck_data_end
_binary_srck_data_end:
.global _binary_srck_data_size
.equ _binary_srck_data_size, .-_binary_srck_data_start

.section .data
.global _binary_srcw_data_start
_binary_srcw_data_start:
.incbin "compare/current/srcw.bin"
.global _binary_srcw_data_end
_binary_srcw_data_end:
.global _binary_srcw_data_size
.equ _binary_srcw_data_size, .-_binary_srcw_data_start

.section .data
.global _binary_srcsq_data_start
_binary_srcsq_data_start:
.incbin "compare/current/srcsq.bin"
.global _binary_srcsq_data_end
_binary_srcsq_data_end:
.global _binary_srcsq_data_size
.equ _binary_srcsq_data_size, .-_binary_srcsq_data_start

.section .data
.global _binary_srcsk_data_start
_binary_srcsk_data_start:
.incbin "compare/current/srcsk.bin"
.global _binary_srcsk_data_end
_binary_srcsk_data_end:
.global _binary_srcsk_data_size
.equ _binary_srcsk_data_size, .-_binary_srcsk_data_start
