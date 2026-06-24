.section .data
.global _binary_input_131072_data_start
_binary_input_131072_data_start:
.incbin "topk/data_obj/input_131072.data"
.global _binary_input_131072_data_end
_binary_input_131072_data_end:
.global _binary_input_131072_data_size
.equ _binary_input_131072_data_size, .-_binary_input_131072_data_start
