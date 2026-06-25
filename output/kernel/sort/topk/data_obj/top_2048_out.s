.section .data
.global _binary_top_2048_out_data_start
_binary_top_2048_out_data_start:
.incbin "topk/data_obj/top_2048_out.data"
.global _binary_top_2048_out_data_end
_binary_top_2048_out_data_end:
.global _binary_top_2048_out_data_size
.equ _binary_top_2048_out_data_size, .-_binary_top_2048_out_data_start
