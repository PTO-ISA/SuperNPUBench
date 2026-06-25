.section .data
.global _binary_lookup_keys_data_start
_binary_lookup_keys_data_start:
.incbin "hashtable_lookup_simd/data_obj/lookup_keys.data"
.global _binary_lookup_keys_data_end
_binary_lookup_keys_data_end:
.global _binary_lookup_keys_data_size
.equ _binary_lookup_keys_data_size, .-_binary_lookup_keys_data_start
