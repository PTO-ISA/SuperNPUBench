.section .v41.input,"aw",@progbits
.balign 64
.globl v41_input
.type v41_input,@object
v41_input:
.incbin "input.bin"
.size v41_input, .-v41_input
.section .note.GNU-stack,"",@progbits
