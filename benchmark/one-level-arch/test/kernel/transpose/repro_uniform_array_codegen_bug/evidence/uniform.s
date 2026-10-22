	.text
	.file	"uniform_const_store.cpp"
	.section	.text.main,"ax",@progbits
	.globl	main                            // -- Begin function main
	.p2align	1
	.type	main,@function
main:                                   // @main
.LBB0_1:                                // %entry
	C.BSTART.STD	RET
	c.movr	zero, 	->a0
.Ltmp0:
	addtpc	%tpcrel_hi(sink), 	->t
	addi	t#1, %tpcrel_lo(.Ltmp0), 	->t
	c.movi	8, 	->t
	swi.u	t#1, [t#3, %tpcrel_lo(.Ltmp0)]
	c.swi	t#1, [t#2, 4]
	c.swi	t#1, [t#2, 8]
	c.swi	t#1, [t#2, 12]
	c.setc.tgt	ra
	C.BSTART.STD	
.Lfunc_end0:
	.size	main, .Lfunc_end0-main
                                        // -- End function
	.ident	"clang version 15.0.4 (linx64v5-musl-local 7a120731d46c3a35ce0a174586404a60302c74e8)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym sink
