	.text
	.file	"uniform_const_store.cpp"
	.section	.text.main,"ax",@progbits
	.globl	main                            // -- Begin function main
	.p2align	1
	.type	main,@function
main:                                   // @main
.LBB0_5:                                // %entry
	FENTRY	[ra ~ s0], sp!, 256
.LBB0_8:                                // %entry
.LBB0_6:                                // %entry
	C.BSTART.STD	
	addi	sp, 256, 	->s0
.LBB0_0:                                // %entry
	c.movr	zero, 	->a0
	swi	a0, [s0, -24]
	c.movi	8, 	->a1
	hl.bfi	a1, a1, 32, 64, 	->a1
	sdi	a1, [s0, -32]
	sdi	a1, [s0, -40]
	swi	a0, [s0, -48]
.LBB0_1:                                // %for.cond
                                        // =>This Inner Loop Header: Depth=1
	L.BSTART.STD	COND, .LBB0_4
	lwi	[s0, -48], 	->a1
	c.movi	3, 	->a0
	setc.lt	a0, a1
.LBB0_2:                                // %for.body
                                        //   in Loop: Header=BB0_1 Depth=1
	C.BSTART.STD	
	lwi	[s0, -48], 	->a2
	subi	s0, 40, 	->a0
	lw	[a0, a2<<2], 	->a0
.Ltmp0:
	addtpc	%tpcrel_hi(sink), 	->a1
	addi	a1, %tpcrel_lo(.Ltmp0), 	->a1
	sw	a0, [a1, a2<<2]
.LBB0_3:                                // %for.inc
                                        //   in Loop: Header=BB0_1 Depth=1
	L.BSTART.STD	DIRECT, .LBB0_1
	lwi	[s0, -48], 	->a0
	addi	a0, 1, 	->a0
	swi	a0, [s0, -48]
.LBB0_4:                                // %for.end
	C.BSTART.STD	
	c.movr	zero, 	->a0
                                        // implicit-def: $r10
                                        // implicit-def: $r11
.LBB0_7:                                // %for.end
	FRET.STK	[ra ~ s0], sp!, 256
	C.BSTART.STD	
.Lfunc_end0:
	.size	main, .Lfunc_end0-main
                                        // -- End function
	.type	.L__const.main.s,@object        // @__const.main.s
	.section	.rodata.cst16,"aM",@progbits,16
	.p2align	2
.L__const.main.s:
	.word	8                               // 0x8
	.word	8                               // 0x8
	.word	8                               // 0x8
	.word	8                               // 0x8
	.size	.L__const.main.s, 16

	.ident	"clang version 15.0.4 (linx64v5-musl-local 7a120731d46c3a35ce0a174586404a60302c74e8)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym sink
