#ifndef TLSU_FINISH_H
#define TLSU_FINISH_H

// TLSU 端到端用例的收尾与结果区约定。
//
// test finisher：两个模型的判定条件不同，取交集才能同时满足 ——
//   gfsim  SimSys.cpp:530-534  按字节累积后取低 16 位 == 0x5555
//   gfrun  Memory.cpp:345-348  要求 width == 4 且 data == 0x5555
// 交集就是一条 32 位 store、值恰为 0x5555。这条 store 不落内存（两侧都在
// 写内存前拦截返回），所以 finisher 地址不能算进结果缓冲。
static inline void tlsu_finish(unsigned int ok)
{
    *(volatile unsigned int *)0x10009000UL = ok ? 0x5555u : 0xDEADu;
}

// 结果缓冲 + 尺寸符号。
//
// cross_model_result_size 必须是绝对符号：run_diff.py:632 取的是符号的
// st_value 而非内容，C 表达不出来。这里用文件作用域内联汇编代替独立的 .s，
// 省去改 Makefile.common 的链接对象列表。
//
// 缓冲放 .bss 即可：ELF.cpp:325-355 按 section 遍历，凡 SHF_ALLOC && sh_size>0
// 一律建 bank，只有 SHT_PROGBITS 才灌内容；.bss 是 SHT_NOBITS，bank 建好、
// 内容为零，正合需要。写入不受保护 —— 只有 SHF_EXECINSTR 的段才进 text_region。
// 注意 TLSU_STR 的两级展开：直接用 #sz 会把宏参数原样字符串化（得到
// "RESULT_SIZE" 而不是 "4096"），汇编器把它当未定义符号，结果符号表里
// 根本没有 cross_model_result_size。
#define TLSU_STR_(x) #x
#define TLSU_STR(x) TLSU_STR_(x)

#define TLSU_RESULT_BUFFER(sz)                           \
    extern "C" { unsigned char cross_model_result[sz]; } \
    __asm__(".globl cross_model_result_size\n"           \
            ".set cross_model_result_size, " TLSU_STR(sz) "\n")

#endif
