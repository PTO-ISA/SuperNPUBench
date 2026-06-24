#ifndef BENCHMARK_HH
#define BENCHMARK_HH

#ifdef __linx
#define BENCHSTART  __asm__ __volatile__("B.HINT TRACE.begin\n" : : :);
#else
#define BENCHSTART 
#endif

#ifdef __linx
#define BENCHEND   __asm__ __volatile__("B.HINT TRACE.end\n" : : :);
#else
#define BENCHEND
#endif

#endif