#ifndef LINX_START_END_H
#define LINX_START_END_H

#define PMC_START() __asm__ __volatile__("B.HINT TRACE.begin\n" : : :);

#define PMC_END() __asm__ __volatile__("B.HINT TRACE.end\n" : : :);

#endif