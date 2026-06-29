#ifndef COMMON_HH
#define COMMON_HH

#ifdef __baremetal
extern void *mmalloc(int nbytes);
#define malloc(size) mmalloc(size)
#endif

#endif


