#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#ifdef SVE
#include <time.h>
typedef uint64_t uint64_t;
#else
#include <sys/time.h>
#include <sys/times.h>
#include <sys/stat.h>
#include <unistd.h>
#endif
#include "sys-sections.h"
#include "chip_def.h"

extern char __bss_start__[], __bss_end__[];
extern void __libc_init_array(void);
extern void __libc_fini_array(void);

BOOT_BSS(char dbg_buffer[256]);
#define dbg_len sizeof(dbg_buffer)

// Dummy function to force the obansw builds to link despite wanting RDI syscalls
BOOT_TEXT(void initialise_monitor_handles(void));
void initialise_monitor_handles(void) { }

BOOT_TEXT(int _close(int fh));
int _close(int fh) { }
BOOT_TEXT(int _gettimeofday(struct timeval* tv, void* tz));
int _gettimeofday(struct timeval* tv,void* tz) { }
BOOT_TEXT(int _isatty(int fh));
int _isatty(int fh) { }
BOOT_TEXT(int _lseek(int fh, uint64_t offset, int whence));
int _lseek(int fh,uint64_t offset, int whence) { }
BOOT_TEXT(int _open(const char *path, int oflag, /* int mode */...));
int _open(const char *path, int oflag, /* int mode */...) { return 1; }
BOOT_TEXT(int _read(int fh, unsigned char *buffer, int count));
int _read(int fh, unsigned char *buffer, int count) { }
BOOT_TEXT(int _rename(const char *old, const char *new_name ));
int _rename(const char *old, const char *new_name ) { }
BOOT_TEXT(clock_t _times(struct tms *buf));
BOOT_TEXT(int _unlink(const char *name));
int _unlink(const char *name) { }
BOOT_TEXT(void _exit(int c) __attribute__ ((noreturn)));
void _exit(int c) {}
BOOT_TEXT(int _kill(int pid, int sig));
int _kill(int pid, int sig) { }
BOOT_TEXT(int _fstat(int fildes, struct stat *buf));
int _fstat(int fildes, struct stat *buf) { }
int _getpid() {}

BOOT_TEXT(int write(int fh, char *buf, int count));
int write(int fh, char *buf, int count){
    int todo;
    volatile char* tube = (char *)TUBE_ADDRESS;
    for(todo=0; todo<count; todo++) {
        *tube = *buf++;
    }
    return count;
}

extern char bottom_of_heap;
extern uintptr_t sys_heap_limit(void);

BOOT_TEXT(void *_sbrk(int nbytes));
void* _sbrk(int nbytes)
{
    BOOT_DATA(static char *heap_end) = &bottom_of_heap;
    BOOT_DATA(static char* heap_limit) = NULL;    
    char *prev_heap_end;    
    
    if (heap_limit == NULL)
        heap_limit = (char*) sys_heap_limit();

    prev_heap_end = heap_end;
    heap_end += nbytes;    
    
    if (nbytes != 0 && heap_end >= heap_limit) {
#ifndef SVE        
        snprintf(dbg_buffer, dbg_len, "ERROR: _sbrk(%d) causes heap overflow (previous heap end %p, heap limit is %p)\n",
                 nbytes, prev_heap_end, heap_limit);
        errno = ENOMEM;
#endif
        return (void*) -1;
    }

    return (void *) prev_heap_end;
}

BOOT_TEXT(void *mmalloc(int nbytes));
void *mmalloc(int nbytes)
{
    return _sbrk(nbytes);
}

BOOT_TEXT(void init_libc(void));
void init_libc(void){    
    // Zero the BSS
    // size_t bss_size = __bss_end__ - __bss_start__;
    // memset(__bss_start__, 0, bss_size);    
    // atexit(__libc_fini_array);    
    // __libc_init_array();
}

extern int main(int argc, char **argv);

extern "C" int _linx_start();
BOOT_TEXT(int _linx_start());
int _linx_start()
{    
    // exit(main(0, NULL));
    main(0, NULL);
    return 0;
}