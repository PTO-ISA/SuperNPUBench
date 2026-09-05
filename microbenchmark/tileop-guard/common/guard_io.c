// Compiled WITHOUT matrix flags (-mlxbc -O2 only) so these loops stay scalar.
// See guard_io.h for the rationale.
#include "guard_io.h"
#include <fcntl.h>
#include <unistd.h>
#include <string.h>

void guard_fill_seq_f32(float* p, int n, float base, float step) {
    for (int i = 0; i < n; ++i) p[i] = base + (float)i * step;
}

void guard_fill_const_f32(float* p, int n, float v) {
    for (int i = 0; i < n; ++i) p[i] = v;
}

void guard_fill_seq_i32(int32_t* p, int n, int32_t base, int32_t step) {
    for (int i = 0; i < n; ++i) p[i] = base + i * step;
}

void guard_fill_seq_f16(uint16_t* p, int n, float base, float step) {
    for (int i = 0; i < n; ++i) {
        __fp16 h = (__fp16)(base + (float)i * step);
        uint16_t bits;
        memcpy(&bits, &h, sizeof(bits));
        p[i] = bits;
    }
}

void guard_dump_bin(const char* path, const void* p, size_t bytes) {
    int fd = open(path, O_CREAT | O_WRONLY | O_TRUNC, 0644);
    if (fd < 0) return;
    (void)write(fd, p, bytes);
    close(fd);
}

void guard_read_bin(const char* path, void* p, size_t bytes) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) return;
    size_t done = 0;
    char* d = (char*)p;
    while (done < bytes) {
        long r = read(fd, d + done, bytes - done);
        if (r <= 0) break;
        done += (size_t)r;
    }
    close(fd);
}
