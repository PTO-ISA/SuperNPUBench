#ifndef WRITE_BINARY_HPP
#define WRITE_BINARY_HPP

#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>

// #define ENABLE_BINARY_OUTPUT

bool writeBinaryFile(const char * filename, const uint8_t* data, size_t size) {
#ifdef ENABLE_BINARY_OUTPUT
    int fd = open(filename, O_CREAT | O_RDWR | O_TRUNC, 0644);
    if (fd <0) {
        fprintf(stderr, "faild\n");
        return false;
    }

    int written = write(fd, data, size);
    if (written != size) {
        fprintf(stderr, "faild\n");
        return false;
    }

    close(fd);
    // gfrun 规避:见 readBinary.h / ISSUE_gfrun_res_check_writev_hang.md。RES_CHECK 下
    // 向 stdout 的 printf 会触发 gfrun writev 死循环(iov 落未映射高栈地址),静音状态打印。
#ifndef RES_CHECK
    printf("data write to file done: %s\n", filename);
    fflush(stdout);
#endif
    return true;
#else
    return true;
#endif
}

#endif