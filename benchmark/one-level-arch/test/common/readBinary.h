#ifndef READ_BINARY_HPP
#define READ_BINARY_HPP

#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <iostream>
#include <string>

// #define ENABLE_BINARY_OUTPUT

bool readBinaryFile(const char * filename, uint8_t* data, size_t size) {
#ifdef ENABLE_BINARY_OUTPUT
    int fd = open(filename, O_CREAT | O_RDWR, 0644);
    if (fd <0) {
        fprintf(stderr, "faild\n");
        return false;
    }

    int readed = read(fd, data, size);
    if (readed != size) {
        fprintf(stderr, "faild\n");
        return false;
    }

    close(fd);
    // gfrun 规避:RES_CHECK 精度流程在 gfrun 下跑时,向 stdout 的 printf 会走 musl
    // buffered writev,其 iov 落到 gfrun 未映射的高栈地址 → writev 死循环挂起
    // (见 ISSUE_gfrun_res_check_writev_hang.md)。此处仅状态打印,静音不影响正确性
    // (脚本按 output.bin + returncode 判定)。手动 ENABLE_BINARY_OUTPUT 调试仍保留。
#ifndef RES_CHECK
    printf("data read to file done: %s\n", filename);
    fflush(stdout);
#endif
    return true;
#else
    return true;
#endif
}

#endif