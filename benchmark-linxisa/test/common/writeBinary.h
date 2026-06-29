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
    int fd = open(filename, O_CREAT | O_RDWR, 0644);
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
    printf("data write to file done: %s\n", filename);
    fflush(stdout);
    return true;
#else
    return true;
#endif
}

#endif