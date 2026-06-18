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
    printf("data read to file done: %s\n", filename);
    fflush(stdout);
    return true;
#else
    return true;
#endif
}

#endif