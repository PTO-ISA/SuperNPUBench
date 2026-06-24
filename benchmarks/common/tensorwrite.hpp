#ifndef TENSOR_WRITE_HPP
#define TENSOR_WRITE_HPP

#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>

template<typename... Args>
std::string fstring(Args&&... args) {
    std::ostringstream oss;
    (oss << ... << args);  // C++17 fold expression
    return oss.str();
}

// 写入数据到文件的函数
template <typename T, int R, int C>
void writeTensorToFile(const T* data, const std::string& filename) {
    std::ofstream outFile(filename);
    if (!outFile.is_open()) {
        std::cerr << "无法打开文件: " << filename << std::endl;
        return;
    }

    for (int i = 0; i < R; ++i) {
        for (int j = 0; j < C; ++j) {
            outFile << std::fixed << std::setprecision(8) << data[i * C + j] << " ";
        }
        outFile << std::endl;
    }

    outFile.close();
}

#endif
