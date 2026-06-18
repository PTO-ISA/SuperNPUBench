#ifndef TENSOR_H
#define TENSOR_H

#include <array>
#include <vector>
#include <numeric>
#include <cassert>

#include <array>
#include <vector>
#include <iostream>
#include <cassert>

template<typename T, size_t... Dims>
class Tensor {
public:
    static constexpr size_t NumDims = sizeof...(Dims);
    static constexpr std::array<size_t, NumDims> shape = {Dims...};
    static constexpr size_t total_size = (Dims * ...);  // C++17 fold expression
    static_assert(total_size > 0, "zero size!");
    // 默认构造：初始化所有元素为零
    Tensor() {
        //std::fill(data_, data_ + total_size, T{});
        std::fill(data_, data_ + total_size, static_cast<T>(1));
    }

    // 构造并初始化所有元素为指定值
    Tensor(const T& init_val) {
        std::fill(data_, data_ + total_size, init_val);
    }

    // 获取维度数量
    constexpr size_t dims() const { return NumDims; }

    // 获取每个维度的大小
    constexpr size_t shape_at(size_t dim) const {
        assert(dim < NumDims);
        return shape[dim];
    }

    // 获取总元素数量
    constexpr size_t size() const { return total_size; }

    // 获取原始数据指针
    template<typename... Indexes>
    T* data(Indexes... rest) { 
        static_assert(sizeof...(rest) < NumDims, "Too many indices for tensor");
        size_t indexes[] = {static_cast<size_t>(rest)...};
        size_t index = 0;
        size_t stride = total_size;
        size_t size = sizeof(indexes)/sizeof(indexes[0]);
        for (size_t i = 0; i < size; ++i) {
            stride /= shape[i];
            index += indexes[i] * stride;
        }
        return &data_[index];
    }

    template<typename... Indexes> 
    const T* data(Indexes... rest) const { 
        static_assert(sizeof...(rest) < NumDims, "Too many indices for tensor");
        size_t indexes[] = {static_cast<size_t>(rest)...};
        size_t index = 0;
        size_t stride = total_size;
        size_t size = sizeof(indexes)/sizeof(indexes[0]);
        for (size_t i = 0; i < size; ++i) {
            stride /= shape[i];
            index += indexes[i] * stride;
        }
        return &data_[index];
    }

    // 单维索引访问接口
    T& operator[](size_t index) {
        assert(index < total_size);
        return data_[index];
    }

    const T& operator[](size_t index) const {
        assert(index < total_size);
        return data_[index];
    }

    // 多维索引访问接口
    template<typename... Indexes>
    T& operator()(size_t first, Indexes... rest) {
        static_assert(sizeof...(rest) < NumDims, "Too many indices for tensor");
        size_t indexes[] = {first, static_cast<size_t>(rest)...};
        size_t index = 0;
        size_t stride = total_size;
        size_t size = sizeof(indexes)/sizeof(indexes[0]);
        for (size_t i = 0; i < size; ++i) {
            stride /= shape[i];
            index += indexes[i] * stride;
        }
        return data_[index];
    }

    template<typename... Indexes>
    const T& operator()(size_t first, Indexes... rest) const {
        static_assert(sizeof...(rest) < NumDims, "Too many indices for tensor");
        size_t indexes[] = {first, static_cast<size_t>(rest)...};
        size_t index = 0;
        size_t stride = total_size;
        size_t size = sizeof(indexes)/sizeof(indexes[0]);
        for (size_t i = 0; i < size; ++i) {
            stride /= shape[i];
            index += indexes[i] * stride;
        }
        return data_[index];
    }

private:
    T data_[total_size] = {};  // 使用多维数组
};

#endif