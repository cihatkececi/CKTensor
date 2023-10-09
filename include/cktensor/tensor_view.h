#pragma once

#include <cassert>

#include <array>
#include <iostream>


namespace ck {


//template<typename T, std::size_t Dims> class Tensor;


class Index {
public:
    Index() = default;

    Index(std::size_t begin, std::size_t end, std::size_t stride = 1) : begin_(begin), end_(end), stride_(stride) {}

    Index(std::size_t i) : Index(i, i + 1) {}

    Index(std::initializer_list<size_t> init_list) : begin_(0), end_(1), stride_(1) {
        assert(init_list.size() >= 1 && init_list.size() <= 3);

        auto it = init_list.begin();
        begin_ = *it;

        end_ = (++it != init_list.end()) ? *it : begin_ + 1;
        stride_ = (++it != init_list.end()) ? *it : 1;
    }

    std::size_t begin() const {
        return begin_;
    }

    std::size_t end() const {
        return end_;
    }

    std::size_t stride() const {
        return stride_;
    }

private:
    std::size_t begin_{};
    std::size_t end_{};
    std::size_t stride_{};
};


template<typename T, typename std::size_t Dims>
class TensorIter {
public:
    using ValueType = T;
    using DifferenceType = std::ptrdiff_t;
    using Reference = T&;
    using Pointer = T*;

    TensorIter() = default;

    TensorIter(T* data, std::array<std::size_t, Dims> shape, std::array<std::size_t, Dims> stride,
               std::array<std::size_t, Dims> index = {})
            : data_{data}, shape_{shape}, index_{index}, stride_{stride} {}

    Reference operator*() {
        return data();
    }

    Pointer operator->() {
        return &data();
    }

    Reference data() {
        return *data_;
    }

    TensorIter& operator++() {
        for (std::size_t axis = Dims - 1; axis != static_cast<std::size_t>(-1); axis--) {
            index_[axis]++;
            if (index_[axis] >= shape_[axis]) [[unlikely]] {
                index_[axis] = 0;
                data_ -= (shape_[axis] - 1) * stride_[axis];
            }
            else {
                data_ += stride_[axis];
                break;
            }
        }

        return *this;
    }

    TensorIter operator++(int) {
        TensorIter temp{*this};

        operator++();

        return temp;
    }

    TensorIter& operator--() {
        for (std::size_t axis = Dims - 1; axis != static_cast<std::size_t>(-1); axis--) {
            if (index_[axis] == 0) [[unlikely]] {
                index_[axis] = shape_[axis] - 1;
                data_ += (stride_[axis] - 1) * shape_[axis];
            }
            else {
                index_[axis]--;
                data_ -= stride_[axis];
                break;
            }
        }

        return *this;
    }

    TensorIter operator--(int) {
        TensorIter temp{*this};

        operator--();

        return temp;
    }

    bool operator==(const TensorIter& other) const {
        return index_ == other.index_;
    }

    bool operator!=(const TensorIter& other) const {
        return index_ != other.index_;
    }

    bool operator<(const TensorIter& other) const {
        return index_ < other.index_;
    }

    bool operator<=(const TensorIter& other) const {
        return index_ <= other.index_;
    }

    bool operator>(const TensorIter& other) const {
        return index_ > other.index_;
    }

    bool operator>=(const TensorIter& other) const {
        return index_ >= other.index_;
    }

private:
    Pointer data_{nullptr};
    std::array<std::size_t, Dims> index_{};
    std::array<std::size_t, Dims> shape_{};
    std::array<std::size_t, Dims> stride_{};
};


template<typename T, std::size_t Dims>
class TensorView {
public:
    using ValueType = T;
    using Iterator = TensorIter<T, Dims>;

    TensorView(T* data, Shape<Dims> shape, std::array<std::size_t, Dims> stride)
            : data_{data}, shape_{shape}, stride_{stride} {}

    TensorView& operator=(const TensorView& other) {

    }

    Iterator begin() {
        return {data_, shape_, stride_};
    }

    Iterator end() {
        std::array<std::size_t, Dims> index = shape_;
        for (std::size_t i = 1; i < Dims; i++)
            index[i]--;

        return {data_, shape_, stride_, index};
    }

private:
    T* data_{nullptr};
    Shape<Dims> shape_{};
    std::array<std::size_t, Dims> stride_{};
};


}
