#pragma once

#include <ostream>
#include <array>
#include <vector>
#include <deque>
#include <list>
#include <tuple>
#include <functional>
#include <algorithm>
#include <numeric>

#include "allocator.h"
#include "util.h"
#include "traits.h"
#include "tensor_view.h"


namespace ck {

template<std::size_t Dims>
struct Shape {
    constexpr const std::size_t& operator[](const std::size_t i) const {
        return shape_[i];
    }

    constexpr std::size_t& operator[](const std::size_t i) {
        return shape_[i];
    }

    constexpr std::size_t& at(const std::size_t i) {
        return shape_[i];
    }

    constexpr const std::size_t& at(const std::size_t i) const {
        return shape_[i];
    }

    constexpr std::size_t& rat(const std::size_t i) {
        return shape_[Dims - i - 1];
    }

    constexpr const std::size_t& rat(const std::size_t i) const {
        return shape_[Dims - i - 1];
    }

    constexpr bool operator==(const Shape& other) const {
        return std::equal(shape_.begin(), shape_.end(), other.shape_.begin());
    }

    constexpr std::size_t dims() const {
        return Dims;
    }

    constexpr std::size_t num_elem() const {
        std::size_t num_elem = 1;
        for (std::size_t dim: shape_)
            num_elem *= dim;

        return num_elem;
    }

    constexpr std::size_t* data() {
        return shape_.data();
    }

    constexpr const std::size_t* data() const {
        return shape_.data();
    }

    constexpr auto begin() const {
        return shape_.begin();
    }

    constexpr auto begin() {
        return shape_.begin();
    }

    constexpr auto end() const {
        return shape_.end();
    }

    constexpr auto end() {
        return shape_.end();
    }

    std::array<std::size_t, Dims> shape_{};
};

template<std::size_t Dims>
std::ostream& operator<<(std::ostream& os, const Shape<Dims>& shape) {
    os << '<';
    for (const auto& dim: shape.shape_)
        os << dim << ',';
    os << '>';
    return os;
}

template<>
struct Shape<0> {
    constexpr std::size_t operator[](const std::size_t i) const {
        assert(i == 0 && "Scalar does not have dimensions");
        return 0;
    }

    constexpr bool operator==(const Shape&) const {
        return true;
    }

    constexpr std::size_t dims() const {
        return 0;
    }

    constexpr std::size_t num_elem() const {
        return 1;
    }
};

template<typename T, std::size_t Dims>
class Tensor {
public:
    using ValueType = T;
    using Reference = T&;
    using ConstReference = const T&;
//    using Iterator = TensorIterator<T>;
//    using ConstIterator = ConstTensorIterator<T>;
    using DifferenceType = std::ptrdiff_t;
    using SizeType = std::size_t;
    using AllocatorType = TensorAllocator<T>;

    /* Constructors */
    Tensor() = default;

//    Tensor(std::initializer_list<std::size_t> shape)
//            : shape_{shape}, capacity_{shape_.num_elem()}, data_{allocator_.allocate(capacity_)} {}

    Tensor(Shape<Dims> shape)
            : shape_{shape}, capacity_{shape.num_elem()}, data_{allocator_.allocate(capacity_)} {}

    Tensor(Shape<Dims> shape, const T& val) : Tensor(shape) {
        std::fill(begin(), end(), val);
    }

    // TODO: Find a solution for nesting initializer lists
    Tensor(std::initializer_list<T> vals) : Tensor(TensorShapeFromSTL<T, Dims>{}(vals)) {
        std::copy(vals.begin(), vals.end(), data_);
    }

    // TODO: There is an ambiguity when trying to construct 2x1 matrix. It confuses with the previous constructor
    Tensor(std::initializer_list<std::initializer_list<T>> vals) : Tensor(TensorShapeFromSTL<T, Dims>{}(vals)) {
        recursive_copy(vals.begin(), vals.end(), data_);
    }

    Tensor(std::initializer_list<std::initializer_list<std::initializer_list<T>>> vals)
            : Tensor(TensorShapeFromSTL<T, Dims>{}(vals)) {
        recursive_copy(vals.begin(), vals.end(), data_);
    }

    Tensor(std::initializer_list<std::initializer_list<std::initializer_list<std::initializer_list<T>>>> vals)
            : Tensor(TensorShapeFromSTL<T, Dims>{}(vals)) {
        recursive_copy(vals.begin(), vals.end(), data_);
    }

    Tensor(const Tensor& other) : Tensor{other.shape_} {
        std::copy(other.begin(), other.end(), begin());
    }

    Tensor(Tensor&& other) noexcept: shape_{std::exchange(other.shape_, {})},
                                     capacity_(std::exchange(other.capacity_, {})),
                                     data_(std::exchange(other.data_, nullptr)) {}

    Tensor& operator=(Tensor other) {
        swap(*this, other);
        return *this;
    }

    ~Tensor() {
        allocator_.deallocate(data_, capacity_);
    }

    template<typename U, typename = std::enable_if_t<std::is_convertible_v<U, T>>>
    Tensor(const Tensor<U, Dims>& other) : Tensor{other.shape()} {
        std::copy(other.begin(), other.end(), begin());
    }

    friend void swap(Tensor& lhs, Tensor& rhs) {
        using std::swap;
        swap(lhs.shape_, rhs.shape_);
        swap(lhs.capacity_, rhs.capacity_);
        swap(lhs.data_, rhs.data_);
    }

    /* STL container constructors */
    // TODO: Check if the container is rectangular.
    // TODO: Add a range constructor for iterators as well as the ranges library.

    template<typename U, size_t Size>
    Tensor(const std::array<U, Size>& arr) : Tensor(TensorShapeFromSTL<T, Dims>{}(arr)) {
        recursive_copy(arr.begin(), arr.end(), data_);
    }

    template<typename U>
    Tensor(const std::vector<U>& vec) : Tensor(TensorShapeFromSTL<T, Dims>{}(vec)) {
        recursive_copy(vec.begin(), vec.end(), data_);
    }

    template<typename U>
    Tensor(const std::deque<U>& deq) : Tensor(TensorShapeFromSTL<T, Dims>{}(deq)) {
        recursive_copy(deq.begin(), deq.end(), data_);
    }

    template<typename U>
    Tensor(const std::list<U>& list) : Tensor(TensorShapeFromSTL<T, Dims>{}(list)) {
        recursive_copy(list.begin(), list.end(), data_);
    }

    template<typename... Args>
    Tensor(const std::tuple<Args...>& tup) : Tensor(TensorShapeFromSTL<T, Dims>{}(tup)) {
        recursive_copy(tup.begin(), tup.end(), data_);
    }

    /* Type-casting */

    template<typename U, typename = std::enable_if_t<std::is_convertible_v<T, U>>>
    Tensor<U, Dims> as() const {
        return *this;
    }

    /* Accessors */

    T& at(std::size_t i) requires (Dims == 1) {
        return data()[i];
    }

    const T& at(std::size_t i) const requires (Dims == 1) {
        return data()[i];
    }

    T& operator()(std::size_t i) requires (Dims == 1) {
        return data()[i];
    }

    const T& operator()(std::size_t i) const requires (Dims == 1) {
        return data()[i];
    }

    T& operator[](std::size_t i) requires (Dims == 1) {
        return data()[i];
    }

    const T& operator[](std::size_t i) const requires (Dims == 1) {
        return data()[i];
    }

    T& at(std::size_t i, std::size_t j) requires (Dims == 2) {
        return data()[i * shape_at(1) + j];
    }

    const T& at(std::size_t i, std::size_t j) const requires (Dims == 2) {
        return data()[i * shape_at(1) + j];
    }

    T& operator()(std::size_t i, std::size_t j) requires (Dims == 2) {
        return data()[i * shape_at(1) + j];
    }

    const T& operator()(std::size_t i, std::size_t j) const requires (Dims == 2) {
        return data()[i * shape_at(1) + j];
    }

//    T& operator[](std::size_t i, std::size_t j) requires (Dims == 2) {
//        return data()[i * shape_at(1) + j];
//    }
//
//    const T& operator[](std::size_t i, std::size_t j) const requires (Dims == 2) {
//        return data()[i * shape_at(1) + j];
//    }

    // TODO: Add other at versions and operator() and operator[] as well as the const versions
    // TODO: Add for higher dimensions (at least to 4-D)

//    template<typename... Indices, typename std::enable_if_t<((std::is_same_v<Indices, Index>) || ...)>>
    template<typename... Indices>
    TensorView<T, Dims> operator[](Indices... indices) {
        // TODO: Implement indexing operator
        std::array<std::size_t, Dims> begins{indices.begin()...};
        std::size_t offset = begins[0];
        for (std::size_t i = 1; i < Dims; i++) {
            offset *= *shape_[i];
            offset += begins[i];
        }
        T* data = data() + offset;

        std::array<std::size_t, Dims> stride{indices.stride()...};
        for (std::size_t i = 0; i < Dims; i++) {
            stride[i] *= shape_[i];
        }

        return {data, {((indices.end() - indices.begin()) / indices.stride())...}, stride};
    }

    const Tensor& slice(std::size_t begin, std::size_t end, std::size_t stride = 1) const {
        // TODO: Implement slice operator
        return {};
    }

    /* Modifications */

    void reserve(size_t n) {
        if (capacity_ < n) {
            T* new_data = allocator_.allocate(n);
            std::move(begin(), end(), new_data);

            allocator_.deallocate(data_, capacity_);
            data_ = new_data;
            capacity_ = n;
        }
    }

    void fill(const T& value) {
        std::fill(begin(), end(), value);
    }

    void inp_reshape(Shape<Dims> new_shape) {
        assert(new_shape.num_elem() == shape_.num_elem());
        shape(new_shape);
    }

    /* Copying transformations */

    template<std::size_t TargetDims>
    Tensor<T, TargetDims> reshape(Shape<TargetDims> new_shape) {
        assert(new_shape.num_elem() == shape_.num_elem());

        Tensor<T, TargetDims> ret = *this;
        ret.shape_ = new_shape;
        return ret;
    }

    // TODO: Generalize transpose function for n-dimensions
    Tensor transpose() {
        static_assert(Dims == 2, "Transpose function is currently implemented only for 2-d tensors");
        Tensor ret{Shape<2>{{shape_at(1), shape_at(0)}}};

        for (std::size_t i = 0; i < shape_at(0); i++) {
            for (std::size_t j = 0; j < shape_at(1); j++) {
                ret.at(j, i) = at(i, j);
            }
        }

        return ret;
    }

    template<class TReturn>
    Tensor<TReturn, Dims> map(std::function<TReturn(T)> function) const {
        Tensor<TReturn, Dims> result{shape()};
        for (std::size_t i = 0; i < num_elem(); i++) {
            result.data_[i] = function(data_[i]);
        }
        return result;
    }

    std::size_t count_nonzero() {
        std::size_t count = 0;
        for (std::size_t i = 0; i < num_elem(); i++) {
            if (data()[i] != T{0})
                ++count;
        }
        return count;
    }

    /* Unary operators */

    Tensor operator!() const;

    Tensor operator-() const;

    /* Inplace operators */

    Tensor& operator+=(const Tensor& other);

    Tensor& operator-=(const Tensor& other);

    Tensor& operator*=(const Tensor& other);

    Tensor& operator/=(const Tensor& other);

    Tensor& operator%=(const Tensor& other);

    /* Binary operators */

    /* Read-only Properties */

    T* data() {
        return data_;
    }

    const T* data() const {
        return data_;
    }

    std::size_t capacity() const {
        return capacity_;
    }

    std::size_t num_elem() const {
        return shape_.num_elem();
    }

    std::size_t size() const {
        return shape_.num_elem();
    }

    std::size_t num_rows() const requires (Dims == 2) {
        return shape_at(0);
    }

    std::size_t num_cols() const requires (Dims == 2) {
        return shape_at(1);
    }

    const Shape<Dims>& shape() const {
        return shape_;
    }

    std::size_t shape_at(std::size_t i) const {
        return shape_[i];
    }

    bool is_equal_shape(const Tensor& other) const {
        return shape() == other.shape();
    }

    T* begin() {
        return data_;
    }

    const T* begin() const {
        return data_;
    }

    T* end() {
        return data_ + num_elem();
    }

    const T* end() const {
        return data_ + num_elem();
    }

//    friend class TensorView<T>;

private:
    Shape<Dims> shape_{};
    std::size_t capacity_{0};
    AllocatorType allocator_{};
    T* data_{nullptr};

};


template<typename T, std::size_t Dims>
Tensor<T, Dims> zeros(Shape<Dims> shape) {
    Tensor<T, Dims> t{shape};
    t.fill(T{0});
    return t;
}

template<typename T, std::size_t Dims>
Tensor<T, Dims> ones(Shape<Dims> shape) {
    Tensor<T, Dims> t{shape};
    t.fill(T{1});
    return t;
}

template<typename T>
Tensor<T, 1> range(T stop) {
    if (stop <= 0)
        return {};

    Tensor<T, 1> t{Shape<1>{static_cast<std::size_t>(stop)}};
    std::iota(t.begin(), t.end(), 0);
    return t;
}

template<typename T>
Tensor<T, 1> range(T start, T stop) {
    if (start >= stop)
        return {};

    Shape<1> shape{static_cast<std::size_t>(stop - start)};
    Tensor<T, 1> t{shape};
    std::iota(t.begin(), t.end(), start);
    return t;
}

template<typename T>
Tensor<T, 1> range(T start, T stop, T step) {
    if (start >= stop)
        return {};

    Tensor<T, 1> t{Shape<1>{static_cast<std::size_t>((stop - start + step - 1) / step)}};
    T* it = t.begin();
    T num = start;
    while (num < stop) {
        *it = num;
        ++it;
        num += step;
    }
    return t;
}

template<typename T, typename U, std::size_t Dims>
bool is_equal(const Tensor<T, Dims>& lhs, const Tensor<U, Dims>& rhs) {
    if constexpr (std::is_same_v<T, U>) {
        if (&lhs == &rhs)
            return true;
    }

    if (lhs.shape() != rhs.shape())
        return false;

    return std::equal(lhs.begin(), lhs.end(), rhs.begin());
}

namespace impl {
template<typename T, std::size_t Dims, std::size_t CurrDim>
void print(std::ostream& os, const Tensor<T, Dims>& tensor, const T* data, std::size_t offset) {
    if constexpr (CurrDim == Dims) {
        os << *data;
    }
    else {
        os << '[';
        for (std::size_t i = 0; i < tensor.shape_at(CurrDim); i++) {
            print<T, Dims, CurrDim + 1>(os, tensor, data, offset / tensor.shape_at(CurrDim));
            os << ',';

            data += offset;
        }
        os << ']';
    }
}
}

template<typename T, std::size_t Dims>
std::ostream& operator<<(std::ostream& os, const Tensor<T, Dims>& tensor) {
    impl::print<T, Dims, 0>(os, tensor, tensor.data(), tensor.num_elem() / tensor.shape_at(0));
    return os;
}


}
