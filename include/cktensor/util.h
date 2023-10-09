#pragma once

#include <array>
#include <vector>


namespace ck {

template<std::size_t Dims>
struct Shape;

namespace impl {
template<typename T, std::size_t Dims>
struct TensorShapeFromSTL {
    template<template<typename, typename> typename Container, typename Alloc>
    Shape<Dims> operator()(const Container<T, Alloc>& val, std::size_t& dim) const {
        Shape<Dims> shape{1};
        shape[dim] = val.size();
        dim--;
        return shape;
    }

    template<template<typename, typename> typename Container, typename U, typename Alloc>
    Shape<Dims> operator()(const Container<U, Alloc>& val, std::size_t& dim) const {
        Shape<Dims> shape = TensorShapeFromSTL<T, Dims>{}(val.front(), dim);
        shape[dim] = val.size();
        dim--;
        return shape;
    }

    Shape<Dims> operator()(const std::initializer_list<T>& val, std::size_t& dim) const {
        Shape<Dims> shape{1};
        shape[dim] = val.size();
        dim--;
        return shape;
    }

    template<typename U>
    Shape<Dims> operator()(const std::initializer_list<U>& val, std::size_t& dim) const {
        Shape<Dims> shape = TensorShapeFromSTL<T, Dims>{}(*val.begin(), dim);
        shape[dim] = val.size();
        dim--;
        return shape;
    }

    template <std::size_t N>
    Shape<Dims> operator()(const std::array<T, N>& val, std::size_t& dim) const {
        Shape<Dims> shape{1};
        shape[dim] = val.size();
        dim--;
        return shape;
    }

    template<typename U, std::size_t N>
    Shape<Dims> operator()(const std::array<U, N>& val, std::size_t& dim) const {
        Shape<Dims> shape = TensorShapeFromSTL<T, Dims>{}(val.front(), dim);
        shape[dim] = val.size();
        dim--;
        return shape;
    }
};
}

template<typename T, std::size_t Dims>
struct TensorShapeFromSTL {
    template<template<typename, typename> typename Container, typename U, typename Alloc>
    Shape<Dims> operator()(const Container<U, Alloc>& val) const {
        std::size_t dim = Dims - 1;
        return impl::TensorShapeFromSTL<T, Dims>{}(val, dim);
    }

    template<typename U>
    Shape<Dims> operator()(const std::initializer_list<U>& val) const {
        std::size_t dim = Dims - 1;
        return impl::TensorShapeFromSTL<T, Dims>{}(val, dim);
    }

    template<typename U, std::size_t N>
    Shape<Dims> operator()(const std::array<U, N>& val) const {
        std::size_t dim = Dims - 1;
        return impl::TensorShapeFromSTL<T, Dims>{}(val, dim);
    }
};

template<typename SrcIt, typename DestIt>
DestIt recursive_copy(SrcIt first, SrcIt last, DestIt dest) {
    if constexpr (std::is_same_v<std::remove_cvref<typename std::iterator_traits<SrcIt>::value_type>,
            std::remove_cvref<typename std::iterator_traits<DestIt>::value_type>>) {
        return std::copy(first, last, dest);
    }
    else {
        for (; first != last; ++first)
            dest = recursive_copy(first->begin(), first->end(), dest);
        return dest;
    }
}

}
