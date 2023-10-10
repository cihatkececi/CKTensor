#pragma once

#include <vector>

#include <cmath>
#include <exception>

#include "cktensor/tensor.h"


namespace ck {

constexpr double pi = M_PI;

template<typename T, std::size_t Dims>
auto stack(const std::vector<Tensor<T, Dims>>& vals) {
    const auto& el_shape = vals.front().shape();

    for (std::size_t i = 1; i < vals.size(); i++) {
        if (vals[i].shape() != el_shape) {
            throw std::runtime_error{"Shapes of the elements are inconsistent"};
        }
    }

    Shape<Dims + 1ul> ret_shape;
    ret_shape[0] = vals.size();
    std::copy(el_shape.begin(), el_shape.end(), ret_shape.begin() + 1);

    Tensor<T, Dims + 1ul> ret{ret_shape};
    auto dest = ret.begin();

    for (const auto& val: vals) {
        dest = std::copy(val.begin(), val.end(), dest);
    }

    return ret;
}

template<typename T, std::size_t Dims>
auto abs(const Tensor<T, Dims>& t) {
    return t.map([](T el){ return std::abs(el); });
}

template<typename T, std::size_t Dims>
auto sqrt(const Tensor<T, Dims>& t) {
    return t.map([](T el){ return std::sqrt(el); });
}

/* Trigonometric functions */

template<typename T, std::size_t Dims>
auto sin(const Tensor<T, Dims>& t) {
    return t.map([](T el){ return std::sin(el); });
}

template<typename T, std::size_t Dims>
auto cos(const Tensor<T, Dims>& t) {
    return t.map([](T el){ return std::cos(el); });
}

template<typename T, std::size_t Dims>
auto tan(const Tensor<T, Dims>& t) {
    return t.map([](T el){ return std::tan(el); });
}

template<typename T, std::size_t Dims>
auto asin(const Tensor<T, Dims>& t) {
    return t.map([](T el){ return std::asin(el); });
}

template<typename T, std::size_t Dims>
auto acos(const Tensor<T, Dims>& t) {
    return t.map([](T el){ return std::acos(el); });
}

template<typename T, std::size_t Dims>
auto atan(const Tensor<T, Dims>& t) {
    return t.map([](T el){ return std::atan(el); });
}

template<typename T, std::size_t Dims>
auto atan2(const Tensor<T, Dims>& t) {
    return t.map([](T el){ return std::atan2(el); });
}

/* Exponential functions */

template<typename T, std::size_t Dims>
auto exp(const Tensor<T, Dims>& t) {
    return t.map([](T el){ return std::exp(el); });
}

template<typename T, std::size_t Dims>
auto log(const Tensor<T, Dims>& t) {
    return t.map([](T el){ return std::log(el); });
}

template<typename T, std::size_t Dims>
auto log2(const Tensor<T, Dims>& t) {
    return t.map([](T el){ return std::log2(el); });
}

template<typename T, std::size_t Dims>
auto log10(const Tensor<T, Dims>& t) {
    return t.map([](T el){ return std::log10(el); });
}

/* Hyperbolic functions */

template<typename T, std::size_t Dims>
auto sinh(const Tensor<T, Dims>& t) {
    return t.map([](T el){ return std::sinh(el); });
}

template<typename T, std::size_t Dims>
auto cosh(const Tensor<T, Dims>& t) {
    return t.map([](T el){ return std::cosh(el); });
}

template<typename T, std::size_t Dims>
auto tanh(const Tensor<T, Dims>& t) {
    return t.map([](T el){ return std::tanh(el); });
}

template<typename T, std::size_t Dims>
auto asinh(const Tensor<T, Dims>& t) {
    return t.map([](T el){ return std::asinh(el); });
}

template<typename T, std::size_t Dims>
auto acosh(const Tensor<T, Dims>& t) {
    return t.map([](T el){ return std::acosh(el); });
}

template<typename T, std::size_t Dims>
auto atanh(const Tensor<T, Dims>& t) {
    return t.map([](T el){ return std::atanh(el); });
}

/* Error and Gamma functions */

template<typename T, std::size_t Dims>
auto erf(const Tensor<T, Dims>& t) {
    return t.map([](T el){ return std::erf(el); });
}

template<typename T, std::size_t Dims>
auto erfc(const Tensor<T, Dims>& t) {
    return t.map([](T el){ return std::erfc(el); });
}

template<typename T, std::size_t Dims>
auto tgamma(const Tensor<T, Dims>& t) {
    return t.map([](T el){ return std::tgamma(el); });
}

template<typename T, std::size_t Dims>
auto lgamma(const Tensor<T, Dims>& t) {
    return t.map([](T el){ return std::lgamma(el); });
}

/* Nearest integer functions */

template<typename T, std::size_t Dims>
auto ceil(const Tensor<T, Dims>& t) {
    return t.map([](T el){ return std::ceil(el); });
}

template<typename T, std::size_t Dims>
auto floor(const Tensor<T, Dims>& t) {
    return t.map([](T el){ return std::floor(el); });
}

template<typename T, std::size_t Dims>
auto round(const Tensor<T, Dims>& t) {
    return t.map([](T el){ return std::round(el); });
}

}
