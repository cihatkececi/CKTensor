#pragma once

#include <complex>

#include "tensor.h"
#include "gemm.h"

// TODO: Add inplace operators

namespace ck {

// Implementation for the broadcasting ops
class BroadcastError : public std::exception {
};

namespace impl {
template<typename Op, typename T, typename U>
struct RetScalarType {
    using type = decltype(std::declval<Op>()(std::declval<T>(), std::declval<U>()));
};

template<typename Op, typename T, std::size_t TDims, typename U, std::size_t UDims>
struct RetTensorType {
    using type = Tensor<RetScalarType<Op, T, U>, std::max(TDims, UDims)>;
};

template<typename T, typename U>
struct Add {
    constexpr auto operator()(const T& lhs, const U& rhs) const {
        return lhs + rhs;
    }
};

template<typename T, typename U>
struct Sub {
    constexpr auto operator()(const T& lhs, const U& rhs) const {
        return lhs - rhs;
    }
};

template<typename T, typename U>
struct Mul {
    constexpr auto operator()(const T& lhs, const U& rhs) const {
        return lhs * rhs;
    }
};

template<typename T, typename U>
struct Div {
    constexpr auto operator()(const T& lhs, const U& rhs) const {
        return lhs / rhs;
    }
};

template<typename T, typename U>
struct Mod {
    constexpr auto operator()(const T& lhs, const U& rhs) const {
        return lhs % rhs;
    }
};

template<typename T, typename U>
struct Eq {
    constexpr auto operator()(const T& lhs, const U& rhs) const {
        return lhs == rhs;
    }
};

template<typename T, typename U>
struct Neq {
    constexpr auto operator()(const T& lhs, const U& rhs) const {
        return lhs != rhs;
    }
};

template<typename T, typename U>
struct Lt {
    constexpr auto operator()(const T& lhs, const U& rhs) const {
        return lhs < rhs;
    }
};

template<typename T, typename U>
struct Le {
    constexpr auto operator()(const T& lhs, const U& rhs) const {
        return lhs <= rhs;
    }
};

template<typename T, typename U>
struct Gt {
    constexpr auto operator()(const T& lhs, const U& rhs) const {
        return lhs > rhs;
    }
};

template<typename T, typename U>
struct Ge {
    constexpr auto operator()(const T& lhs, const U& rhs) const {
        return lhs >= rhs;
    }
};

template<typename T, typename U>
struct Pow {
    constexpr auto operator()(const T& lhs, const U& rhs) const {
        return std::pow(lhs, rhs);
    }
};

template<std::size_t LHSDim, std::size_t RHSDim>
auto broadcast_shape(const Shape<LHSDim>& lhs, const Shape<RHSDim>& rhs) {
    Shape<std::max(LHSDim, RHSDim)> ret;

    std::size_t i = 0;
    for (; i < std::min(LHSDim, RHSDim); i++) {
        if (lhs.rat(i) == rhs.rat(i)) {
            ret.rat(i) = lhs.rat(i);
        }
        else if (lhs.rat(i) == 1) {
            ret.rat(i) = rhs.rat(i);
        }
        else if (rhs.rat(i) == 1) {
            ret.rat(i) = lhs.rat(i);
        }
        else {
            throw BroadcastError{};
        }
    }

    if constexpr (LHSDim > RHSDim) {
        for (; i < LHSDim; i++) {
            ret.rat(i) = lhs.rat(i);
        }
    }
    else if constexpr (RHSDim > LHSDim) {
        for (; i < RHSDim; i++) {
            ret.rat(i) = rhs.rat(i);
        }
    }

    return ret;
}

template<typename Op, std::size_t dims, std::size_t curr_dim>
void binary_op_broadcasting_impl(const auto*& lhs, const std::size_t* lhs_shape,
                                 const auto*& rhs, const std::size_t* rhs_shape,
                                 auto*& out) {
    Op op;

    if constexpr (curr_dim == dims) {
        *out = op(*lhs, *rhs);
        lhs++;
        rhs++;
        out++;
        return;
    }
    else {
        if (*lhs_shape == *rhs_shape) {
            for (std::size_t i = 0; i < *lhs_shape; i++) {
                binary_op_broadcasting_impl<Op, dims, curr_dim + 1>(lhs, lhs_shape + 1, rhs, rhs_shape + 1, out);
            }
        }
        else if (*lhs_shape == 1) {
            for (std::size_t i = 0; i < *rhs_shape; i++) {
                auto lhs_begin = lhs;
                binary_op_broadcasting_impl<Op, dims, curr_dim + 1>(lhs, lhs_shape + 1, rhs, rhs_shape + 1, out);

                if (i != *rhs_shape - 1)
                    lhs = lhs_begin;
            }
        }
        else if (*rhs_shape == 1) {
            for (std::size_t i = 0; i < *lhs_shape; i++) {
                auto rhs_begin = rhs;
                binary_op_broadcasting_impl<Op, dims, curr_dim + 1>(lhs, lhs_shape + 1, rhs, rhs_shape + 1, out);

                if (i != *lhs_shape - 1)
                    rhs = rhs_begin;
            }
        }
    }
}

template<typename Op, typename T, std::size_t TDims, typename U, std::size_t UDims>
struct BinaryOpBroadcasting {
    using RetType = typename RetScalarType<Op, T, U>::type;

    constexpr auto operator()(const Tensor<T, TDims>& lhs, const Tensor<U, UDims>& rhs) const {
        auto out_shape = broadcast_shape(lhs.shape(), rhs.shape());
        Tensor<RetType, TDims> out{out_shape};
        Shape<out_shape.dims()> lhs_shape;
        for (std::size_t i = 0; i < TDims; i++)
            lhs_shape.rat(i) = lhs.shape().rat(i);
        for (std::size_t i = TDims; i < out_shape.dims(); i++)
            lhs_shape.rat(i) = 1;

        Shape<out_shape.dims()> rhs_shape;
        for (std::size_t i = 0; i < UDims; i++)
            rhs_shape.rat(i) = rhs.shape().rat(i);
        for (std::size_t i = UDims; i < out_shape.dims(); i++)
            rhs_shape.rat(i) = 1;

        auto* lhs_data = lhs.data();
        auto* rhs_data = rhs.data();
        auto* out_data = out.data();

        binary_op_broadcasting_impl<Op, out_shape.dims(), 0>(lhs_data, lhs_shape.data(), rhs_data, rhs_shape.data(),
                                                             out_data);

        return out;
    }
};

template<typename Op, typename T, std::size_t TDims, typename U, std::size_t UDims>
struct BinaryOpSameShape {
    using RetType = typename RetScalarType<Op, T, U>::type;

    constexpr auto operator()(const Tensor<T, TDims>& lhs, const Tensor<U, UDims>& rhs) const {
        Op op;

        Tensor<RetType, TDims> res{lhs.shape()};

        for (std::size_t i = 0; i < lhs.num_elem(); i++) {
            res.data()[i] = op(lhs.data()[i], rhs.data()[i]);
        }

        return res;
    }
};

// TODO: I can optionally construct the return value inside this function and pass a reference to the impl functions
template<typename Op, typename T, std::size_t TDims, typename U, std::size_t UDims>
struct BinaryOp {
    constexpr auto operator()(const Tensor<T, TDims>& lhs, const Tensor<U, UDims>& rhs) const {
        if constexpr (TDims == UDims) {
            if (lhs.shape() == rhs.shape()) {
                return BinaryOpSameShape<Op, T, TDims, U, UDims>{}(lhs, rhs);
            }
        }

        return BinaryOpBroadcasting<Op, T, TDims, U, UDims>{}(lhs, rhs);
    }
};

template<typename Op, typename T, std::size_t TDims, typename U>
struct BinaryOp<Op, T, TDims, U, 0> {
    using RetType = typename RetScalarType<Op, T, U>::type;

    constexpr auto operator()(const Tensor<T, TDims>& lhs, const U& rhs) const {
        Op op;

        Tensor<RetType, TDims> res{lhs.shape()};

        for (std::size_t i = 0; i < lhs.num_elem(); i++) {
            res.data()[i] = op(lhs.data()[i], rhs);
        }

        return res;
    }

    constexpr auto operator()(const U& lhs, const Tensor<T, TDims>& rhs) const {
        Op op;

        Tensor<RetType, TDims> res{rhs.shape()};

        for (std::size_t i = 0; i < rhs.num_elem(); i++) {
            res.data()[i] = op(lhs, rhs.data()[i]);
        }

        return res;
    }
};
}

/* Unary operators */

template<typename T, std::size_t Dims>
Tensor<T, Dims> Tensor<T, Dims>::operator!() const {
    Tensor<T, Dims> result{*this};
    for (size_t i = 0; i < num_elem(); ++i) {
        result[i] = !data_[i];
    }
    return result;
}

template<typename T, std::size_t Dims>
Tensor<T, Dims> Tensor<T, Dims>::operator-() const {
    Tensor<T, Dims> result{*this};
    for (size_t i = 0; i < num_elem(); ++i) {
        result[i] = -data_[i];
    }
    return result;
}

/* Inplace operators */

template<typename T, std::size_t Dims>
Tensor<T, Dims>& Tensor<T, Dims>::operator+=(const Tensor<T, Dims>& other) {
    assert(shape() == other.shape());

    for (size_t i = 0; i < num_elem(); ++i) {
        data_[i] += other.data_[i];
    }
    return *this;
}

template<typename T, std::size_t Dims>
Tensor<T, Dims>& Tensor<T, Dims>::operator-=(const Tensor<T, Dims>& other) {
    assert(shape() == other.shape());

    for (size_t i = 0; i < num_elem(); ++i) {
        data_[i] -= other.data_[i];
    }
    return *this;
}

template<typename T, std::size_t Dims>
Tensor<T, Dims>& Tensor<T, Dims>::operator*=(const Tensor<T, Dims>& other) {
    assert(shape() == other.shape());

    for (size_t i = 0; i < num_elem(); ++i) {
        data_[i] *= other.data_[i];
    }
    return *this;
}

template<typename T, std::size_t Dims>
Tensor<T, Dims>& Tensor<T, Dims>::operator/=(const Tensor<T, Dims>& other) {
    assert(shape() == other.shape());

    for (size_t i = 0; i < num_elem(); ++i) {
        data_[i] /= other.data_[i];
    }
    return *this;
}

template<typename T, std::size_t Dims>
Tensor<T, Dims>& Tensor<T, Dims>::operator%=(const Tensor<T, Dims>& other) {
    assert(shape() == other.shape());

    for (size_t i = 0; i < num_elem(); ++i) {
        data_[i] %= other.data_[i];
    }
    return *this;
}

/* Binary operators */

template<typename T, std::size_t TDims, typename U, std::size_t UDims>
auto operator+(const Tensor<T, TDims>& lhs, const Tensor<U, UDims>& rhs) {
    return impl::BinaryOp<impl::Add<T, U>, T, TDims, U, UDims>{}(lhs, rhs);
}

template<typename T, std::size_t TDims, typename U>
auto operator+(const Tensor<T, TDims>& lhs, const U& rhs) {
    return impl::BinaryOp<impl::Add<T, U>, T, TDims, U, 0>{}(lhs, rhs);
}

template<typename T, std::size_t TDims, typename U>
auto operator+(const U& lhs, const Tensor<T, TDims>& rhs) {
    return impl::BinaryOp<impl::Add<T, U>, T, TDims, U, 0>{}(lhs, rhs);
}

template<typename T, std::size_t TDims, typename U, std::size_t UDims>
auto operator-(const Tensor<T, TDims>& lhs, const Tensor<U, UDims>& rhs) {
    return impl::BinaryOp<impl::Sub<T, U>, T, TDims, U, UDims>{}(lhs, rhs);
}

template<typename T, std::size_t TDims, typename U>
auto operator-(const Tensor<T, TDims>& lhs, const U& rhs) {
    return impl::BinaryOp<impl::Sub<T, U>, T, TDims, U, 0>{}(lhs, rhs);
}

template<typename T, std::size_t TDims, typename U>
auto operator-(const U& lhs, const Tensor<T, TDims>& rhs) {
    return impl::BinaryOp<impl::Sub<T, U>, T, TDims, U, 0>{}(lhs, rhs);
}

template<typename T, std::size_t TDims, typename U, std::size_t UDims>
auto operator*(const Tensor<T, TDims>& lhs, const Tensor<U, UDims>& rhs) {
    return impl::BinaryOp<impl::Mul<T, U>, T, TDims, U, UDims>{}(lhs, rhs);
}

template<typename T, std::size_t TDims, typename U>
auto operator*(const Tensor<T, TDims>& lhs, const U& rhs) {
    return impl::BinaryOp<impl::Mul<T, U>, T, TDims, U, 0>{}(lhs, rhs);
}

template<typename T, std::size_t TDims, typename U>
auto operator*(const U& lhs, const Tensor<T, TDims>& rhs) {
    return impl::BinaryOp<impl::Mul<T, U>, T, TDims, U, 0>{}(lhs, rhs);
}

template<typename T, std::size_t TDims, typename U, std::size_t UDims>
auto operator/(const Tensor<T, TDims>& lhs, const Tensor<U, UDims>& rhs) {
    return impl::BinaryOp<impl::Div<T, U>, T, TDims, U, UDims>{}(lhs, rhs);
}

template<typename T, std::size_t TDims, typename U>
auto operator/(const Tensor<T, TDims>& lhs, const U& rhs) {
    return impl::BinaryOp<impl::Div<T, U>, T, TDims, U, 0>{}(lhs, rhs);
}

template<typename T, std::size_t TDims, typename U>
auto operator/(const U& lhs, const Tensor<T, TDims>& rhs) {
    return impl::BinaryOp<impl::Div<T, U>, T, TDims, U, 0>{}(lhs, rhs);
}

template<typename T, std::size_t TDims, typename U, std::size_t UDims>
auto operator%(const Tensor<T, TDims>& lhs, const Tensor<U, UDims>& rhs) {
    return impl::BinaryOp<impl::Mod<T, U>, T, TDims, U, UDims>{}(lhs, rhs);
}

template<typename T, std::size_t TDims, typename U>
auto operator%(const Tensor<T, TDims>& lhs, const U& rhs) {
    return impl::BinaryOp<impl::Mod<T, U>, T, TDims, U, 0>{}(lhs, rhs);
}

template<typename T, std::size_t TDims, typename U>
auto operator%(const U& lhs, const Tensor<T, TDims>& rhs) {
    return impl::BinaryOp<impl::Mod<T, U>, T, TDims, U, 0>{}(lhs, rhs);
}

template<typename T, std::size_t TDims, typename U, std::size_t UDims>
auto operator==(const Tensor<T, TDims>& lhs, const Tensor<U, UDims>& rhs) {
    return impl::BinaryOp<impl::Eq<T, U>, T, TDims, U, UDims>{}(lhs, rhs);
}

template<typename T, std::size_t TDims, typename U>
auto operator==(const Tensor<T, TDims>& lhs, const U& rhs) {
    return impl::BinaryOp<impl::Eq<T, U>, T, TDims, U, 0>{}(lhs, rhs);
}

template<typename T, std::size_t TDims, typename U>
auto operator==(const U& lhs, const Tensor<T, TDims>& rhs) {
    return impl::BinaryOp<impl::Eq<T, U>, T, TDims, U, 0>{}(lhs, rhs);
}

template<typename T, std::size_t TDims, typename U, std::size_t UDims>
auto operator!=(const Tensor<T, TDims>& lhs, const Tensor<U, UDims>& rhs) {
    return impl::BinaryOp<impl::Neq<T, U>, T, TDims, U, UDims>{}(lhs, rhs);
}

template<typename T, std::size_t TDims, typename U>
auto operator!=(const Tensor<T, TDims>& lhs, const U& rhs) {
    return impl::BinaryOp<impl::Neq<T, U>, T, TDims, U, 0>{}(lhs, rhs);
}

template<typename T, std::size_t TDims, typename U>
auto operator!=(const U& lhs, const Tensor<T, TDims>& rhs) {
    return impl::BinaryOp<impl::Neq<T, U>, T, TDims, U, 0>{}(lhs, rhs);
}

template<typename T, std::size_t TDims, typename U, std::size_t UDims>
auto operator<(const Tensor<T, TDims>& lhs, const Tensor<U, UDims>& rhs) {
    return impl::BinaryOp<impl::Lt<T, U>, T, TDims, U, UDims>{}(lhs, rhs);
}

template<typename T, std::size_t TDims, typename U>
auto operator<(const Tensor<T, TDims>& lhs, const U& rhs) {
    return impl::BinaryOp<impl::Lt<T, U>, T, TDims, U, 0>{}(lhs, rhs);
}

template<typename T, std::size_t TDims, typename U>
auto operator<(const U& lhs, const Tensor<T, TDims>& rhs) {
    return impl::BinaryOp<impl::Lt<T, U>, T, TDims, U, 0>{}(lhs, rhs);
}

template<typename T, std::size_t TDims, typename U, std::size_t UDims>
auto operator<=(const Tensor<T, TDims>& lhs, const Tensor<U, UDims>& rhs) {
    return impl::BinaryOp<impl::Le<T, U>, T, TDims, U, UDims>{}(lhs, rhs);
}

template<typename T, std::size_t TDims, typename U>
auto operator<=(const Tensor<T, TDims>& lhs, const U& rhs) {
    return impl::BinaryOp<impl::Le<T, U>, T, TDims, U, 0>{}(lhs, rhs);
}

template<typename T, std::size_t TDims, typename U>
auto operator<=(const U& lhs, const Tensor<T, TDims>& rhs) {
    return impl::BinaryOp<impl::Le<T, U>, T, TDims, U, 0>{}(lhs, rhs);
}

template<typename T, std::size_t TDims, typename U, std::size_t UDims>
auto operator>(const Tensor<T, TDims>& lhs, const Tensor<U, UDims>& rhs) {
    return impl::BinaryOp<impl::Gt<T, U>, T, TDims, U, UDims>{}(lhs, rhs);
}

template<typename T, std::size_t TDims, typename U>
auto operator>(const Tensor<T, TDims>& lhs, const U& rhs) {
    return impl::BinaryOp<impl::Gt<T, U>, T, TDims, U, 0>{}(lhs, rhs);
}

template<typename T, std::size_t TDims, typename U>
auto operator>(const U& lhs, const Tensor<T, TDims>& rhs) {
    return impl::BinaryOp<impl::Gt<T, U>, T, TDims, U, 0>{}(lhs, rhs);
}

template<typename T, std::size_t TDims, typename U, std::size_t UDims>
auto operator>=(const Tensor<T, TDims>& lhs, const Tensor<U, UDims>& rhs) {
    return impl::BinaryOp<impl::Ge<T, U>, T, TDims, U, UDims>{}(lhs, rhs);
}

template<typename T, std::size_t TDims, typename U>
auto operator>=(const Tensor<T, TDims>& lhs, const U& rhs) {
    return impl::BinaryOp<impl::Ge<T, U>, T, TDims, U, 0>{}(lhs, rhs);
}

template<typename T, std::size_t TDims, typename U>
auto operator>=(const U& lhs, const Tensor<T, TDims>& rhs) {
    return impl::BinaryOp<impl::Ge<T, U>, T, TDims, U, 0>{}(lhs, rhs);
}

template<typename T, std::size_t TDims, typename U, std::size_t UDims>
auto pow(const Tensor<T, TDims>& lhs, const Tensor<U, UDims>& rhs) {
    return impl::BinaryOp<impl::Pow<T, U>, T, TDims, U, UDims>{}(lhs, rhs);
}

template<typename T, std::size_t TDims, typename U>
auto pow(const Tensor<T, TDims>& lhs, const U& rhs) {
    return impl::BinaryOp<impl::Pow<T, U>, T, TDims, U, 0>{}(lhs, rhs);
}

template<typename T, std::size_t TDims, typename U>
auto pow(const U& lhs, const Tensor<T, TDims>& rhs) {
    return impl::BinaryOp<impl::Pow<T, U>, T, TDims, U, 0>{}(lhs, rhs);
}

template<typename T, std::size_t TDims, typename U, std::size_t UDims>
auto matmul(const Tensor<T, TDims>& lhs, const Tensor<U, UDims>& rhs) {
    return impl::MatMul<T, TDims, U, UDims>{}(lhs, rhs);
}

}
