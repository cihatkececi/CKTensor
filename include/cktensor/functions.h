#pragma once

#include <vector>
#include <exception>

#include "cktensor/tensor.h"


namespace ck {

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

}
