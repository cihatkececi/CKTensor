#include "catch.hpp"

#include "cktensor/tensor.h"
#include "cktensor/tensor_view.h"


using namespace ck;

TEST_CASE("TensorIter", "[TensorIter]") {
//    Tensor<int, 2> t{TensorShape<2>{3, 5}};
    Tensor<int, 2> t{{1, 2, 3, 4, 5},
                     {6, 7, 8, 9, 10},
                     {11, 12, 13, 14, 15}};
    // The slice (0:2, 1:4:2)
    TensorIter<int, 2> it{t.data() + 1 + 0*5, {2, 2}, {1*5, 2}};
    REQUIRE(*it == 2);
    REQUIRE(*(++it) == 4);
    REQUIRE(*(++it) == 7);
    REQUIRE(*(++it) == 9);

    REQUIRE(*(--it) == 7);
    REQUIRE(*(--it) == 4);
    REQUIRE(*(--it) == 2);
}

TEST_CASE("TensorView Constructor", "[TensorView]") {
    Tensor<int, 3> t;
}
