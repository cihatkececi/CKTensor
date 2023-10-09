#include "catch.hpp"

#include "cktensor/functions.h"

using namespace ck;

TEST_CASE("Stack", "[Functions]") {
    Tensor<int, 1> t1{1, 2};
    Tensor<int, 1> t2{3, 4};

    auto res = stack(std::vector{t1, t2});

    REQUIRE(res.shape() == Shape<2>{2, 2});
    REQUIRE(is_equal(res, Tensor<int, 2>{{1, 2},
                                         {3, 4}}));
}
