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

TEST_CASE("Sine", "[Functions]") {
    Tensor<double, 1> t{0, pi/2, 0, -pi/2};

    auto res = sin(t);

    REQUIRE(res.shape() == t.shape());
    REQUIRE(res.at(0) == 0.0);
    REQUIRE(res.at(1) == 1.0);
    REQUIRE(res.at(2) == 0.0);
    REQUIRE(res.at(3) == -1.0);
}
