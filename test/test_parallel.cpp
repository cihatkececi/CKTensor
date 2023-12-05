#include "catch.hpp"

#include "cktensor/parallel.h"

using namespace ck;

TEST_CASE("Map", "[par]") {
    auto t = range(10);

    auto res = par::map([](auto x){ return x * x; }, t, 12);

    REQUIRE(is_equal(res, Tensor<int, 1>{0, 1, 4, 9, 16, 25, 36, 49, 64, 81}));
}

TEST_CASE("Reduce", "[par]") {
    auto t = range(10);
    auto res = par::reduce_add(t, 12);
    REQUIRE(res == 45);
}
