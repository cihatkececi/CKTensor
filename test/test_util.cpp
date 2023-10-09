#include "catch.hpp"

#include "cktensor/util.h"
#include "cktensor/tensor.h"


using namespace ck;

TEST_CASE("Test tensor_shape_from_stl", "[Util]") {
    SECTION("One Dim") {
        std::vector<int> vec{1, 2, 3};
        auto s0 = TensorShapeFromSTL<int, 1>{}(vec);
        REQUIRE(s0.dims() == 1);
        REQUIRE(s0[0] == 3);
    }SECTION("Two Dims") {
        std::vector<std::vector<int>> vec{
                {1, 2, 3},
                {4, 5, 6},
        };
        auto s0 = TensorShapeFromSTL<int, 2>{}(vec);
        REQUIRE(s0.dims() == 2);
        REQUIRE(s0[0] == 2);
        REQUIRE(s0[1] == 3);
    }SECTION("Three Dims") {
        std::vector<std::vector<std::vector<int>>> vec{
                {
                        {1, 2, 3},
                        {4, 5, 6}
                },
                {
                        {7, 8, 9},
                        { 10, 11, 12 }
                },
        };
        auto s0 = TensorShapeFromSTL<int, 3>{}(vec);
        REQUIRE(s0.dims() == 3);
        REQUIRE(s0[0] == 2);
        REQUIRE(s0[1] == 2);
        REQUIRE(s0[2] == 3);
    }
}

TEST_CASE("Test recursive_copy", "[Util]") {
    SECTION("One Dim") {
        std::vector<int> v1{1, 2, 3};
        std::vector<int> v2(3);

        recursive_copy(v1.begin(), v1.end(), v2.begin());
        REQUIRE(v2.size() == 3);
        REQUIRE(v2[0] == 1);
        REQUIRE(v2[1] == 2);
        REQUIRE(v2[2] == 3);
    }

    SECTION("Two Dims") {
        std::vector<std::vector<int>> v1{{1, 2}, {3, 4}};
        std::vector<int> v2(4);

        recursive_copy(v1.begin(), v1.end(), v2.begin());
        REQUIRE(v2.size() == 4);
        REQUIRE(v2[0] == 1);
        REQUIRE(v2[1] == 2);
        REQUIRE(v2[2] == 3);
        REQUIRE(v2[3] == 4);
    }
}

