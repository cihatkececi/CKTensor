#include "catch.hpp"

#include "cktensor/ops.h"


using namespace ck;

TEST_CASE("Addition", "[Ops]") {
    Tensor<int, 1> lhs{1, 2, 3};
    Tensor<int, 1> rhs{6, 5, 4};
    Tensor<int, 1> sum{7, 7, 7};

    SECTION("Binary") {
        SECTION("Same type") {
            auto result = lhs + rhs;
            REQUIRE(is_equal(result, sum));
            STATIC_REQUIRE(std::is_same_v<decltype(result)::ValueType, int>);
        }

        SECTION("Double -Int") {
            auto result = Tensor<double, 1>(lhs) + rhs;
            REQUIRE(is_equal(result, sum));
            STATIC_REQUIRE(std::is_same_v<decltype(result)::ValueType, double>);
        }

        SECTION("Int - Double") {
            auto result = lhs + Tensor<double, 1>(rhs);
            REQUIRE(is_equal(result, sum));
            STATIC_REQUIRE(std::is_same_v<decltype(result)::ValueType, double>);
        }
    }

    SECTION("Inplace add") {
        lhs += rhs;
        REQUIRE(is_equal(lhs, sum));
    }
}

TEST_CASE("Subtraction", "[Ops]") {
    Tensor<int, 1> lhs{1, 2, 3};
    Tensor<int, 1> rhs{6, 5, 4};
    Tensor<int, 1> expected{-5, -3, -1};

    SECTION("Binary") {
        SECTION("Same type") {
            auto result = lhs - rhs;
            REQUIRE(is_equal(result, expected));
            STATIC_REQUIRE(std::is_same_v<decltype(result)::ValueType, int>);
        }

        SECTION("Double -Int") {
            auto result = Tensor<double, 1>(lhs) - rhs;
            REQUIRE(is_equal(result, expected));
            STATIC_REQUIRE(std::is_same_v<decltype(result)::ValueType, double>);
        }

        SECTION("Int - Double") {
            auto result = lhs - Tensor<double, 1>(rhs);
            REQUIRE(is_equal(result, expected));
            STATIC_REQUIRE(std::is_same_v<decltype(result)::ValueType, double>);
        }
    }

    SECTION("Inplace") {
        lhs -= rhs;
        REQUIRE(is_equal(lhs, expected));
    }
}

TEST_CASE("Multiplication", "[Ops]") {
    Tensor<int, 1> lhs{2, 2, 2};
    Tensor<int, 1> rhs{1, 2, 3};
    Tensor<int, 1> mult{2, 4, 6};

    SECTION("Binary") {
        SECTION("Same type") {
            auto result = lhs * rhs;
            REQUIRE(is_equal(result, mult));
            STATIC_REQUIRE(std::is_same_v<decltype(result)::ValueType, int>);
        }

        SECTION("Double -Int") {
            auto result = Tensor<double, 1>(lhs) * rhs;
            REQUIRE(is_equal(result, mult));
            STATIC_REQUIRE(std::is_same_v<decltype(result)::ValueType, double>);
        }

        SECTION("Int - Double") {
            auto result = lhs * Tensor<double, 1>(rhs);
            REQUIRE(is_equal(result, mult));
            STATIC_REQUIRE(std::is_same_v<decltype(result)::ValueType, double>);
        }
    }

    SECTION("Inplace") {
        lhs *= rhs;
        REQUIRE(is_equal(lhs, mult));
    }
}


TEST_CASE("Division", "[Ops]") {
    Tensor<int, 1> lhs{2, 4, 6};
    Tensor<int, 1> rhs{2, 2, 2};
    Tensor<int, 1> expected{1, 2, 3};

    SECTION("Binary") {
        SECTION("Same type") {
            auto result = lhs / rhs;
            REQUIRE(is_equal(result, expected));
            STATIC_REQUIRE(std::is_same_v<decltype(result)::ValueType, int>);
        }

        SECTION("Double -Int") {
            auto result = Tensor<double, 1>(lhs) / rhs;
            REQUIRE(is_equal(result, expected));
            STATIC_REQUIRE(std::is_same_v<decltype(result)::ValueType, double>);
        }

        SECTION("Int - Double") {
            auto result = lhs / Tensor<double, 1>(rhs);
            REQUIRE(is_equal(result, expected));
            STATIC_REQUIRE(std::is_same_v<decltype(result)::ValueType, double>);
        }
    }

    SECTION("Inplace") {
        lhs /= rhs;
        REQUIRE(is_equal(lhs, expected));
    }
}

TEST_CASE("Modulo", "[Ops]") {
    Tensor<int, 1> lhs{2, 2, 6};
    Tensor<int, 1> rhs{2, 2, 4};
    Tensor<int, 1> expected{0, 0, 2};

    SECTION("Binary") {
        SECTION("Same type") {
            auto result = lhs % rhs;
            REQUIRE(is_equal(result, expected));
            STATIC_REQUIRE(std::is_same_v<decltype(result)::ValueType, int>);
        }

        SECTION("Unsigned -Int") {
            auto result = Tensor<unsigned int, 1>(lhs) % rhs;
            REQUIRE(is_equal(result, expected));
            STATIC_REQUIRE(std::is_same_v<decltype(result)::ValueType, unsigned int>);
        }

        SECTION("Int - Unsigned") {
            auto result = lhs % Tensor<unsigned int, 1>(rhs);
            REQUIRE(is_equal(result, expected));
            STATIC_REQUIRE(std::is_same_v<decltype(result)::ValueType, unsigned int>);
        }
    }

    SECTION("Inplace") {
        lhs %= rhs;
        REQUIRE(is_equal(lhs, expected));
    }
}

TEST_CASE("Matrix product", "[Ops]") {
    Tensor<int, 2> lhs{{1, 2},
                       {3, 4}};
    Tensor<int, 2> rhs{{4, 3},
                       {5, 8}};
    Tensor<int, 2> expected{{14, 19},
                            {32, 41}};

    SECTION("Same type Int") {
        auto result = matmul(lhs, rhs);
        REQUIRE(is_equal(result, expected));
        STATIC_REQUIRE(std::is_same_v<decltype(result)::ValueType, int>);
    }

    SECTION("Same type Double") {
        auto result = matmul(lhs.as<double>(), rhs.as<double>());
        REQUIRE(is_equal(result, expected.as<double>()));
        STATIC_REQUIRE(std::is_same_v<decltype(result)::ValueType, double>);
    }

    SECTION("Double - Int") {
        auto result = matmul(lhs.as<double>(), rhs);
        REQUIRE(is_equal(result, expected.as<double>()));
        STATIC_REQUIRE(std::is_same_v<decltype(result)::ValueType, double>);
    }

    SECTION("Int - Double") {
        auto result = matmul(lhs, rhs.as<double>());
        REQUIRE(is_equal(result, expected.as<double>()));
        STATIC_REQUIRE(std::is_same_v<decltype(result)::ValueType, double>);
    }

    SECTION("Float -Int") {
        auto result = matmul(lhs.as<float>(), rhs);
        REQUIRE(is_equal(result, expected.as<float>()));
        STATIC_REQUIRE(std::is_same_v<decltype(result)::ValueType, float>);
    }
}

TEST_CASE("Equality", "[Ops]") {
    Tensor<int, 2> lhs{{1, 2},
                       {3, 4}};
    Tensor<int, 2> rhs{{1, 5},
                       {8, 4}};
    Tensor<bool, 2> expected{{true,  false},
                             {false, true}};

    SECTION("Same type") {
        auto result = (lhs == rhs);
        REQUIRE(is_equal(result, expected));
        STATIC_REQUIRE(std::is_same_v<decltype(result)::ValueType, bool>);
    }

    SECTION("Double -Int") {
        auto result = Tensor<double, 2>(lhs) == rhs;
        REQUIRE(is_equal(result, expected));
        STATIC_REQUIRE(std::is_same_v<decltype(result)::ValueType, bool>);
    }

    SECTION("Int - Double") {
        auto result = lhs == Tensor<double, 2>(rhs);
        REQUIRE(is_equal(result, expected));
        STATIC_REQUIRE(std::is_same_v<decltype(result)::ValueType, bool>);
    }
}

TEST_CASE("Inequality", "[Ops]") {
    Tensor<int, 2> lhs{{1, 2},
                       {3, 4}};
    Tensor<int, 2> rhs{{1, 5},
                       {8, 4}};
    Tensor<bool, 2> expected{{false, true},
                             {true,  false}};

    SECTION("Same type") {
        auto result = (lhs != rhs);
        REQUIRE(is_equal(result, expected));
        STATIC_REQUIRE(std::is_same_v<decltype(result)::ValueType, bool>);
    }

    SECTION("Double -Int") {
        auto result = Tensor<double, 2>(lhs) != rhs;
        REQUIRE(is_equal(result, expected));
        STATIC_REQUIRE(std::is_same_v<decltype(result)::ValueType, bool>);
    }

    SECTION("Int - Double") {
        auto result = lhs != Tensor<double, 2>(rhs);
        REQUIRE(is_equal(result, expected));
        STATIC_REQUIRE(std::is_same_v<decltype(result)::ValueType, bool>);
    }
}

TEST_CASE("Other comparison ops", "[Ops]") {
    Tensor<int, 2> lhs{{1, 2},
                       {8, 4}};
    Tensor<int, 2> rhs{{1, 5},
                       {3, 4}};

    SECTION("Less than") {
        auto result = lhs < rhs;
        REQUIRE(is_equal(result, Tensor<bool, 2>{{false, true},
                                                 {false, false}}));
        STATIC_REQUIRE(std::is_same_v<decltype(result)::ValueType, bool>);
    }

    SECTION("Less than or is_equal") {
        auto result = lhs <= rhs;
        REQUIRE(is_equal(result, Tensor<bool, 2>{{true,  true},
                                                 {false, true}}));
        STATIC_REQUIRE(std::is_same_v<decltype(result)::ValueType, bool>);
    }

    SECTION("Greater than") {
        auto result = lhs > rhs;
        REQUIRE(is_equal(result, Tensor<bool, 2>{{false, false},
                                                 {true,  false}}));
        STATIC_REQUIRE(std::is_same_v<decltype(result)::ValueType, bool>);
    }

    SECTION("Greater than or is_equal") {
        auto result = lhs >= rhs;
        REQUIRE(is_equal(result, Tensor<bool, 2>{{true, false},
                                                 {true, true}}));
        STATIC_REQUIRE(std::is_same_v<decltype(result)::ValueType, bool>);
    }
}

// TODO: Change the function call to operator+
TEST_CASE("Broadcasting ops", "[Ops]") {
    SECTION("Same dims") {
        SECTION("(2) and (2)") {
            const Tensor<int, 1> lhs{1, 2};
            const Tensor<int, 1> rhs{3, 4};
            const Tensor<int, 1> expected{4, 6};

            auto result = lhs + rhs;
            REQUIRE(is_equal(result, expected));
        }SECTION("(2x2) and (2x2)") {
            const Tensor<int, 2> lhs{{1, 2},
                                     {3, 4}};
            const Tensor<int, 2> rhs{{5, 6},
                                     {7, 8}};
            const Tensor<int, 2> expected{{6,  8},
                                          {10, 12}};

            auto result = lhs + rhs;
            REQUIRE(is_equal(result, expected));
        }SECTION("(2x2) and (1x2)") {
            const Tensor<int, 2> lhs{{1, 2},
                                     {3, 4}};
            const Tensor<int, 2> rhs{{5, 6}};
            const Tensor<int, 2> expected{{6, 8},
                                          {8, 10}};

            auto result = lhs + rhs;
            REQUIRE(is_equal(result, expected));
        }SECTION("(2x2) and (2x1)") {
            const Tensor<int, 2> lhs{{1, 2},
                                     {3, 4}};
            const Tensor<int, 2> rhs = Tensor<int, 2>{{5, 7}}.transpose();
            const Tensor<int, 2> expected{{6,  7},
                                          {10, 11}};

            auto result = lhs + rhs;

            REQUIRE(is_equal(result, expected));
        }SECTION("(2x2x2) and (2x2x2)") {
            const Tensor<int, 3> lhs{{{1, 2},
                                             {3, 4}},
                                     {{5, 6},
                                             {7, 8}}};
            const Tensor<int, 3> rhs{{{5, 6},
                                             {7, 8}},
                                     {{1, 2},
                                             {3, 4}}};
            const Tensor<int, 3> expected{{{6, 8},
                                                  {10, 12}},
                                          {{6, 8},
                                                  {10, 12}}};

            auto result = lhs + rhs;
            REQUIRE(is_equal(result, expected));
        }
    }

    SECTION("Different dims") {
        SECTION("(2x2) and (2)") {
            const Tensor<int, 2> lhs{{1, 2},
                                     {3, 4}};
            const Tensor<int, 1> rhs{5, 6};
            const Tensor<int, 2> expected{{6, 8},
                                          {8, 10}};

            auto result = lhs + rhs;
            REQUIRE(is_equal(result, expected));
        }
    }

}
