#include "catch.hpp"

#include "cktensor/tensor.h"

using namespace ck;


TEST_CASE("TensorShape Constructors", "[TensorShape]") {
    Shape<0> s0{};
    REQUIRE(s0.num_elem() == 1);

    Shape<1> s1{3};
    REQUIRE(s1.num_elem() == 3);
    REQUIRE(s1[0] == 3);

    Shape<2> s2{3, 5};
    REQUIRE(s2.num_elem() == 15);
    REQUIRE(s2[0] == 3);
    REQUIRE(s2[1] == 5);

    Shape<3> s3{3, 5, 7};
    REQUIRE(s3.num_elem() == 105);
    REQUIRE(s3[0] == 3);
    REQUIRE(s3[1] == 5);
    REQUIRE(s3[2] == 7);
}

TEST_CASE("Tensor Constructors", "[Tensor]") {
    SECTION("Default Constructor") {
        Tensor<int, 0> t0;
        Tensor<int, 1> t1;
        Tensor<int, 2> t2;
        Tensor<int, 3> t3;
        Tensor<int, 4> t4;
    }

//    SECTION("Implicit Shape Constructor") {
//        Tensor<float, 1> t1{3};
//        REQUIRE(t1.shape().dims() == 1);
//        REQUIRE(t1.shape()[0] == 3);
//
//        Tensor<int, 2> t2{3, 5};
//        REQUIRE(t1.shape().dims() == 2);
//        REQUIRE(t1.shape()[0] == 3);
//        REQUIRE(t1.shape()[1] == 5);
//
//        Tensor<int, 3> t3{3, 5, 7};
//        REQUIRE(t1.shape().dims() == 3);
//        REQUIRE(t1.shape()[0] == 3);
//        REQUIRE(t1.shape()[1] == 5);
//        REQUIRE(t1.shape()[2] == 7);
//
//        Tensor<int, 4> t4{3, 5, 7, 9};
//        REQUIRE(t1.shape().dims() == 4);
//        REQUIRE(t1.shape()[0] == 3);
//        REQUIRE(t1.shape()[1] == 5);
//        REQUIRE(t1.shape()[2] == 7);
//        REQUIRE(t1.shape()[3] == 9);
//    }

    SECTION("Explicit Shape Constructor") {
        Tensor<int, 0> t0(Shape<0>{});
        Tensor<int, 1> t1(Shape<1>{3});
        Tensor<int, 2> t2(Shape<2>{3, 5});
        Tensor<int, 3> t3(Shape<3>{3, 5, 7});
        Tensor<int, 4> t4(Shape<4>{3, 5, 7, 9});
    }

    SECTION("From Initializer List") {
        Tensor<int, 1> t1{3, 5, 7};
        REQUIRE(t1.shape()[0] == 3);

        Tensor<int, 2> t2{{1, 2, 3},
                          {4, 5, 6}};
        REQUIRE(t2.shape()[0] == 2);
        REQUIRE(t2.shape()[1] == 3);

        Tensor<int, 3> t3{{{1, 2}, {4, 5}},
                          {{6, 7}, {8, 9}}};
        REQUIRE(t3.shape()[0] == 2);
        REQUIRE(t3.shape()[1] == 2);
        REQUIRE(t3.shape()[2] == 2);
    }

    SECTION("From STL Containers") {
        SECTION("From std::array") {
            Tensor<int, 1> t1(std::array<int, 3>{3, 5, 7});
            REQUIRE(t1.shape()[0] == 3);

            Tensor<int, 2> t2(std::array<std::array<int, 3>, 2>{{{1, 2, 3},
                                                                 {4, 5, 6}}});
            REQUIRE(t2.shape()[0] == 2);
            REQUIRE(t2.shape()[1] == 3);
        }

        SECTION("From std::vector") {
            Tensor<int, 1> t1(std::vector<int>{3, 5, 7});
            REQUIRE(t1.shape()[0] == 3);

            Tensor<int, 2> t2(std::vector<std::vector<int>>{{1, 2, 3},
                                                            {4, 5, 6}});
            REQUIRE(t2.shape()[0] == 2);
            REQUIRE(t2.shape()[1] == 3);
        }

        SECTION("From std::deque") {
            Tensor<int, 1> t1(std::deque<int>{3, 5, 7});
            REQUIRE(t1.shape()[0] == 3);

            Tensor<int, 2> t2(std::deque<std::deque<int>>{{1, 2, 3},
                                                          {4, 5, 6}});
            REQUIRE(t2.shape()[0] == 2);
            REQUIRE(t2.shape()[1] == 3);
        }

        SECTION("From std::list") {
            Tensor<int, 1> t1(std::list<int>{3, 5, 7});
            REQUIRE(t1.shape()[0] == 3);

            Tensor<int, 2> t2(std::list<std::list<int>>{{1, 2, 3},
                                                        {4, 5, 6}});
            REQUIRE(t2.shape()[0] == 2);
            REQUIRE(t2.shape()[1] == 3);
        }

        SECTION("Non-trivial objects") {
            Tensor<std::string, 1> t{Shape<1>{3}};
            REQUIRE(t.shape_at(0) == 3);
            REQUIRE(t.at(0) == "");
            REQUIRE(t.at(1) == "");
            REQUIRE(t.at(2) == "");
        }

        SECTION("Non-trivial objects with value") {
            Tensor<std::string, 1> t{Shape<1>{3}, "hello"};
            REQUIRE(t.shape_at(0) == 3);
            REQUIRE(t.at(0) == "hello");
            REQUIRE(t.at(1) == "hello");
            REQUIRE(t.at(2) == "hello");
        }

        SECTION("Non-trivial objects with initialization list") {
            Tensor<std::string, 1> t{"hello", "world"};
            REQUIRE(t.shape_at(0) == 2);
            REQUIRE(t.at(0) == "hello");
            REQUIRE(t.at(1) == "world");
        }
    }
}

TEST_CASE("Tensor typecasting", "[Tensor]") {
    Tensor<int, 1> t{1, 2, 3};
    auto dt = t.as<double>();
    REQUIRE(dt.shape()[0] == 3);
    STATIC_REQUIRE(std::is_same_v<std::remove_reference_t<decltype(dt.at(0))>, double>);
    REQUIRE(dt.at(0) == 1);
    REQUIRE(dt.at(1) == 2);
    REQUIRE(dt.at(2) == 3);

    auto ft = t.as<float>();
    REQUIRE(ft.shape()[0] == 3);
    STATIC_REQUIRE(std::is_same_v<std::remove_reference_t<decltype(ft.at(0))>, float>);
    REQUIRE(ft.at(0) == 1);
    REQUIRE(ft.at(1) == 2);
    REQUIRE(ft.at(2) == 3);
}

TEST_CASE("Reserve", "[Tensor]") {
    SECTION("Fresh Tensor") {
        Tensor<int, 1> t;
        t.reserve(4);
        REQUIRE(t.capacity() == 4);

        t.reserve(2);
        REQUIRE(t.capacity() == 4);
    }

    SECTION("Initialized Tensor") {
        Tensor<int, 2> t{{1, 2},
                         {3, 4}};
        t.reserve(6);

        REQUIRE(t.capacity() == 6);
        REQUIRE(t.num_elem() == 4);
        REQUIRE(t.num_rows() == 2);
        REQUIRE(t.num_cols() == 2);
        REQUIRE(t.at(0, 0) == 1);
        REQUIRE(t.at(0, 1) == 2);
        REQUIRE(t.at(1, 0) == 3);
        REQUIRE(t.at(1, 1) == 4);
    }
}

TEST_CASE("Fill", "[Tensor]") {
    Tensor<int, 2> t{Shape<2>{2, 2}, 5};

    REQUIRE(t.at(0, 0) == 5);
    REQUIRE(t.at(0, 1) == 5);
    REQUIRE(t.at(0, 2) == 5);
    REQUIRE(t.at(1, 0) == 5);

    t.fill(3);

    REQUIRE(t.at(0, 0) == 3);
    REQUIRE(t.at(0, 1) == 3);
    REQUIRE(t.at(0, 2) == 3);
    REQUIRE(t.at(1, 0) == 3);
}

TEST_CASE("Transpose", "[Tensor]") {
    Tensor<int, 2> t{{1, 2, 3},
                     {4, 5, 6}};
    t = t.transpose();

    REQUIRE(t.shape_at(0) == 3);
    REQUIRE(t.shape_at(1) == 2);
    REQUIRE(t.at(0, 0) == 1);
    REQUIRE(t.at(0, 1) == 4);
    REQUIRE(t.at(1, 0) == 2);
    REQUIRE(t.at(1, 1) == 5);
    REQUIRE(t.at(2, 0) == 3);
    REQUIRE(t.at(2, 1) == 6);
}

TEST_CASE("Map", "[Tensor]") {
    SECTION("Square") {
        const Tensor<int, 2> t{{1, 2},
                               {3, 4}};

        const auto res = t.map([](const auto el) { return el * el; });
        REQUIRE(res.shape() == t.shape());
        REQUIRE(res.at(0, 0) == 1);
        REQUIRE(res.at(0, 1) == 4);
        REQUIRE(res.at(1, 0) == 9);
        REQUIRE(res.at(1, 1) == 16);
    }

    SECTION("To String") {
        const Tensor<int, 2> t{{1, 2},
                               {3, 4}};

        const auto res = t.map([](const auto el) { return std::to_string(el); });
        REQUIRE(res.shape() == t.shape());
        REQUIRE(res.at(0, 0) == "1");
        REQUIRE(res.at(0, 1) == "2");
        REQUIRE(res.at(1, 0) == "3");
        REQUIRE(res.at(1, 1) == "4");
    }
}

TEST_CASE("Stats", "[Tensor]") {
    const Tensor<double, 2> t{{3.0, 6.0},
                              {5.0, 4.0}};

    REQUIRE(t.min() == 3.0);
    REQUIRE(t.max() == 6.0);
    REQUIRE(t.mean() == 4.5);
    REQUIRE(t.var() == 1.25);
    REQUIRE(t.std() == 1.118033988749895);
}

TEST_CASE("Zeros", "[Tensor]") {
    constexpr Shape<2> shape{2, 2};
    auto t = zeros<int>(shape);
    REQUIRE(t.shape() == shape);
    REQUIRE(t.at(0, 0) == 0);
    REQUIRE(t.at(0, 1) == 0);
    REQUIRE(t.at(1, 0) == 0);
    REQUIRE(t.at(1, 1) == 0);
}

TEST_CASE("Ones", "[Tensor]") {
    constexpr Shape<2> shape{2, 2};
    auto t = ones<int>(shape);
    REQUIRE(t.shape() == shape);
    REQUIRE(t.at(0, 0) == 1);
    REQUIRE(t.at(0, 1) == 1);
    REQUIRE(t.at(1, 0) == 1);
    REQUIRE(t.at(1, 1) == 1);
}

TEST_CASE("Range", "[Tensor]") {
    SECTION("Stop") {
        auto t = range<int>(3);
        REQUIRE(t.shape_at(0) == 3);
        REQUIRE(t.at(0) == 0);
        REQUIRE(t.at(1) == 1);
        REQUIRE(t.at(2) == 2);
    }

    SECTION("Start-Stop") {
        auto t = range<int>(1, 3);
        REQUIRE(t.shape_at(0) == 2);
        REQUIRE(t.at(0) == 1);
        REQUIRE(t.at(1) == 2);
    }

    SECTION("Start-Stop-Step 1") {
        auto t = range<int>(1, 4, 2);
        REQUIRE(t.shape_at(0) == 2);
        REQUIRE(t.at(0) == 1);
        REQUIRE(t.at(1) == 3);
    }

    SECTION("Start-Stop-Step 2") {
        auto t = range<int>(1, 5, 2);
        REQUIRE(t.shape_at(0) == 2);
        REQUIRE(t.at(0) == 1);
        REQUIRE(t.at(1) == 3);
    }
}

TEST_CASE("Complex Numbers", "[Tensor]") {
    using namespace std::complex_literals;
    const Tensor<std::complex<double>, 2> t{{1.0 + 4i, 2.0 + 3i},
                                            {3.0 + 2i, 4.0 + 1i}};
    const auto t_real = t.real();
    const auto t_imag = t.imag();

    STATIC_REQUIRE(std::is_same_v<decltype(t_real)::ValueType, double>);
    REQUIRE(t_real.at(0, 0) == 1.0);
    REQUIRE(t_real.at(0, 1) == 2.0);
    REQUIRE(t_real.at(1, 0) == 3.0);
    REQUIRE(t_real.at(1, 1) == 4.0);

    STATIC_REQUIRE(std::is_same_v<decltype(t_imag)::ValueType, double>);
    REQUIRE(t_imag.at(0, 0) == 4.0);
    REQUIRE(t_imag.at(0, 1) == 3.0);
    REQUIRE(t_imag.at(1, 0) == 2.0);
    REQUIRE(t_imag.at(1, 1) == 1.0);
}

TEST_CASE("Tensor Equality", "[Tensor]") {
    Tensor<int, 1> t1{1, 2, 3};
    Tensor<int, 1> t2{1, 2, 3};
    Tensor<int, 1> t3{4, 5, 6};

    REQUIRE(is_equal(t1, t2));
    REQUIRE(!is_equal(t1, t3));
}
