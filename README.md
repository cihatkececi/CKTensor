# CKTensor

A header only numeric library for C++ language.

It is a hobby project and a work in progress. So, please use with caution.

## Installation

CKTensor is a header only library. You only need the header files.

Uses cmake as a build system. You could download the source via git:

## Getting Started

The most important class is `ck::Tensor<T, Dims>`. There are operator overloads and functions for basic tensor operations.

A simple program looks like this:

```c++
#include <iostream>
#include "cktensor.h"

using namespace ck;

int main() {
    Tensor<int, 2> t{{1, 2},
                         {3, 4}};

    std::cout << t(0, 1) << std::endl;  // 2
    std::cout << t(1, 0) << std::endl;  // 3

    auto t2 = ones<int, 2>({2, 2});
    auto sum = t + t2;

    std::cout << sum << std::endl;  // 3, 4,
                                    // 5, 6,

    return 0;
}
```

