#pragma once

#include <memory>

#ifdef CKTENSOR_USE_MKL
#include "mkl.h"
#endif

namespace ck {

constexpr size_t default_alignment = 64;

// TODO: Cache memory blocks
#ifdef CKTENSOR_USE_MKL

template <typename T, size_t Alignment=default_alignment>
class TensorAllocator : public std::allocator<T> {
public:
    using size_type = size_t;
    using difference_type = ptrdiff_t;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;
    using value_type = T;

    template<typename U>
    struct rebind {
        using other = TensorAllocator<U, Alignment> ;
    };

    pointer allocate(size_type n) {
        return static_cast<pointer>(mkl_malloc(n * sizeof(T), Alignment));
    }

    constexpr void deallocate(pointer p, size_type) {
        mkl_free(p);
    }
};

#else

template <typename T, size_t Alignment=default_alignment>
class TensorAllocator : public std::allocator<T> {
public:
    using size_type = size_t;
    using difference_type = ptrdiff_t;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;
    using value_type = T;

    template<typename U>
    struct rebind {
        using other = TensorAllocator<U, Alignment> ;
    };

    pointer allocate(size_type n) {
        return static_cast<T*>(::operator new(n * sizeof(T), std::align_val_t(Alignment)));
    }

    constexpr void deallocate(pointer p, size_type n) {
        ::operator delete(p, std::align_val_t(std::align_val_t(Alignment)));
    }
};

#endif

}
