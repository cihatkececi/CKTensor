#pragma once

#include <cassert>

#include "tensor.h"

#ifdef CKTENSOR_USE_MKL
#include "mkl.h"
#else
using CBLAS_LAYOUT = int;
using CBLAS_TRANSPOSE = int;
constexpr int CblasNoTrans = 0;
constexpr int CblasTrans = 1;
constexpr int CblasRowMajor = 0;
#endif


namespace ck::impl {


template<typename T>
struct GEMM {
    void operator()(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE TransA,
                    const CBLAS_TRANSPOSE TransB, const int M, const int N,
                    const int K, const float alpha, const T* A,
                    const int lda, const T* B, const int ldb,
                    const T beta, T* C, const int ldc) const {
        assert(Layout == CblasRowMajor && "Only row major matrices are supported.");

        if (beta != T{1}) {
            for (int i = 0; i < M * N; ++i) {
                C[i] *= beta;
            }
        }

        if (TransA == CblasNoTrans && TransB == CblasNoTrans) {
            // #pragma omp parallel for
            for (int i = 0; i < M; i++) {
                for (int k = 0; k < K; k++) {
                    for (int j = 0; j < N; j++) {
                        C[ldc * i + j] += alpha * A[lda * i + k] * B[ldb * k + j];
                    }
                }
            }
        }
        else if (TransA == CblasTrans && TransB == CblasNoTrans) {
            // #pragma omp parallel for
            for (int k = 0; k < K; k++) {
                for (int i = 0; i < M; i++) {
                    for (int j = 0; j < N; j++) {
                        C[ldc * i + j] += alpha * A[lda * k + i] * B[ldb * k + j];
                    }
                }
            }
        }
        else if (TransA == CblasNoTrans && TransB == CblasTrans) {
            // #pragma omp parallel for
            for (int i = 0; i < M; i++) {
                for (int j = 0; j < N; j++) {
                    for (int k = 0; k < K; k++) {
                        C[ldc * i + j] += alpha * A[lda * i + k] * B[ldb * j + k];
                    }
                }
            }
        }
        else {
            // #pragma omp parallel for
            for (int i = 0; i < M; i++) {
                for (int j = 0; j < N; j++) {
                    for (int k = 0; k < K; k++) {
                        C[ldc * i + j] += alpha * A[lda * k + i] * B[ldb * j + k];
                    }
                }
            }
        }
    }
};

#ifdef CKTENSOR_USE_MKL

template<>
struct GEMM<float> {
    template<typename... Args>
    void operator()(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE TransA,
                    const CBLAS_TRANSPOSE TransB, const MKL_INT M, const MKL_INT N,
                    const MKL_INT K, const float alpha, const float *A,
                    const MKL_INT lda, const float *B, const MKL_INT ldb,
                    const float beta, float *C, const MKL_INT ldc) const noexcept {
        cblas_sgemm(Layout, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    }
};

template<>
struct GEMM<double> {
    template<typename... Args>
    void operator()(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE TransA,
                    const CBLAS_TRANSPOSE TransB, const MKL_INT M, const MKL_INT N,
                    const MKL_INT K, const double alpha, const double *A,
                    const MKL_INT lda, const double *B, const MKL_INT ldb,
                    const double beta, double *C, const MKL_INT ldc) const noexcept {
        cblas_dgemm(Layout, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    }
};

template<>
struct GEMM<std::complex<float>> {
    template<typename... Args>
    void operator()(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE TransA,
                    const CBLAS_TRANSPOSE TransB, const MKL_INT M, const MKL_INT N,
                    const MKL_INT K, const void *alpha, const void *A,
                    const MKL_INT lda, const void *B, const MKL_INT ldb,
                    const void *beta, void *C, const MKL_INT ldc) const noexcept {
        cblas_cgemm(Layout, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    }
};

template<>
struct GEMM<std::complex<double>> {
    template<typename... Args>
    void operator()(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE TransA,
                    const CBLAS_TRANSPOSE TransB, const MKL_INT M, const MKL_INT N,
                    const MKL_INT K, const void *alpha, const void *A,
                    const MKL_INT lda, const void *B, const MKL_INT ldb,
                    const void *beta, void *C, const MKL_INT ldc) const noexcept {
        cblas_zgemm(Layout, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    }
};

#endif // CKTENSOR_USE_MKL

template<typename T, std::size_t TDims, typename U, std::size_t UDims>
struct MatMul {};


template<typename T, typename U>
struct MatMul<T, 2, U, 2> {
    auto operator()(const Tensor<T, 2>& lhs, const Tensor<U, 2>& rhs) const {
        assert(lhs.shape().rat(0) == rhs.shape().at(0) && "Incompatible shapes");

        using RetType = decltype(std::declval<T>() * std::declval<U>());

        if constexpr (!(std::is_same_v<T, RetType> && std::is_same_v<U, RetType>)) {
            return MatMul<RetType, 2, RetType, 2>{}(lhs, rhs);
        }
        else {
            auto ret = zeros<RetType, 2>({lhs.shape()[0], rhs.shape()[1]});

            GEMM<RetType>{}(CblasRowMajor, CblasNoTrans, CblasNoTrans, lhs.shape_at(0), rhs.shape_at(1),
                            lhs.shape_at(1),
                            1.0, lhs.data(), lhs.shape_at(1), rhs.data(), rhs.shape_at(1), 0.0, ret.data(),
                            ret.shape_at(1));

            return ret;
        }
    }
};

}
