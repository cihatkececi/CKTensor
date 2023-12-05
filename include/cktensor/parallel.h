#pragma once

#include <thread>

#include "cktensor/tensor.h"

namespace ck::par {

namespace impl {
std::vector<size_t> generate_indices(std::size_t num_elem, std::size_t num_workers) {
    std::size_t num_elem_per_thread = num_elem / num_workers;
    std::size_t remainder = num_elem % num_workers;

    std::vector<std::size_t> indices;
    indices.reserve(num_workers + 1);

    for (std::size_t i = 0; i < num_elem; i += num_elem_per_thread) {
        indices.push_back(i);

        if (remainder > 0) {
            i++;
            remainder--;
        }
    }
    for (std::size_t i = indices.size(); i <= num_workers; i++)
        indices.push_back(num_elem);

    return indices;
}
}

template<typename F, typename T, std::size_t Dims>
auto map(F f, const Tensor<T, Dims>& t, size_t num_workers) {
    Tensor<std::invoke_result_t<F, T>, Dims> result{t.shape()};

    auto indices = impl::generate_indices(t.num_elem(), num_workers);

    const auto worker_fn = [&](std::size_t id) {
        for (std::size_t i = indices[id]; i < indices[id + 1]; i++) {
            result.data()[i] = f(t.data()[i]);
        }
    };

    std::vector<std::thread> threads;
    threads.reserve(num_workers);

    for (std::size_t i = 0; i < num_workers; i++)
        threads.emplace_back(worker_fn, i);


    for (auto& thread: threads)
        thread.join();

    return result;
}

// TODO: Implement reduce functions: add, mult, max, min, mean, var, std, all, any

template<typename T, std::size_t Dims>
T reduce_add(const Tensor<T, Dims>& t, size_t num_workers) {
    auto indices = impl::generate_indices(t.num_elem(), num_workers);
    std::vector<T> worker_results(num_workers);

    const auto worker_fn = [&](std::size_t id) {
        worker_results[id] = std::accumulate(&t.data()[indices[id]], &t.data()[indices[id+1]], T{}, std::plus<>{});
    };

    std::vector<std::thread> threads;
    threads.reserve(num_workers);

    for (std::size_t i = 0; i < num_workers; i++)
        threads.emplace_back(worker_fn, i);


    for (auto& thread: threads)
        thread.join();

    T result = std::accumulate(worker_results.begin(), worker_results.end(), T{}, std::plus<>{});

    return result;
}

}
