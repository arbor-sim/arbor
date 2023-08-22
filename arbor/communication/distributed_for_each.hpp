#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <type_traits>
#include <utility>

#include "distributed_context.hpp"
#include "util/range.hpp"

namespace arb {

namespace impl {
template <class FUNC, typename... T, std::size_t... Is>
void for_each_in_tuple(FUNC&& func, std::tuple<T...>& t, std::index_sequence<Is...>) {
    (func(Is, std::get<Is>(t)), ...);
}

template <class FUNC, typename... T>
void for_each_in_tuple(FUNC&& func, std::tuple<T...>& t) {
    for_each_in_tuple(func, t, std::index_sequence_for<T...>());
}

template <class FUNC, typename... T1, typename... T2, std::size_t... Is>
void for_each_in_tuple_pair(FUNC&& func,
    std::tuple<T1...>& t1,
    std::tuple<T2...>& t2,
    std::index_sequence<Is...>) {
    (func(Is, std::get<Is>(t1), std::get<Is>(t2)), ...);
}

template <class FUNC, typename... T1, typename... T2>
void for_each_in_tuple_pair(FUNC&& func, std::tuple<T1...>& t1, std::tuple<T2...>& t2) {
    for_each_in_tuple_pair(func, t1, t2, std::index_sequence_for<T1...>());
}

}  // namespace impl



/*
 * Iterate through multiple ranges from each distributed rank. Only usable with trvially copyable value types.
 * The provided function objects is expected to be callable with the following signature:
 * (const util::range<util::range<ARGS>::value_type*>&...) -> void
 * Given 'n' distributed ranks, the function will be called 'n' times with data from each rank.
 * There is no guaranteed order.
 */
template <typename FUNC, typename... ARGS>
void distributed_for_each(FUNC&& func,
    const distributed_context& distributed,
    const util::range<ARGS>&... args) {

    static_assert(sizeof...(args) > 0);
    auto arg_tuple = std::forward_as_tuple(args...);

    struct vec_info {
        std::size_t offset;  // offset in bytes
        std::size_t size;    // size in bytes
    };

    std::array<vec_info, sizeof...(args)> info;
    std::size_t buffer_size = 0;

    // Compute offsets in bytes for each vector when placed in common buffer
    {
        std::size_t offset = info.size() * sizeof(vec_info);
        impl::for_each_in_tuple(
            [&](std::size_t i, auto&& vec) {
                using T = typename std::remove_reference_t<decltype(vec)>::value_type;
                static_assert(std::is_trivially_copyable_v<T>);
                static_assert(alignof(std::max_align_t) >= alignof(T));
                static_assert(alignof(std::max_align_t) % alignof(T) == 0);

                // make sure alignment of offset fulfills requirement
                const auto alignment_excess = offset % alignof(T);
                offset += alignment_excess > 0 ? alignof(T) - (alignment_excess) : 0;

                const auto size_in_bytes = vec.size() * sizeof(T);

                info[i].size = size_in_bytes;
                info[i].offset = offset;

                buffer_size = offset + size_in_bytes;
                offset += size_in_bytes;
            },
            arg_tuple);
    }

    // compute maximum buffer size between ranks, such that we only allocate once
    const std::size_t max_buffer_size = distributed.max(buffer_size);

    std::tuple<util::range<typename std::remove_reference_t<decltype(args)>::value_type*>...>
        ranges;

    if (max_buffer_size == info.size() * sizeof(vec_info)) {
        // if all empty, call function with empty ranges for each step and exit
        impl::for_each_in_tuple_pair(
            [&](std::size_t i, auto&& vec, auto&& r) {
                using T = typename std::remove_reference_t<decltype(vec)>::value_type;
                r = util::range<T*>(nullptr, nullptr);
            },
            arg_tuple,
            ranges);

        for (std::size_t step = 0; step < distributed.size(); ++step) { std::apply(func, ranges); }
        return;
    }

    // use malloc for std::max_align_t alignment
    auto deleter = [](char* ptr) { std::free(ptr); };
    std::unique_ptr<char[], void (*)(char*)> buffer((char*)std::malloc(max_buffer_size), deleter);
    std::unique_ptr<char[], void (*)(char*)> recv_buffer(
        (char*)std::malloc(max_buffer_size), deleter);

    // copy offset and size info to front of buffer
    std::memcpy(buffer.get(), info.data(), info.size() * sizeof(vec_info));

    // copy each vector to each location in buffer
    impl::for_each_in_tuple(
        [&](std::size_t i, auto&& vec) {
            using T = typename std::remove_reference_t<decltype(vec)>::value_type;
            std::copy(vec.begin(), vec.end(), (T*)(buffer.get() + info[i].offset));
        },
        arg_tuple);


    const auto my_rank = distributed.id();
    const auto left_rank = my_rank == 0 ? distributed.size() - 1 : my_rank - 1;
    const auto right_rank = my_rank == distributed.size() - 1 ? 0 : my_rank + 1;

    // exchange buffer in ring pattern and apply function at each step
    for (std::size_t step = 0; step < distributed.size() - 1; ++step) {
        // always expect to recieve the max size but send actual size. MPI_recv only expects a max
        // size, not the actual size.
        const auto current_info = (const vec_info*)buffer.get();

        auto request = distributed.send_recv_nonblocking(max_buffer_size,
            recv_buffer.get(),
            right_rank,
            current_info[info.size() - 1].offset + current_info[info.size() - 1].size,
            buffer.get(),
            left_rank,
            0);

        // update ranges
        impl::for_each_in_tuple_pair(
            [&](std::size_t i, auto&& vec, auto&& r) {
                using T = typename std::remove_reference_t<decltype(vec)>::value_type;
                r = util::range<T*>((T*)(buffer.get() + current_info[i].offset),
                    (T*)(buffer.get() + current_info[i].offset + current_info[i].size));
            },
            arg_tuple,
            ranges);

        // call provided function with ranges pointing to current buffer
        std::apply(func, ranges);

        request.finalize();
        buffer.swap(recv_buffer);
    }

    // final step does not require any exchange
    const auto current_info = (const vec_info*)buffer.get();
    impl::for_each_in_tuple_pair(
        [&](std::size_t i, auto&& vec, auto&& r) {
            using T = typename std::remove_reference_t<decltype(vec)>::value_type;
            r = util::range<T*>((T*)(buffer.get() + current_info[i].offset),
                (T*)(buffer.get() + current_info[i].offset + current_info[i].size));
        },
        arg_tuple,
        ranges);

    // call provided function with ranges pointing to current buffer
    std::apply(func, ranges);
}

}  // namespace arb
