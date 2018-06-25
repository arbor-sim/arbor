#pragma once

#include <atomic>
#include <string>

#include <tbb/tbb.h>
#include <tbb/tbb_stddef.h>
#include <tbb/compat/thread>
#include <tbb/enumerable_thread_specific.h>

namespace arb {
namespace threading {
inline namespace tbb {

template <typename T>
using enumerable_thread_specific = ::tbb::enumerable_thread_specific<T>;

struct parallel_for {
    template <typename F>
    static void apply(int left, int right, F f) {
        ::tbb::parallel_for(left, right, f);
    }
};

inline std::string description() {
    return "TBBv" + std::to_string(::tbb::TBB_runtime_interface_version());
}

constexpr bool multithreaded() { return true; }

template <typename T>
using parallel_vector = ::tbb::concurrent_vector<T>;

using task_group = ::tbb::task_group;

inline
std::size_t thread_id() {
    static std::atomic<std::size_t> num_threads(0);
    thread_local std::size_t thread_id = num_threads++;
    return thread_id;
}

template <typename RandomIt>
void sort(RandomIt begin, RandomIt end) {
    ::tbb::parallel_sort(begin, end);
}

template <typename RandomIt, typename Compare>
void sort(RandomIt begin, RandomIt end, Compare comp) {
    ::tbb::parallel_sort(begin, end, comp);
}

template <typename Container>
void sort(Container& c) {
    ::tbb::parallel_sort(c.begin(), c.end());
}

} // namespace tbb
} // namespace threading
} // namespace arb

