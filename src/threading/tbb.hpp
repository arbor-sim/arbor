#pragma once

#if !defined(NMC_HAVE_TBB)
    #error this header can only be loaded if NMC_HAVE_TBB is set
#endif

#include <string>

#include <tbb/tbb.h>
#include <tbb/compat/thread>
#include <tbb/enumerable_thread_specific.h>

namespace nest {
namespace mc {
namespace threading {

template <typename T>
using enumerable_thread_specific = tbb::enumerable_thread_specific<T>;

struct parallel_for {
    template <typename F>
    static void apply(int left, int right, F f) {
        tbb::parallel_for(left, right, f);
    }
};

inline std::string description() {
    return "TBB";
}

struct timer {
    using time_point = tbb::tick_count;

    static inline time_point tic() {
        return tbb::tick_count::now();
    }

    static inline double toc(time_point t) {
        return (tic() - t).seconds();
    }

    static inline double difference(time_point b, time_point e) {
        return (e-b).seconds();
    }
};

constexpr bool multithreaded() { return true; }

template <typename T>
using parallel_vector = tbb::concurrent_vector<T>;

using task_group = tbb::task_group;

template <typename RandomIt>
void sort(RandomIt begin, RandomIt end) {
    tbb::parallel_sort(begin, end);
}

template <typename RandomIt, typename Compare>
void sort(RandomIt begin, RandomIt end, Compare comp) {
    tbb::parallel_sort(begin, end, comp);
}

template <typename Container>
void sort(Container& c) {
    tbb::parallel_sort(c.begin(), c.end());
}

} // threading
} // mc
} // nest

namespace tbb {
    /// comparison operator for tbb::tick_count type
    /// returns true iff time stamp l occurred before timestamp r
    inline bool operator< (tbb::tick_count l, tbb::tick_count r) {
        return (l-r).seconds() < 0.;
    }
}

