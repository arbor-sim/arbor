#pragma once

#if !defined(WITH_TBB)
    #error this header can only be loaded if WITH_TBB is set
#endif

#include <string>

#include <tbb/tbb.h>
#include <tbb/compat/thread>
#include <tbb/enumerable_thread_specific.h>

namespace nest {
namespace mc {
namespace threading {

template <typename T>
class enumerable_thread_specific;

struct parallel_for {
    template <typename F>
    static void apply(int left, int right, F f) {
        tbb::parallel_for(left, right, f);
    }
};

static
std::string description() {
    return "TBB";
}

struct timer {
    using time_point = tbb::tick_count;

    static
    inline time_point tic()
    {
        return tbb::tick_count::now();
    }

    static
    inline double toc(time_point t)
    {
        return (tic() - t).seconds();
    }
};

} // threading
} // mc
} // nest

