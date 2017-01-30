#pragma once

#if !defined(NMC_HAVE_SERIAL)
    #error "this header can only be loaded if NMC_HAVE_SERIAL is set"
#endif

#include <algorithm>
#include <array>
#include <chrono>
#include <string>
#include <vector>

namespace nest {
namespace mc {
namespace threading {

///////////////////////////////////////////////////////////////////////
// types
///////////////////////////////////////////////////////////////////////
template <typename T>
class enumerable_thread_specific {
    std::array<T, 1> data;

public :
    using iterator = typename std::array<T, 1>::iterator;
    using const_iterator = typename std::array<T, 1>::const_iterator;

    enumerable_thread_specific() = default;

    enumerable_thread_specific(const T& init) :
        data{init}
    {}

    enumerable_thread_specific(T&& init) :
        data{std::move(init)}
    {}

    T& local() { return data[0]; }
    const T& local() const { return data[0]; }

    auto size() -> decltype(data.size()) const { return data.size(); }

    iterator begin() { return data.begin(); }
    iterator end()   { return data.end(); }

    const_iterator begin() const { return data.begin(); }
    const_iterator end()   const { return data.end(); }

    const_iterator cbegin() const { return data.cbegin(); }
    const_iterator cend()   const { return data.cend(); }
};


///////////////////////////////////////////////////////////////////////
// algorithms
///////////////////////////////////////////////////////////////////////
struct parallel_for {
    template <typename F>
    static void apply(int left, int right, F f) {
        for(int i=left; i<right; ++i) {
            f(i);
        }
    }
};

template <typename RandomIt>
void sort(RandomIt begin, RandomIt end) {
    std::sort(begin, end);
}

template <typename RandomIt, typename Compare>
void sort(RandomIt begin, RandomIt end, Compare comp) {
    std::sort(begin, end, comp);
}

template <typename Container>
void sort(Container& c) {
    std::sort(c.begin(), c.end());
}

template <typename T>
using parallel_vector = std::vector<T>;

inline std::string description() {
    return "serial";
}

struct timer {
    using time_point = std::chrono::time_point<std::chrono::system_clock>;

    static inline time_point tic() {
        return std::chrono::system_clock::now();
    }

    static inline double toc(time_point t) {
        return std::chrono::duration<double>(tic() - t).count();
    }

    static inline double difference(time_point b, time_point e) {
        return std::chrono::duration<double>(e-b).count();
    }
};

constexpr bool multithreaded() { return false; }

/// Proxy for tbb task group.
/// The tbb version launches tasks asynchronously, returning control to the
/// caller. The serial version implemented here simply runs the task, before
/// returning control, effectively serializing all asynchronous calls.
class task_group {
public:
    task_group() = default;

    template<typename Func>
    void run(const Func& f) {
        f();
    }

    template<typename Func>
    void run_and_wait(const Func& f) {
        f();
    }

    void wait()
    {}

    bool is_canceling() {
        return false;
    }

    void cancel()
    {}
};

} // threading
} // mc
} // nest

