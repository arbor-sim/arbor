#pragma once

#if !defined(WITH_SERIAL)
    #error "this header can only be loaded if WITH_SERIAL is set"
#endif

#include <array>
#include <chrono>
#include <string>

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

    auto begin() -> decltype(data.begin()) { return data.begin(); }
    auto end()   -> decltype(data.end())   { return data.end(); }

    auto cbegin() -> decltype(data.cbegin()) const { return data.cbegin(); }
    auto cend()   -> decltype(data.cend())   const { return data.cend(); }
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


} // threading
} // mc
} // nest

