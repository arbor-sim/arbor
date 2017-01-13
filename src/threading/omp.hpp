#pragma once

#if !defined(NMC_HAVE_OMP)
    #error "this header can only be loaded if NMC_HAVE_OMP is set"
#endif

#include <omp.h>
#include "parallel_stable_sort.h"

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
    using storage_class = std::vector<T>;
    storage_class data;
public :
    using iterator = typename storage_class::iterator;
    using const_iterator = typename storage_class::const_iterator;

    enumerable_thread_specific() {
        data = std::vector<T>(omp_get_max_threads());
    }

    enumerable_thread_specific(const T& init) {
        data = std::vector<T>(omp_get_max_threads(), init);
    }

    T& local() { return data[omp_get_thread_num()]; }
    const T& local() const { return data[omp_get_thread_num()]; }

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
        #pragma omp parallel for
        for(int i=left; i<right; ++i) {
            f(i);
        }
    }
};

template <typename RandomIt>
void sort(RandomIt begin, RandomIt end) {
    pss::parallel_stable_sort(begin, end);
}

template <typename RandomIt, typename Compare>
void sort(RandomIt begin, RandomIt end, Compare comp) {
    pss::parallel_stable_sort(begin, end ,comp);
}

template <typename Container>
void sort(Container& c) {
    pss::parallel_stable_sort(c.begin(), c.end());
}


template <typename T>
class parallel_vector {
    using value_type = T;
    std::vector<value_type> data_;
public:
    parallel_vector() = default;
    using iterator = typename std::vector<value_type>::iterator;
    using const_iterator = typename std::vector<value_type>::const_iterator;

    iterator begin() { return data_.begin(); }
    iterator end()   { return data_.end(); }

    const_iterator begin() const { return data_.begin(); }
    const_iterator end()   const { return data_.end(); }

    const_iterator cbegin() const { return data_.cbegin(); }
    const_iterator cend()   const { return data_.cend(); }

    void push_back (const value_type& val) {
        #pragma omp critical
        data_.push_back(val);
    }

    void push_back (value_type&& val) {
        #pragma omp critical
        data_.push_back(std::move(val));
    }
};

inline std::string description() {
    return "OpenMP";
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

constexpr bool multithreaded() { return true; }


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

