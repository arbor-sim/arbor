#include <chrono>
#include <iostream>

#include <omp.h>

#include <Vector.hpp>

using value_type = double;
using size_type  = std::size_t;

using clock_type    = std::chrono::high_resolution_clock;
using duration_type = std::chrono::duration<double>;

// let's use 256-bit byte alignment
// aka. the alignment of an AVX register
constexpr std::size_t alignment() { return 256/8; }

using namespace memory;
template <typename T>
using vector
    = Array<T, HostCoordinator<T, Allocator<T, impl::AlignedPolicy<alignment()>>>>;

template <typename T>
void triad(vector<T>      & a,
           vector<T> const& b,
           vector<T> const& c,
           T scalar)
{
    auto const n = a.size();
    #pragma ivdep
    #pragma vector nontemporal
    for(auto i=size_type{0}; i<n; ++i) {
        a[i] = b[i] + scalar * c[i];
    }
}

template <typename T>
void scale(vector<T>      & a,
           vector<T> const& b,
           T scalar)
{
    auto const n = a.size();
    #pragma ivdep
    #pragma vector nontemporal
    for(auto i=size_type{0}; i<n; ++i) {
        a[i] = scalar * b[i];
    }
}

template <typename T>
void copy(vector<T>      & a,
          vector<T> const& b)
{
    auto const n = a.size();
    #pragma ivdep
    #pragma vector nontemporal
    for(auto i=size_type{0}; i<n; ++i) {
        a[i] = b[i];
    }
}

template <typename T>
void   add(vector<T>      & a,
           vector<T> const& b,
           vector<T> const& c)
{
    auto const n = a.size();
    #pragma ivdep
    #pragma vector nontemporal
    for(auto i=size_type{0}; i<n; ++i) {
        a[i] = b[i] + c[i];
    }
}

template <typename T>
void init(vector<T> & a,
          vector<T> & b,
          vector<T> & c)
{
    auto const n = a.size();
    #pragma ivdep
    #pragma vector nontemporal
    for(auto i=size_type{0}; i<n; ++i) {
        a[i] = T{1};
        b[i] = T{2};
        c[i] = T{3};
    }
}

int main(int argc, char **argv) {
    size_type pow = 22;
    if(argc>1) {
        pow = std::stod(argv[1]);
    }
    auto const N = 3 * (size_type{1} << pow);
    auto num_trials = 5;

    std::cout << "------------------------------------" << std::endl;
    std::cout << "arrays of length " << N << " == 3*2^" << pow << std::endl;
    std::cout << "threads          " << omp_get_max_threads() << std::endl;

    #pragma omp parallel
    {
        auto num_threads = omp_get_num_threads();
        auto n = N/num_threads;

        // create arrays
        vector<value_type> a(n);
        vector<value_type> b(n);
        vector<value_type> c(n);

        auto scalar = value_type{2};

        // initialize and touch memory
        init(a, b, c);

        // do timed runs
        auto triad_time = 0.;
        auto copy_time  = 0.;
        auto add_time  = 0.;
        auto scale_time  = 0.;
        for(auto i=0; i<num_trials; ++i) {
            {
                #pragma omp barrier
                auto start = clock_type::now();
                triad(a, b, c, scalar);
                #pragma omp barrier
                triad_time += duration_type(clock_type::now()-start).count();
            }
            {
                #pragma omp barrier
                auto start = clock_type::now();
                copy(a, b);
                #pragma omp barrier
                copy_time += duration_type(clock_type::now()-start).count();
            }
            {
                #pragma omp barrier
                auto start = clock_type::now();
                add(a, b, c);
                #pragma omp barrier
                add_time += duration_type(clock_type::now()-start).count();
            }
            {
                #pragma omp barrier
                auto start = clock_type::now();
                scale(a, b, scalar);
                #pragma omp barrier
                scale_time += duration_type(clock_type::now()-start).count();
            }
        }

        auto bytes_per_array = sizeof(value_type)*N*num_trials;
        auto copy_BW   = 2 * bytes_per_array / copy_time;
        auto scale_BW  = 2 * bytes_per_array / scale_time;
        auto add_BW    = 3 * bytes_per_array / add_time;
        auto triad_BW  = 3 * bytes_per_array / triad_time;

        #pragma omp master
        {
            std::cout << "triad " << triad_BW/1.e9 << " GB/s" << std::endl;
            std::cout << "copy  " << copy_BW/1.e9  << " GB/s" << std::endl;
            std::cout << "add   " << add_BW/1.e9   << " GB/s" << std::endl;
            std::cout << "scale " << scale_BW/1.e9 << " GB/s" << std::endl;
        }
    }

    return 0;
}

