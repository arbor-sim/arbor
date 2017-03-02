#include <cassert>

#include <numeric>

#include <util/span.hpp>

#include <memory/memory.hpp>
#include <memory/wrappers.hpp>

#include <cub/cub.cuh>

#include "managed_ptr.hpp"


////////////////////////////////////////////////////
////////////////////////////////////////////////////
using namespace nest::mc;
void test_tricky();
int main() {
    test_tricky();
    return 0;
}
////////////////////////////////////////////////////
////////////////////////////////////////////////////

template <typename T>
class gpu_stack {
    using allocator = managed_allocator<T>;
    int capacity_;
    int size_;
    T* data_;

public:

    gpu_stack(int capacity):
        capacity_(capacity), size_(0u)
    {
        data_ = allocator().allocate(capacity_);
    }

    ~gpu_stack() {
        allocator().deallocate(data_, capacity_);
    }

    __device__
    void push_back(const T& value, bool do_push) {
        if (do_push) {
            int position = atomicAdd(&size_, 1);

            // It is possible that size_>capacity_. In this case, only capacity_
            // entries are stored, and additional values are lost. The size_
            // will contain the total number of attempts to push,
            if (position<capacity_) {
                printf("thread %4d -- writing at %4d\n", int(threadIdx.x), position);
                data_[position] = value;
            }
        }
    }

    __host__ __device__
    int size() const {
        return size_;
    }

    __host__ __device__
    int capacity() const {
        return capacity_;
    }

    T* begin() {
        return data_;
    }
    const T* begin() const {
        return data_;
    }

    T* end() {
        return data_ + size_;
    }
    const T* end() const {
        return data_ + size_;
    }
};

template <typename T, typename I, typename Stack>
__global__
void kernel(
    float t, float t_prev, int size,
    Stack& stack,
    uint8_t* is_spiking,
    const T* values,
    T* prev_values,
    const T* thresholds)
{
    int i = threadIdx.x + blockIdx.x*blockDim.x;

    bool crossed = false;
    float crossing_time;

    if (i<size) {
        // Test for threshold crossing
        const auto v_prev = prev_values[i];
        const auto v = values[i];
        const auto thresh = thresholds[i];

        if (!is_spiking[i]) {
            if (v>=thresh) {
                // The threshold has been passed, so estimate the time using
                // linear interpolation
                auto pos = (thresh - v_prev)/(v - v_prev);
                crossing_time = t_prev + pos*(t - t_prev);

                is_spiking[i] = 1;
                crossed = true;
            }
        }
        else if (v<thresh) {
            is_spiking[i]=0;
        }

        prev_values[i] = values[i];
    }

    stack.push_back({i, crossing_time}, crossed);

    // the first thread updates the buffer size
    if (i==0) {
        printf("%d is stack size\n", stack.size());
    }
}

//
// type that manages threshold crossing checker on the GPU
//
template <typename T, typename I, Time>
struct threshold_watcher {
    using value_type = T;
    using index_type = I;
    using time_type = Time;

    struct crossing_type {
        index_type index;
        time_type time;
    };

    using stack_type = gpu_stack<crossing_type>;

    template <typename U>
    using array = memory::device_vector<U>;

    template <typename U>
    using array_view = typename array<U>::const_view_type;

    threshold_watcher(array_view<T> values, const std::vector<T>& thresholds, T t):
        values_(values),
        prev_values_(values),
        thresholds_(memory::make_const_view(thresholds)),
        is_spiking_(size(), 0),
        t_prev_(t),
        stack_(make_managed_ptr<stack_type>(10*values.size()))  // initialize stack with buffer 10 crossings on each value
    {
        // calculate the initial spiking state on the host
        auto v = memory::on_host(values);
        auto spiking = std::vector<uint8_t>(values.size());
        for (auto i: util::make_span(0u, values.size())) {
            spiking[i] = v[i] < thresholds[i] ? 0 : 1;
        }

        // copy the initial spiking state to device memory
        is_spiking_ = memory::on_gpu(spiking);
    }

    void test_for_crossings(T t) {
        EXPECTS(t_prev_<t);

        constexpr int block_dim = 128;
        const int n = size();
        const int grid_dim = (n+block_dim-1)/block_dim;
        kernel<T, I, stack_type><<<grid_dim, block_dim>>>(
            t, t_prev_, size(),
            *stack_,
            is_spiking_.data(),
            values_.data(),
            prev_values_.data(),
            thresholds_.data());

        t_prev_ = t;
    }

    /// returns a vector that contains the current set of crossings
    std::vector<crossing_type> get_crossings() const {
        return std::vector<crossing_type>(stack_->begin(), stack_->end());
    }

    /// Empties the spike buffer
    /// All recorded spike information will be lost.
    void reset() {
        return stack_->empty();
    }

    /// returns the number of threshold values that are being watched
    std::size_t size() const {
        return thresholds_.size();
    }

    array_view<T> values_;

    array<T> prev_values_;
    array<T> thresholds_;
    array<uint8_t> is_spiking_;
    T t_prev_;
    managed_ptr<stack_type> stack_;
};

void test_tricky() {
    using watcher = threshold_watcher<float, int>;

    constexpr auto N = 24;
    auto values = memory::device_vector<float>(N, 0);
    auto thresholds = std::vector<float>(N, 0.5);

    EXPECTS(values.size()==N);
    EXPECTS(thresholds.size()==N);

    auto w = watcher(values, thresholds, 0);

    memory::fill(values, 1); w.test(1);
    memory::fill(values, 0); w.test(2);
    memory::fill(values, 2); w.test(3);
    cudaDeviceSynchronize();

    auto crossings = w.get_crossings();
    std::cout << "CROSSINGS (" << crossings.size() << ")\n";
    for(auto c: crossings) std::cout << "  " << c.index << " -- " << c.time << "\n";
}

