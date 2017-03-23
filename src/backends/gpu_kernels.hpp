#pragma once

#include <cstdint>
#include <cfloat>

#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace nest {
namespace mc {
namespace gpu {

namespace impl {
    __host__ __device__
    constexpr inline int block_dim() {
        return 32;
    }

    __host__ __device__
    constexpr inline int load_width() {
        return 32;
    }

    __host__ __device__
    constexpr inline int matrix_padding() {
        return load_width();
    }

    __host__ __device__
    constexpr inline int block_count(int n, int block_size) {
        return (n+block_size-1)/block_size;
    }

    inline int padded_size (int n, int block_dim) {
        const int over = n%block_dim;
        return over ? n+block_dim-over: n;
    }

    template <typename T> __host__ __device__ constexpr T npos() { return T(); }
    template <> __host__ __device__ constexpr char npos<char>() { return CHAR_MAX; }
    template <> __host__ __device__ constexpr unsigned char npos<unsigned char>() { return UCHAR_MAX; }
    template <> __host__ __device__ constexpr short npos<short>() { return SHRT_MAX; }
    template <> __host__ __device__ constexpr int npos<int>() { return INT_MAX; }
    template <> __host__ __device__ constexpr long npos<long>() { return LONG_MAX; }
    template <> __host__ __device__ constexpr float npos<float>() { return FLT_MAX; }
    template <> __host__ __device__ constexpr double npos<double>() { return DBL_MAX; }
    template <> __host__ __device__ constexpr unsigned short npos<unsigned short>() { return USHRT_MAX; }
    template <> __host__ __device__ constexpr unsigned int npos<unsigned int>() { return UINT_MAX; }
    template <> __host__ __device__ constexpr unsigned long npos<unsigned long>() { return ULONG_MAX; }
    template <> __host__ __device__ constexpr long long npos<long long>() { return LLONG_MAX; }

    template <typename T>
    __host__ __device__
    bool is_npos(T v) {
        return v == npos<T>();
    }

    template <typename Seq>
    void print_vec(std::string name, const Seq& vec) {
        std::cout << name << ": ";
        for (auto v: vec) {
            if (is_npos(v)) std::cout << " *";
            else            std::cout << " " << v;
        }
        std::cout << "\n";
    }

}

/// GPU implementation of Hines Matrix solver.
/// Flat format
template <typename T, typename I>
__global__
void solve_matrix_flat(
    T* rhs, T* d, const T* u, const I* p, const I* cell_index, int num_mtx)
{
    auto tid = threadIdx.x + blockDim.x*blockIdx.x;

    if (tid<num_mtx) {
        // get range of this thread's cell matrix
        auto first = cell_index[tid];
        auto last  = cell_index[tid+1];

        // backward sweep
        for(auto i=last-1; i>first; --i) {
            auto factor = u[i] / d[i];
            d[p[i]]   -= factor * u[i];
            rhs[p[i]] -= factor * rhs[i];
        }

        rhs[first] /= d[first];

        // forward sweep
        for(auto i=first+1; i<last; ++i) {
            rhs[i] -= u[i] * rhs[p[i]];
            rhs[i] /= d[i];
        }
    }
}

/// GPU implementation of Hines Matrix solver.
/// Block-interleaved format
template <typename T, typename I, int BlockWidth>
__global__
void solve_matrix_interleaved(
    T* rhs, T* d, const T* u, const I* p, const I* sizes,
    int padded_size, int num_mtx)
{
    auto tid = threadIdx.x + blockDim.x*blockIdx.x;

    if(tid < num_mtx) {
        auto block       = tid/BlockWidth;
        auto block_start = block*BlockWidth;
        auto block_lane  = tid - block_start;

        // get range of this thread's cell matrix
        auto first = block_start*padded_size + block_lane;
        auto last  = first + BlockWidth*(sizes[tid]-1);

        // backward sweep
        for(auto i=last; i>first; i-=BlockWidth) {
            auto factor = u[i] / d[i];
            d[p[i]]   -= factor * u[i];
            rhs[p[i]] -= factor * rhs[i];
        }

        __syncthreads();
        rhs[first] /= d[first];

        // forward sweep
        for(auto i=first+BlockWidth; i<=last; i+=BlockWidth) {
            rhs[i] -= u[i] * rhs[p[i]];
            rhs[i] /= d[i];
        }
    }
}

// Data is to be interleaved into blocks of width BlockWidth.
// The kernel assigns LoadWidth threads to each lane in the block.
// Hence each thread block is responsible for loading a single block
// of interleaved matrices.

template <typename T, typename I, int BlockWidth, int LoadWidth, int Threads>
__global__
void interleave(
    const T* in, T* out, const I* sizes, const I* starts, int padded_size, int num_mtx)
{
    static_assert(BlockWidth*LoadWidth==Threads, "");

    __shared__ T buffer[Threads];

    const auto tid = threadIdx.x + blockIdx.x*blockDim.x;
    const auto lid = threadIdx.x;

    const auto mtx_id   = tid/LoadWidth;
    const auto mtx_lane = tid - mtx_id*LoadWidth;

    const auto blk_id   = tid/(BlockWidth*LoadWidth);
    const auto blk_row  = lid/BlockWidth;
    const auto blk_lane = lid - blk_row*BlockWidth;

    const auto blk_pos  = LoadWidth*blk_lane + blk_row;

    const bool do_load  = mtx_id<num_mtx;

    // only threads that participate in loading access starts and sizes arrays
    auto load_pos  = do_load? starts[mtx_id] + mtx_lane     : 0;
    const auto end = do_load? starts[mtx_id] + sizes[mtx_id]: 0;
    auto store_pos = blk_id*BlockWidth*padded_size + (blk_row*BlockWidth + blk_lane);

    for (auto i=0; i<padded_size; i+=LoadWidth) {
        auto loaded = impl::npos<T>();
        if (do_load && load_pos<end) {
            loaded = in[load_pos];
        }
        buffer[lid] = loaded;
        __syncthreads();
        if (i+blk_row<padded_size) {
            out[store_pos] = buffer[blk_pos];
        }
        load_pos  += LoadWidth;
        store_pos += LoadWidth*BlockWidth;
    }
}

// host side wrapper for the interleave kernel
template <typename T, typename I, int BlockWidth, int LoadWidth>
void interleave(const T* in, T* out, const I* sizes, const I* starts, int padded_size, int num_mtx)
{
    constexpr int Threads = BlockWidth*LoadWidth;
    const int blocks = impl::block_count(num_mtx, BlockWidth);

    interleave<T, I, BlockWidth, LoadWidth, Threads>
        <<<blocks, Threads>>> (in, out, sizes, starts, padded_size, num_mtx);
}

// A helper that performs the interleave operation on host memory.
template <typename T, typename I>
std::vector<T> interleave_host(
        const std::vector<T>& in,
        const std::vector<I>& sizes,
        const std::vector<I>& starts,
        int block_width, int num_mtx, int padded_length)
{
    auto num_blocks = impl::block_count(num_mtx, block_width);
    std::vector<T> out(num_blocks*block_width*padded_length, impl::npos<T>());
    for (auto mtx: util::make_span(0, num_mtx)) {
        auto block = mtx/block_width;
        auto lane  = mtx%block_width;

        auto len = sizes[mtx];
        auto src = starts[mtx];
        auto dst = block*(block_width*padded_length) + lane;
        for (auto i: util::make_span(0, len)) {
            out[dst] = in[src+i];
            dst += block_width;
        }
    }
    return out;
};

template <typename T, typename I, int BlockWidth, int LoadWidth, int THREADS>
__global__
void reverse_interleave(
    const T* in, T* out, const I* sizes, const I* starts, int padded_size, int num_mtx)
{
    static_assert(BlockWidth*LoadWidth==THREADS, "");

    __shared__ T buffer[THREADS];

    const auto tid = threadIdx.x + blockIdx.x*blockDim.x;
    const auto lid = threadIdx.x;

    const auto mtx_id   = tid/LoadWidth;
    const auto mtx_lane = tid - mtx_id*LoadWidth;

    const auto blk_id   = tid/(BlockWidth*LoadWidth);
    const auto blk_row  = lid/BlockWidth;
    const auto blk_lane = lid - blk_row*BlockWidth;

    const auto blk_pos  = LoadWidth*blk_lane + blk_row;

    const bool do_store = mtx_id<num_mtx;

    // only threads that participate in storing access starts and sizes arrays
    auto store_pos = do_store? starts[mtx_id] + mtx_lane     : 0;
    const auto end = do_store? starts[mtx_id] + sizes[mtx_id]: 0;
    auto load_pos  = blk_id*BlockWidth*padded_size + (blk_row*BlockWidth + blk_lane);

    for (auto i=0; i<padded_size; i+=LoadWidth) {
        auto loaded = impl::npos<T>();
        if (i+blk_row<padded_size) {
            loaded = in[load_pos];
        }
        buffer[blk_pos] = loaded;
        __syncthreads();
        if (do_store && store_pos<end) {
            out[store_pos] = buffer[lid];
        }
        load_pos  += LoadWidth*BlockWidth;
        store_pos += LoadWidth;
    }
}

template <typename T, typename I, int BlockWidth, int LoadWidth>
void reverse_interleave(const T* in, T* out, const I* sizes, const I* starts, int padded_size, int num_mtx)
{
    constexpr int Threads = BlockWidth*LoadWidth;
    const int blocks = impl::block_count(num_mtx, BlockWidth);

    reverse_interleave<T, I, BlockWidth, LoadWidth, Threads>
        <<<blocks, Threads>>> (in, out, sizes, starts, padded_size, num_mtx);
}

/// GPU implementatin of Hines matrix assembly
/// Flat layout
/// For a given time step size dt
///     - use the precomputed alpha and alpha_d values to construct the diagonal
///       and off diagonal of the symmetric Hines matrix.
///     - compute the RHS of the linear system to solve
template <typename T, typename I>
__global__
void assemble_matrix_flat(
        T* d, T* rhs, const T* invariant_d,
        const T* voltage, const T* current, const T* cv_capacitance,
        T dt, int n)
{
    auto tid = threadIdx.x + blockDim.x*blockIdx.x;

    T factor = 1e-3/dt;
    if (tid<n) {
        auto gi = factor * cv_capacitance[tid];

        d[tid] = gi + invariant_d[tid];

        rhs[tid] = gi*voltage[tid] - current[tid];
    }
}

/// GPU implementatin of Hines matrix assembly
/// Interleaved layout
/// For a given time step size dt
///     - use the precomputed alpha and alpha_d values to construct the diagonal
///       and off diagonal of the symmetric Hines matrix.
///     - compute the RHS of the linear system to solve
template <typename T, typename I, int BlockWidth, int LoadWidth, int Threads>
__global__
void assemble_matrix_interleaved(
        T* d,
        T* rhs,
        const T* invariant_d,
        const T* voltage,
        const T* current,
        const T* cv_capacitance,
        const I* sizes,
        const I* starts,
        T dt, int padded_size, int num_mtx)
{
    static_assert(BlockWidth*LoadWidth==Threads,
        "number of threads must equal number of values to process per block");
    __shared__ T buffer_v[Threads];
    __shared__ T buffer_i[Threads];

    const auto tid = threadIdx.x + blockIdx.x*blockDim.x;
    const auto lid = threadIdx.x;

    const auto mtx_id   = tid/LoadWidth;
    const auto mtx_lane = tid - mtx_id*LoadWidth;

    const auto blk_id   = tid/(BlockWidth*LoadWidth);
    const auto blk_row  = lid/BlockWidth;
    const auto blk_lane = lid - blk_row*BlockWidth;

    const auto blk_pos  = LoadWidth*blk_lane + blk_row;

    const bool do_load  = mtx_id<num_mtx;

    auto load_pos  = do_load? starts[mtx_id] + mtx_lane     : 0;
    const auto end = do_load? starts[mtx_id] + sizes[mtx_id]: 0;
    auto store_pos = blk_id*BlockWidth*padded_size + (blk_row*BlockWidth + blk_lane);

    const auto max_size = sizes[0];

    T factor = 1e-3/dt;
    for (auto j=0; j<max_size; j+=LoadWidth) {
        if (do_load && load_pos<end) {
            buffer_v[lid] = voltage[load_pos];
            buffer_i[lid] = current[load_pos];
        }
        __syncthreads();

        const T v = buffer_v[blk_pos];
        const T i = buffer_i[blk_pos];

        if (i+blk_row<padded_size) {
            const auto gi = factor * cv_capacitance[store_pos];
            d[store_pos]   = gi + invariant_d[store_pos];
            rhs[store_pos] = gi*v - i;
        }

        store_pos += LoadWidth*BlockWidth;
        load_pos  += LoadWidth;
    }
}

/// kernel used to test for threshold crossing test code.
/// params:
///     t       : current time (ms)
///     t_prev  : time of last test (ms)
///     size    : number of values to test
///     is_crossed  : crossing state at time t_prev (true or false)
///     prev_values : values at sample points (see index) sampled at t_prev
///     index      : index with locations in values to test for crossing
///     values     : values at t_prev
///     thresholds : threshold values to watch for crossings
template <typename T, typename I, typename Stack>
__global__
void test_thresholds(
    float t, float t_prev, int size,
    Stack& stack,
    I* is_crossed, T* prev_values,
    const I* index, const T* values, const T* thresholds)
{
    int i = threadIdx.x + blockIdx.x*blockDim.x;

    bool crossed = false;
    float crossing_time;

    if (i<size) {
        // Test for threshold crossing
        const auto v_prev = prev_values[i];
        const auto v      = values[index[i]];
        const auto thresh = thresholds[i];

        if (!is_crossed[i]) {
            if (v>=thresh) {
                // The threshold has been passed, so estimate the time using
                // linear interpolation
                auto pos = (thresh - v_prev)/(v - v_prev);
                crossing_time = t_prev + pos*(t - t_prev);

                is_crossed[i] = 1;
                crossed = true;
            }
        }
        else if (v<thresh) {
            is_crossed[i]=0;
        }

        prev_values[i] = v;
    }

    if (crossed) {
        stack.push_back({I(i), crossing_time});
    }
}


} // namespace gpu
} // namespace mc
} // namespace nest
