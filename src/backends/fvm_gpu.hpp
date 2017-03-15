#pragma once

#include <map>
#include <string>

#include <common_types.hpp>
#include <mechanism.hpp>
#include <memory/memory.hpp>
#include <memory/managed_ptr.hpp>
#include <util/span.hpp>
#include <util/rangeutil.hpp>

#include "stimulus_gpu.hpp"
#include "gpu_stack.hpp"

template <typename T>
void print_vec(const char* name, std::vector<T>& v) {
    std::cout << name << ": ";
    for(auto vv: v) {
        if (vv==std::numeric_limits<T>::max())
            std::cout <<  "* ";
        else
            std::cout <<  vv  << " ";
    }
    std::cout << "\n";
};

template <typename T>
void print_vec(const char* name, const T& v) {
    std::cout << name << ": ";
    for(auto vv: v) {
        std::cout <<  vv  << " ";
    }
    std::cout << "\n";
};

namespace nest {
namespace mc {
namespace gpu {

namespace impl {
    __host__ __device__
    constexpr inline int block_dim() {
        return 8;
    }

    __host__ __device__
    constexpr inline int load_width() {
        return 4;
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
}

template <typename T, typename I, int BlockWidth, int LoadWidth, int Threads>
__global__
void assemble_matrix(
    T* d, T* rhs, const T* invariant_d,
    const T* voltage, const T* current, const T* cv_capacitance,
    const I* lengths, const I* starts, T dt, int n);

// forward declarations of the matrix solver implementation
// see the bottom of the file for implementation

template <typename T, typename I, int BlockWidth>
__global__
void matrix_solve(
    T* rhs, T* d, const T* u, const I* p, const I* lengths, int num_mtx, int padded_mtx_length);

template <typename T, typename I, int BlockWidth, int LoadWidth>
void reverse_interleave(const T* in, T* out, const I* lengths, const I* starts, int max_length, int n);

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
    const I* index, const T* values, const T* thresholds);

struct backend {
    /// define the real and index types
    using value_type = double;
    using size_type  = nest::mc::cell_lid_type;

    /// define storage types
    using array  = memory::device_vector<value_type>;
    using iarray = memory::device_vector<size_type>;

    using view       = typename array::view_type;
    using const_view = typename array::const_view_type;

    using iview       = typename iarray::view_type;
    using const_iview = typename iarray::const_view_type;

    using host_array  = typename memory::host_vector<value_type>;
    using host_iarray = typename memory::host_vector<size_type>;

    using host_view   = typename host_array::view_type;
    using host_iview  = typename host_iarray::const_view_type;

    static std::string name() {
        return "gpu";
    }

    //
    // matrix infrastructure
    //

    /// matrix state
    struct matrix_state {
        //
        // Permutation and index information required for forward and backward
        // interleave-permutation of vectors.
        //

        // size of each matrix (after permutation in ascending size)
        iarray matrix_sizes;
        // start values corresponding to matrix i in the external storage
        iarray matrix_index;

        //
        // Storage for the matrix and parent index in interleaved format.
        // Includes the cv_capacitance, which is required for matrix assembly.
        //
        iarray parent_index;
        array d;   // [μS]
        array u;   // [μS]
        array rhs; // [nA]

        // required for matrix assembly
        array cv_capacitance; // [pF]

        // the invariant part of the matrix diagonal
        array invariant_d;    // [μS]

        int num_matrices() const {
            return matrix_sizes.size();
        }

        int padded_matrix_size() const {
            return parent_index.size()/num_matrices();
        }

        //
        //  Storage for solution in uninterleaved format.
        //  Used to hold the storage for passing to caller, and must be updated
        //  after each call to the ::solve() method.
        //

        array solution;

        unsigned matrix_dim;

        // default constructor
        matrix_state() = default;

        // Construct matrix state for a set of matrices defined by parent_index p
        // The matrix solver stores the matrix in an "interleaved" structure for
        // optimal solution, which requires a significant amount of precomputing
        // of indexes and data structures in the constructor.
        //  cv_cap      // [pF]
        //  face_cond   // [μS]
        matrix_state(const std::vector<size_type>& p,
                     const std::vector<size_type>& cell_index,
                     const std::vector<value_type>& cv_cap,
                     const std::vector<value_type>& face_cond)
        {
            using util::make_span;

            // convenience for commonly used type in this routine
            using svec = std::vector<size_type>;

            //
            // sort matrices in descending order of size
            //

            // find the size of each matrix
            const auto num_mtx = cell_index.size()-1;
            svec sizes;
            for (auto it=cell_index.begin()+1; it!=cell_index.end(); ++it) {
                sizes.push_back(*it - *(it-1));
            }

            // find permutations and sort indexes/sizes
            svec perm(num_mtx);
            std::iota(perm.begin(), perm.end(), 0);
            // calculate the permutation of matrices to put the in ascending size
            util::stable_sort_by(perm, [&sizes](size_type i){ return sizes[i]; });
            std::reverse(perm.begin(), perm.end());

            // TODO: refactor to be less verbose with permutation_view
            svec sizes_p;
            for (auto i: make_span(0, num_mtx)) {
                sizes_p.push_back(sizes[perm[i]]);
            }
            svec cell_index_p;
            for (auto i: make_span(0, num_mtx)) {
                cell_index_p.push_back(cell_index[perm[i]]);
            }

            print_vec("perm   ", perm);
            print_vec("sizes  ", sizes);
            print_vec("sizes_p", sizes_p);
            print_vec("index_p", cell_index_p);

            //
            // Calculate dimensions required to store matrices.
            //
            using impl::block_dim;
            using impl::matrix_padding;

            // To start, take simplest approach of assuming all matrices stored
            // in blocks of the same dimension: matrix_dim
            matrix_dim = impl::padded_size(sizes_p[0], matrix_padding());
            const auto num_blocks = impl::block_count(num_mtx, block_dim());

            const auto total_storage = num_blocks*block_dim()*matrix_dim;

            std::cout << "matrix dimension: " << matrix_dim
                << " (padding " << matrix_padding() << ") requires total storage "
                << total_storage << "\n";

            // calculate the interleaved and permuted p vector
            constexpr auto npos = std::numeric_limits<size_type>::max();
            host_iarray p_tmp(total_storage, npos);
            for (auto mtx: make_span(0, num_mtx)) {
                auto block = mtx/block_dim();
                auto lane  = mtx%block_dim();

                auto len = sizes_p[mtx];
                auto src = cell_index_p[mtx];
                auto dst = block*(block_dim()*matrix_dim) + lane;
                for (auto i: make_span(0, len)) {
                    // the p indexes are always relative to the start of the p vector.
                    // the addition and subtraction of dst and src respectively is to convert from
                    // the original offset to the new padded and permuted offset.
                    p_tmp[dst+block_dim()*i] = dst + block_dim()*(p[src+i]-src);
                }
            }

            d   = array(total_storage);
            u   = array(total_storage);
            rhs = array(total_storage);
            parent_index = p_tmp;

            //
            //  Calculate the invariant part of the matrix diagonal and the
            //  upper diagonal on the host, then copy to the device.
            //

            std::vector<value_type> invariant_d_tmp(p.size(), 0);
            std::vector<value_type> u_tmp(p.size(), 0);
            auto face_conductance_tmp = memory::on_host(face_cond);
            for(auto i: util::make_span(1u, p.size())) {
                auto gij = face_conductance_tmp[i];

                u_tmp[i] = -gij;
                invariant_d_tmp[i] += gij;
                invariant_d_tmp[p[i]] += gij;
            }

            // This lambda is a helper that performs the interleave operation
            // on host memory.
            auto interleave = [&](const std::vector<value_type>& v) {
                host_array out(total_storage);
                for (auto mtx: make_span(0, num_mtx)) {
                    auto block = mtx/block_dim();
                    auto lane  = mtx%block_dim();

                    auto len = sizes_p[mtx];
                    auto src = cell_index_p[mtx];
                    auto dst = block*(block_dim()*matrix_dim) + lane;
                    for (auto i: make_span(0, len)) {
                        out[dst] = v[src+i];
                        dst += block_dim();
                    }
                }
                return out;
            };

            u              = interleave(u_tmp);
            invariant_d    = interleave(invariant_d_tmp);
            cv_capacitance = interleave(cv_cap);
            matrix_sizes = memory::make_const_view(sizes_p);
            matrix_index = memory::make_const_view(cell_index_p);

            EXPECTS(num_mtx == num_matrices());

            /*
            {
                std::vector<value_type> base(p.size());
                std::iota(base.begin(), base.end(), 0);
                array forward = interleave(base);
                array backward(p.size());

                reverse_interleave<value_type, size_type, impl::block_dim(), impl::load_width()>
                    (forward.data(),
                     backward.data(),
                     matrix_sizes.data(),
                     matrix_index.data(),
                     matrix_dim, num_mtx);

                print_vec("base", base);
                std::cout << "\n";
                print_vec("fwd ", memory::on_host(forward));
                std::cout << "\n";
                print_vec("bck ", memory::on_host(backward));
                exit(0);
            }
            */
        }

        // Assemble the matrix
        // Afterwards the diagonal and RHS will have been set given dt, voltage and current
        //   dt      [ms]
        //   voltage [mV]
        //   current [nA]
        void assemble(value_type dt, const_view voltage, const_view current) {
            EXPECTS(has_fvm_state());

            constexpr auto bd = impl::block_dim();
            constexpr auto lw = impl::load_width();
            constexpr auto block_dim = bd*lw;

            // the number of threads is threads_per_matrix*num_mtx
            const auto num_blocks = impl::block_count(num_matrices()*lw, block_dim);

            assemble_matrix <value_type, size_type, bd, lw, block_dim>
                <<<num_blocks, block_dim>>>
                ( d.data(), rhs.data(), invariant_d.data(),
                  voltage.data(), current.data(), cv_capacitance.data(),
                  matrix_sizes.data(), matrix_index.data(), dt, num_matrices());

        }

        void solve() {
            // perform the Hines solve
            auto const grid_dim = impl::block_count(num_matrices(), impl::block_dim());

            matrix_solve<value_type, size_type, impl::block_dim()>
                <<<grid_dim, impl::block_dim()>>>
                ( rhs.data(), d.data(), u.data(), parent_index.data(), matrix_sizes.data(),
                  num_matrices(), padded_matrix_size());

            // copy the solution out
        }

        // Test if the matrix has the full state required to assemble the
        // matrix in the fvm scheme.
        bool has_fvm_state() const {
            return cv_capacitance.size()>0;
        }
    };

    //
    // mechanism infrastructure
    //
    using ion = mechanisms::ion<backend>;

    using mechanism = mechanisms::mechanism_ptr<backend>;

    using stimulus = mechanisms::gpu::stimulus<backend>;

    static mechanism make_mechanism(
        const std::string& name,
        view vec_v, view vec_i,
        const std::vector<value_type>& weights,
        const std::vector<size_type>& node_indices)
    {
        if (!has_mechanism(name)) {
            throw std::out_of_range("no mechanism in database : " + name);
        }

        return mech_map_.find(name)->
            second(vec_v, vec_i, memory::make_const_view(weights), memory::make_const_view(node_indices));
    }

    static bool has_mechanism(const std::string& name) { return mech_map_.count(name)>0; }

    /// threshold crossing logic
    /// used as part of spike detection back end
    class threshold_watcher {
    public:
        /// stores a single crossing event
        struct threshold_crossing {
            size_type index;    // index of variable
            value_type time;    // time of crossing
            __host__ __device__
            friend bool operator==
                (const threshold_crossing& lhs, const threshold_crossing& rhs)
            {
                return lhs.index==rhs.index && lhs.time==rhs.time;
            }
        };

        using stack_type = gpu_stack<threshold_crossing>;

        threshold_watcher() = default;

        threshold_watcher(
                const_view values,
                const std::vector<size_type>& index,
                const std::vector<value_type>& thresh,
                value_type t=0):
            values_(values),
            index_(memory::make_const_view(index)),
            thresholds_(memory::make_const_view(thresh)),
            prev_values_(values),
            is_crossed_(size()),
            stack_(memory::make_managed_ptr<stack_type>(10*size()))
        {
            reset(t);
        }

        /// Remove all stored crossings that were detected in previous calls
        /// to test()
        void clear_crossings() {
            stack_->clear();
        }

        /// Reset state machine for each detector.
        /// Assume that the values in values_ have been set correctly before
        /// calling, because the values are used to determine the initial state
        void reset(value_type t=0) {
            clear_crossings();

            // Make host-side copies of the information needed to calculate
            // the initial crossed state
            auto values = memory::on_host(values_);
            auto thresholds = memory::on_host(thresholds_);
            auto index = memory::on_host(index_);

            // calculate the initial crossed state in host memory
            auto crossed = std::vector<size_type>(size());
            for (auto i: util::make_span(0u, size())) {
                crossed[i] = values[index[i]] < thresholds[i] ? 0 : 1;
            }

            // copy the initial crossed state to device memory
            is_crossed_ = memory::on_gpu(crossed);

            // reset time of last test
            t_prev_ = t;
        }

        bool is_crossed(size_type i) const {
            return is_crossed_[i];
        }

        const std::vector<threshold_crossing> crossings() const {
            return std::vector<threshold_crossing>(stack_->begin(), stack_->end());
        }

        /// The time at which the last test was performed
        value_type last_test_time() const {
            return t_prev_;
        }

        /// Tests each target for changed threshold state.
        /// Crossing events are recorded for each threshold that has been
        /// crossed since current time t, and the last time the test was
        /// performed.
        void test(value_type t) {
            EXPECTS(t_prev_<t);

            constexpr int block_dim = 128;
            const int grid_dim = (size()+block_dim-1)/block_dim;
            test_thresholds<<<grid_dim, block_dim>>>(
                t, t_prev_, size(),
                *stack_,
                is_crossed_.data(), prev_values_.data(),
                index_.data(), values_.data(), thresholds_.data());

            // Check that the number of spikes has not exceeded
            // the capacity of the stack.
            EXPECTS(stack_->size() <= stack_->capacity());

            t_prev_ = t;
        }

        /// the number of threashold values that are being monitored
        std::size_t size() const {
            return index_.size();
        }

        /// Data type used to store the crossings.
        /// Provided to make type-generic calling code.
        using crossing_list =  std::vector<threshold_crossing>;

    private:

        const_view values_;         // values to watch: on gpu
        iarray index_;              // indexes of values to watch: on gpu

        array thresholds_;          // threshold for each watch: on gpu
        value_type t_prev_;         // time of previous sample: on host
        array prev_values_;         // values at previous sample time: on host
        iarray is_crossed_;         // bool flag for state of each watch: on gpu

        memory::managed_ptr<stack_type> stack_;
    };

private:

    using maker_type = mechanism (*)(view, view, array&&, iarray&&);
    static std::map<std::string, maker_type> mech_map_;

    template <template <typename> class Mech>
    static mechanism maker(view vec_v, view vec_i, array&& weights, iarray&& node_indices) {
        return mechanisms::make_mechanism<Mech<backend>>
            (vec_v, vec_i, std::move(weights), std::move(node_indices));
    }
};

/// GPU implementation of Hines Matrix solver.
/// Naive implementation with one CUDA thread per matrix.
template <typename T, typename I, int BlockWidth>
__global__
void matrix_solve(
    T* rhs, T* d, const T* u, const I* p, const I* lengths, int num_mtx, int padded_mtx_length)
{
    auto tid = threadIdx.x + blockDim.x*blockIdx.x;

    if(tid < num_mtx) {
        auto block       = tid/BlockWidth;
        auto block_start = block*BlockWidth;
        auto block_index = tid - block_start;

        // get range of this thread's cell matrix
        auto first = block_start*padded_mtx_length + block_index;
        auto last  = first + BlockWidth*(lengths[tid]-1);

        // backward sweep
        for(auto i=last-1; i>first; --i) {
            auto factor = u[i] / d[i];
            d[p[i]]   -= factor * u[i];
            rhs[p[i]] -= factor * rhs[i];
        }

        __syncthreads();
        rhs[first] /= d[first];

        // forward sweep
        for(auto i=first+1; i<last; ++i) {
            rhs[i] -= u[i] * rhs[p[i]];
            rhs[i] /= d[i];
        }
    }
}

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

template <typename T, typename I, int BlockWidth, int LoadWidth, int THREADS>
__global__
void reverse_interleave(
    const T* in, T* out, const I* lengths, const I* starts, int max_length, int n)
{
    static_assert(BlockWidth*LoadWidth==THREADS, "");

    __shared__ T buffer[THREADS];

    auto tid = threadIdx.x + blockIdx.x*blockDim.x;
    auto lid = threadIdx.x;

    auto mtx_id   = tid/LoadWidth;
    auto mtx_lane = tid - mtx_id*LoadWidth;

    auto blk_id   = tid/BlockWidth;
    auto blk_start= blk_id*BlockWidth;
    auto blk_lane = tid - blk_start;

    auto blk_lid  = lid/BlockWidth;
    auto blk_pos  = blk_lane*LoadWidth + blk_lid;

    auto store_pos = starts[mtx_id] + mtx_lane;
    auto end       = starts[mtx_id] + lengths[mtx_id];
    auto load_pos  = blk_start + blk_lane;

    if (mtx_id<n) {
        for (auto i=0; i<max_length; i+=LoadWidth) {
            buffer[blk_pos] = in[load_pos];
            __syncthreads();
            if (store_pos<end) {
                out[store_pos] = buffer[lid];
            }
            load_pos  += LoadWidth*BlockWidth;
            store_pos += LoadWidth;
        }
    }
}

template <typename T, typename I, int BlockWidth, int LoadWidth>
void reverse_interleave(const T* in, T* out, const I* lengths, const I* starts, int max_length, int n)
{
    constexpr int Threads = BlockWidth*LoadWidth;
    reverse_interleave<T, I, BlockWidth, LoadWidth, Threads>
        <<<impl::block_count(n, Threads), Threads>>>
        (in, out, lengths, starts, max_length, n);
}

/// GPU implementatin of Hines matrix assembly
/// For a given time step size dt
///     - use the precomputed alpha and alpha_d values to construct the diagonal
///       and off diagonal of the symmetric Hines matrix.
///     - compute the RHS of the linear system to solve
template <typename T, typename I, int BlockWidth, int LoadWidth, int Threads>
__global__
void assemble_matrix(
        T* d,
        T* rhs,
        const T* invariant_d,
        const T* voltage,
        const T* current,
        const T* cv_capacitance,
        const I* lengths,
        const I* starts,
        T dt, int n)
{
    static_assert(BlockWidth*LoadWidth==Threads,
        "number of threads must equal number of values to process per block");
    __shared__ T buffer_v[Threads];
    __shared__ T buffer_i[Threads];

    const auto tid = threadIdx.x + blockDim.x*blockIdx.x;
    const auto lid = threadIdx.x;

    const auto max_length = lengths[0];

    const auto mtx_id   = tid/LoadWidth;
    const auto mtx_lane = tid - mtx_id*LoadWidth;

    const auto blk_id   = tid/BlockWidth;
    const auto blk_start= blk_id*BlockWidth;
    const auto blk_lane = tid - blk_start;

    const auto blk_lid  = lid/BlockWidth;
    const auto blk_pos  = blk_lane*LoadWidth + blk_lid;

    const auto end = starts[mtx_id] + lengths[mtx_id];
    auto load_pos  = starts[mtx_id] + mtx_lane;
    auto store_pos = blk_start + blk_lane;

    T factor = 1e-3/dt;
    if (mtx_id<n) {
        for (auto j=0; j<max_length; j+=LoadWidth) {
            if (load_pos<end) {
                buffer_v[lid] = voltage[load_pos];
                buffer_i[lid] = current[load_pos];
            }
            __syncthreads();

            // TODO: Potential bank conflicts on these buffers.
            // Test with nvprof, and add pad-to-prime-number
            // if needed.
            const T v = buffer_v[blk_pos];
            const T i = buffer_i[blk_pos];

            const auto gi = factor * cv_capacitance[store_pos];
            d[store_pos]   = gi + invariant_d[store_pos];
            rhs[store_pos] = gi*v - i;

            store_pos += LoadWidth*BlockWidth;
            load_pos  += LoadWidth;
        }
    }
}

} // namespace gpu
} // namespace mc
} // namespace nest
