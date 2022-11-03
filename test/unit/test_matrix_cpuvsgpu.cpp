#include <numeric>
#include <random>
#include <vector>

#include <arbor/math.hpp>

#include "matrix.hpp"
#include "memory/memory.hpp"
#include "util/span.hpp"

#include "backends/gpu/fvm.hpp"
#include "backends/multicore/fvm.hpp"

#include <gtest/gtest.h>
#include "common.hpp"


using namespace arb;

using util::make_span;
using util::assign_from;
using memory::on_gpu;
using memory::on_host;

using testing::seq_almost_eq;

using std::begin;
using std::end;


// Test that matrix assembly works.
// The test proceeds by assembling a reference matrix on the host and
// device backends, then performs solve, and compares solution.
//
// limitations of test
//  * matrices all have same size and structure
TEST(matrix, assemble)
{
    using gpu_state = gpu::backend::cable_solver;
    using mc_state  = multicore::backend::cable_solver;

    using T = arb_value_type;
    using I = arb_index_type;

    using gpu_array  = typename gpu::backend::array;
    using host_array = typename multicore::backend::array;

    // There are two matrix structures:
    //
    // p_1: 3 branches, 6 compartments
    //
    //           3
    //          /.
    // 0 - 1 - 2
    //          \.
    //           4
    //            \.
    //             5
    //
    // p_2: 5 branches, 8 compartments
    //
    //             4
    //            /.
    //           3
    //          / \.
    // 0 - 1 - 2   5
    //          \.
    //           6
    //            \.
    //             7
    // p_3: 1 branch, 1 compartment
    //
    // 0

    // The parent indexes that define the two matrix structures
    std::vector<std::vector<I>>
        p_base = { {0,0,1,2,2,4}, {0,0,1,2,3,3,2,6}, {0} };

    // Make a set of matrices based on repeating this pattern.
    // We assign the patterns round-robin, i.e. so that the input
    // matrices will have alternating sizes of 6 and 8, which will
    // test the solver with variable matrix size, and exercise
    // solvers that reorder matrices according to size.
    const int num_mtx = 100;

    std::vector<I> p;
    std::vector<I> cell_index;
    std::vector<I> intdom_index;
    for (auto m=0; m<num_mtx; ++m) {
        auto &p_ref = p_base[m%p_base.size()];
        auto first = p.size();
        for (auto i: p_ref) {
            p.push_back(i + first);
        }
        cell_index.push_back(first);
        intdom_index.push_back(m);
    }
    cell_index.push_back(p.size());

    auto group_size = cell_index.back();

    // Build the capacitance and conductance vectors and
    // populate with nonzero random values.
    auto gen  = std::mt19937();
    auto dist = std::uniform_real_distribution<T>(1, 2);

    std::vector<T> Cm(group_size);
    std::generate(Cm.begin(), Cm.end(), [&](){return dist(gen);});

    std::vector<T> g(group_size);
    std::generate(g.begin(), g.end(), [&](){return dist(gen);});

    std::vector<T> area(group_size, 1e3);

    // Make the reference matrix and the gpu matrix
    auto m_mc  = mc_state( p, cell_index, Cm, g, area, intdom_index); // on host
    auto m_gpu = gpu_state(p, cell_index, Cm, g, area, intdom_index); // on gpu

    // Set the integration times for the cells to be between 0.1 and 0.2 ms.
    std::vector<T> dt(num_mtx);

    auto dt_dist = std::uniform_real_distribution<T>(0.1, 0.2);
    std::generate(dt.begin(), dt.end(), [&](){return dt_dist(gen);});

    // Voltage, current, and conductance values
    auto result_h = host_array(group_size, -64);
    auto x_d = gpu_array(group_size);
    m_mc.solve(result_h, host_array(dt.begin(), dt.end()), host_array(group_size, 10), host_array(group_size, 3));
    m_gpu.assemble(on_gpu(dt), gpu_array(group_size, -64), gpu_array(group_size, 10), gpu_array(group_size, 3));
    m_gpu.solve(x_d);
    auto result_g = on_host(x_d);

    // Compare the GPU and CPU results.
    // Cast result to float, because we are happy to ignore small differencs
    EXPECT_TRUE(seq_almost_eq<float>(result_h, result_g));
}

