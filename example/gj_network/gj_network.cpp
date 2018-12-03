#include <fstream>
#include <iomanip>
#include <iostream>

#include <nlohmann/json.hpp>

#include <arbor/assert_macro.hpp>
#include <arbor/common_types.hpp>
#include <arbor/context.hpp>
#include <arbor/load_balance.hpp>
#include <arbor/mc_cell.hpp>
#include <arbor/profile/meter_manager.hpp>
#include <arbor/profile/profiler.hpp>
#include <arbor/simple_sampler.hpp>
#include <arbor/simulation.hpp>
#include <arbor/recipe.hpp>
#include <arbor/version.hpp>

#include <sup/ioutil.hpp>
#include <sup/json_meter.hpp>

#ifdef ARB_MPI_ENABLED
#include <mpi.h>
#include <sup/with_mpi.hpp>
#endif

using arb::cell_gid_type;
using arb::cell_lid_type;
using arb::cell_size_type;
using arb::cell_member_type;
using arb::cell_kind;
using arb::time_type;
using arb::cell_probe_address;

// TODO
// The tests here will only test domain decomposition with GPUs when compiled
// with CUDA support and run on a system with a GPU.
// Ideally the tests should test domain decompositions under all conditions, however
// to do that we have to refactor the partition_load_balance algorithm.
// The partition_load_balance performs the decomposition to distribute
// over resources described by the user-supplied arb::context, which is a
// provides an interface to resources available at runtime.
// The best way to test under all conditions, is probably to refactor the
// partition_load_balance into components that can be tested in isolation.

namespace {

class gj_recipe : public arb::recipe {
public:
    gj_recipe(cell_size_type s) : size_(s) {}

    cell_size_type num_cells() const override {
        return size_;
    }

    arb::util::unique_any get_cell_description(cell_gid_type) const override {
        return {};
    }

    cell_kind get_cell_kind(cell_gid_type gid) const override {
        return gid % 2 ?
               cell_kind::spike_source :
               cell_kind::cable1d_neuron;
    }

    std::vector<cell_gid_type> group_with(cell_gid_type gid) const override {
        switch (gid) {
            case 1 : return {2};
            case 2 : return {1};
            case 5 : return {16};
            case 12 : return {13, 22};
            case 13 : return {12, 22};
            case 16 : return {5};
            case 18 : return {19, 28};
            case 19 : return {18, 29};
            case 22 : return {12, 13};
            case 28 : return {18, 29};
            case 29 : return {19, 28};
            case 31 : return {42};
            case 32 : return {42, 41};
            case 35 : return {54, 56};
            case 41 : return {32};
            case 42 : return {32, 31};
            case 44 : return {46, 56};
            case 46 : return {44, 54};
            case 54 : return {35, 46};
            case 56 : return {44, 35};
            case 62 : return {63, 72};
            case 63 : return {62, 72};
            case 66 : return {75, 77};
            case 67 : return {78};
            case 72 : return {81, 82, 62, 63};
            case 75 : return {66};
            case 77 : return {66, 78};
            case 78 : return {67, 77};
            case 81 : return {72, 82};
            case 82 : return {81, 72};
            case 85 : return {94};
            case 94 : return {85};
            case 97 : return {98};
            case 98 : return {97};
            default : return {};
        }
    }

private:
    cell_size_type size_;
};
}

int main(int argc, char** argv) {
    arb::proc_allocation resources{1, -1};
    int rank = 0;
#ifdef ARB_MPI_ENABLED
    sup::with_mpi guard(argc, argv, false);
    auto ctx = make_context(resources, MPI_COMM_WORLD);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#else
    auto ctx = make_context(resources);
#endif
    auto R = gj_recipe(100);
    const auto D = partition_load_balance(R, ctx);

    auto groups = D.groups;
    for (auto g: groups) {
        std::cout << "{ ";
        for (auto id : g.gids) {
            std::cout << id << " ";
        }
        std::cout <<" } - "  << rank <<"\n";
    }
    return 0;
}