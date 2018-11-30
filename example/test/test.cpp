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
            case 0 : return {};
            case 1 : return {};
            case 2 : return {7, 8};
            case 3 : return {};
            case 4 : return {};
            case 5 : return {};
            case 6 : return {};
            case 7 : return {2, 8};
            case 8 : return {2, 7};
            case 9 : return {16};
            case 10 : return {};
            case 11 : return {};
            case 12 : return {};
            case 13 : return {19};
            case 14 : return {};
            case 15 : return {};
            case 16 : return {9, 21};
            case 17 : return {23};
            case 18 : return {};
            case 19 : return {13, 26};
            case 20 : return {21};
            case 21 : return {20, 16};
            case 22 : return {};
            case 23 : return {17};
            case 24 : return {};
            case 25 : return {};
            case 26 : return {19, 27};
            case 27 : return {26};
            case 28 : return {29};
            case 29 : return {28};
        }
    }

private:
    cell_size_type size_;
};
}

int main(int argc, char** argv) {
        arb::proc_allocation resources{1, -1};
#ifdef ARB_MPI_ENABLED
        sup::with_mpi guard(argc, argv, false);
        auto ctx = make_context(resources, MPI_COMM_WORLD);
#else
        auto ctx = make_context(resources);
#endif
        auto R = gj_recipe(30);
        const auto D = partition_load_balance(R, ctx);

        return 0;
}