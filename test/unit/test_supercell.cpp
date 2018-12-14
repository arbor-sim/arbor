#include "../gtest.h"

#include <cmath>
#include <tuple>
#include <vector>

#include <arbor/constants.hpp>
#include <arbor/mechcat.hpp>
#include <arbor/util/optional.hpp>
#include <arbor/mc_cell.hpp>

#include "backends/multicore/fvm.hpp"
#include "backends/multicore/mechanism.hpp"
#include "util/maputil.hpp"
#include "util/range.hpp"

#include "common.hpp"
#include "mech_private_field_access.hpp"

using namespace arb;

using backend = ::arb::multicore::backend;
using shared_state = backend::shared_state;
using value_type = backend::value_type;
using size_type = backend::size_type;

// Access to more mechanism protected data:

ACCESS_BIND(const value_type* multicore::mechanism::*, vec_v_ptr, &multicore::mechanism::vec_v_)
ACCESS_BIND(value_type* multicore::mechanism::*, vec_i_ptr, &multicore::mechanism::vec_i_)

TEST(supercell, sync_time_to) {
    using value_type = multicore::backend::value_type;
    using index_type = multicore::backend::index_type;

    int num_cell = 10;

    std::vector<gap_junction> gj = {};
    std::vector<index_type> deps = {4, 0, 0, 0, 3, 0, 0, 2, 0, 0};

    shared_state state(num_cell, std::vector<index_type>(num_cell, 0), deps, gj, 1u);

    state.time_to = {0.3, 0.1, 0.2, 0.4, 0.5, 0.6, 0.1, 0.1, 0.6, 0.9};
    state.sync_time_to();

    std::vector<value_type> expected = {0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.9};

    for (unsigned i = 0; i < state.time_to.size(); i++) {
        EXPECT_EQ(expected[i], state.time_to[i]);
    }

    state.time_dep = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    state.time_to = {0.3, 0.1, 0.2, 0.4, 0.5, 0.6, 0.1, 0.1, 0.6, 0.9};
    expected      = {0.3, 0.1, 0.2, 0.4, 0.5, 0.6, 0.1, 0.1, 0.6, 0.9};
    state.sync_time_to();

    for (unsigned i = 0; i < state.time_to.size(); i++) {
        EXPECT_EQ(expected[i], state.time_to[i]);
    }

    state.time_dep = {10, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    state.time_to = {0.3, 0.1, 0.2, 0.4, 0.5, 0.6, 0.1, 0.1, 0.6, 0.9};
    expected      = {0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1};
    state.sync_time_to();

    for (unsigned i = 0; i < state.time_to.size(); i++) {
        EXPECT_EQ(expected[i], state.time_to[i]);
    }

}

