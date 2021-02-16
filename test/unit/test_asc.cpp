#include <iostream>
#include <fstream>

#include <arbor/cable_cell.hpp>
#include <arbor/morph/primitives.hpp>
#include <arbor/morph/segment_tree.hpp>

#include <arborio/neurolucida.hpp>

#include "../gtest.h"

TEST(asc, parse) {
    auto y = arborio::load_asc("/home/bcumming/software/github/arbor/test/unit/neurolucida/01bc.asc");
    auto z = arborio::load_asc("/home/bcumming/software/github/arbor/test/unit/neurolucida/pair-140514-C2-1_split_1.asc");
    auto a = arborio::load_asc("/home/bcumming/software/github/arbor/test/unit/neurolucida/soma_10c.asc");
    auto b = arborio::load_asc("/home/bcumming/software/github/arbor/test/unit/neurolucida/stellate.asc");
}
