#include <limits>

#include "gtest.h"

#include <fvm_cell.hpp>
#include <cell_group.hpp>

nest::mc::cell make_cell() {
    using namespace nest::mc;

    // setup global state for the mechanisms
    mechanisms::setup_mechanism_helpers();

    nest::mc::cell cell;

    // Soma with diameter 12.6157 um and HH channel
    auto soma = cell.add_soma(12.6157/2.0);
    soma->add_mechanism(hh_parameters());

    // add dendrite of length 200 um and diameter 1 um with passive channel
    auto dendrite = cell.add_cable(0, segmentKind::dendrite, 0.5, 0.5, 200);
    dendrite->add_mechanism(pas_parameters());
    dendrite->set_compartments(101);

    dendrite->mechanism("membrane").set("r_L", 100);

    // add stimulus
    cell.add_stimulus({1,1}, {5., 80., 0.3});

    cell.add_detector({0,0}, 0);

    return cell;
}

TEST(cell_group, test)
{
    using namespace nest::mc;

    using cell_type = cell_group<fvm::fvm_cell<double, int>>;

    auto cell = cell_type{make_cell()};

    cell.advance(50, 0.01);

    // a bit lame...
    EXPECT_EQ(cell.spikes().size(), 4u);
}

