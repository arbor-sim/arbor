#include "../gtest.h"
#include "common_cells.hpp"

#include "s_expr.hpp"

#include <arbor/cable_cell.hpp>
#include <arbor/cable_cell_param.hpp>

using namespace arb;

TEST(cable_cell, decor) {
    auto tree = segments_from_points(
            {{0, 0, 0, 10}, {0, 0, 10, 10},
             {0, 0, 10, 1}, {0, 0, 100, 1},
             {0, 0, 0, 2}, {0, 0, -20, 2}
            },
            {mnpos, 0, 1, 2, mnpos, 4},
            {1, 1, 3, 3, 2, 2});

    label_dict dict;
    dict.set("soma", region("(tag 1)"));
    dict.set("axon", region("(tag 2)"));
    dict.set("dend", region("(tag 3)"));
    dict.set("term", locset("(terminal)"));
    dict.set("stim-site", locset("(location 0 0.5)"));

    auto mech = mechanism_desc{"hh"};
    mech.set("foo", 12);
    mech.set("bar", 1.2);
    decor decorations;
    decorations.paint("(tag 2)", axial_resistivity{23});
    decorations.paint("\"soma\"", membrane_capacitance{1.2});
    decorations.paint("\"dend\"", "hh");
    decorations.paint("\"axon\"", mech);
    decorations.place("(terminal)", threshold_detector{-10});
    decorations.set_default(init_membrane_potential{-60});


    cable_cell cell1(tree, dict, decorations);

    //cable_cell c2(tree, dict);

    //write_s_expr(std::cout, c2) << "\n==============================================\n";
    //c2.decorate(c.decorations());
    //write_s_expr(std::cout, c2) << "\n==============================================\n";
}
