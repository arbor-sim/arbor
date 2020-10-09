#include "../gtest.h"
#include "common_cells.hpp"

#include "s_expr.hpp"

#include <arbor/cable_cell.hpp>
#include <arbor/cable_cell_param.hpp>
#include <arbor/morph/io.hpp>

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
    cable_cell c(tree, dict);
    auto mech = mechanism_desc{"hh"};
    mech.set("foo", 12);
    mech.set("bar", 1.2);
    c.paint("(tag 2)", axial_resistivity{23});
    c.paint("\"soma\"", membrane_capacitance{1.2});
    c.paint("\"dend\"", "hh");
    c.paint("\"axon\"", mech);
    c.place("(terminal)", threshold_detector{-10});
    c.set_default(arb::neuron_parameter_defaults);
    c.set_default(init_membrane_potential{-60});

    write_s_expr(std::cout, c) << "\n==============================================\n";

    cable_cell c2(tree, dict);

    write_s_expr(std::cout, c2) << "\n==============================================\n";
    c2.decorate(c.decorations());
    write_s_expr(std::cout, c2) << "\n==============================================\n";
}

TEST(cable_cell, printer) {
    using namespace arb::s_expr_literals;
    std::cout << slist() << "\n";
    std::cout << "-------\n";
    std::cout << slist(42) << "\n";
    std::cout << "-------\n";
    std::cout << slist(21, 21) << "\n";
    std::cout << "-------\n";
    std::cout << slist(1, 2, slist(1), 3) << "\n";
    std::cout << "-------\n";
    std::cout << slist("transform"_symbol,
                       slist("all"_symbol),
                       slist("join"_symbol,
                             slist("tag"_symbol, 1),
                             slist("tag"_symbol, 2),
                             1, 2, 3))
               << "\n";
}
