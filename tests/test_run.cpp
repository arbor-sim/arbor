#include "gtest.h"

#include "../src/cell.hpp"

TEST(run, init)
{
    using namespace nestmc;

    nestmc::cell cell;

    cell.add_soma(12.6157/2.0);
    //auto& props = cell.soma()->properties;

    cell.add_cable(0, segmentKind::dendrite, 0.5, 0.5, 200);

    EXPECT_EQ(cell.graph().num_segments(), 2u);

    for(auto &s : cell.segments()) {
        std::cout << "volume : " << s->volume()
                  << " area : " << s->area()
                  << " ratio : " << s->volume()/s->area() << std::endl;
    }

#ifdef example_1

    // in this context (i.e. attached to a segment on a high-level cell)
    // a mechanism is essentially a set of parameters
    // - the only "state" is that used to define parameters
    cell.soma()->add_mechanism("hh");

    auto& soma_hh = cell.soma()->mechanisms("hh");
    soma_hh.set("gnabar", 0.12);
    soma_hh.set("gkbar", 0.036);
    soma_hh.set("gl", 0.0003);
    soma_hh.set("el", -54.3);

    fvm_cell fv(cell);
#endif
}
