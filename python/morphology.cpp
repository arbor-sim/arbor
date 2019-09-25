#include <pybind11/pybind11.h>

#include <arbor/morph/primitives.hpp>

#include "error.hpp"
#include "strprintf.hpp"

namespace pyarb {

void register_morphology(pybind11::module& m) {
    using namespace pybind11::literals;

    // segment location
    pybind11::class_<arb::mlocation> location(m, "location",
        "A location on a cable cell.");
    location
        .def(pybind11::init(
            [](arb::msize_t branch, double pos) {
                const arb::mlocation mloc{branch, pos};
                pyarb::assert_throw(arb::test_invariants(mloc), "invalid location");
                return mloc;
            }),
            "branch"_a, "position"_a,
            "Construct a location specification holding:\n"
            "  branch:   The id of the branch.\n"
            "  position: The relative position (from 0., proximal, to 1., distal) on the branch.\n")
        .def_readonly("branch",  &arb::mlocation::branch,  "The id of the branch.")
        .def_readonly("position", &arb::mlocation::pos, "The relative position on the branch (∈ [0.,1.], where 0. means proximal and 1. distal).")
        .def("__str__", [](arb::mlocation l) {return util::pprintf("<arbor.location: branch {}, position {}>", l.branch, l.pos);})
        .def("__repr__", [](arb::mlocation l) {return util::pprintf("<arbor.location: branch {}, position {}>", l.branch, l.pos);});

}
} // namespace pyarb
