#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <fstream>

#include <arbor/morph/morphology.hpp>
#include <arbor/morph/primitives.hpp>
#include <arbor/morph/segment_tree.hpp>

#include <arborio/swcio.hpp>

#include "error.hpp"
#include "strprintf.hpp"

namespace pyarb {

void check_trailing(std::istream& in, std::string fname) {
    if (!(in >> std::ws).eof()) {
        throw pyarb_error(util::pprintf("Trailing data found at end of file '{}'", fname));
    }
}

void register_morphology(pybind11::module& m) {
    using namespace pybind11::literals;

    //
    //  primitives: points, segments, locations, cables... etc.
    //

    m.attr("mnpos") = arb::mnpos;

    // arb::mlocation
    pybind11::class_<arb::mlocation> location(m, "location",
        "A location on a cable cell.");
    location
        .def(pybind11::init(
            [](arb::msize_t branch, double pos) {
                const arb::mlocation mloc{branch, pos};
                pyarb::assert_throw(arb::test_invariants(mloc), "invalid location");
                return mloc;
            }),
            "branch"_a, "pos"_a,
            "Construct a location specification holding:\n"
            "  branch:   The id of the branch.\n"
            "  pos:      The relative position (from 0., proximal, to 1., distal) on the branch.\n")
        .def_readonly("branch",  &arb::mlocation::branch,
            "The id of the branch.")
        .def_readonly("pos", &arb::mlocation::pos,
            "The relative position on the branch (∈ [0.,1.], where 0. means proximal and 1. distal).")
        .def("__str__",
            [](arb::mlocation l) { return util::pprintf("(location {} {})", l.branch, l.pos); })
        .def("__repr__",
            [](arb::mlocation l) { return util::pprintf("(location {} {})", l.branch, l.pos); });

    // arb::mpoint
    pybind11::class_<arb::mpoint> mpoint(m, "mpoint");
    mpoint
        .def(pybind11::init(
                [](double x, double y, double z, double r) {
                    return arb::mpoint{x,y,z,r};
                }),
                "x"_a, "y"_a, "z"_a, "radius"_a, "All values in μm.")
        .def_readonly("x", &arb::mpoint::x, "X coordinate [μm].")
        .def_readonly("y", &arb::mpoint::y, "Y coordinate [μm].")
        .def_readonly("z", &arb::mpoint::z, "Z coordinate [μm].")
        .def_readonly("radius", &arb::mpoint::radius,
            "Radius of cable at sample location centered at coordinates [μm].")
        .def("__str__",
            [](const arb::mpoint& p) {
                return util::pprintf("<arbor.mpoint: x {}, y {}, z {}, radius {}>", p.x, p.y, p.z, p.radius);
            })
        .def("__repr__",
            [](const arb::mpoint& p) {return util::pprintf("{}>", p);});

    // arb::msegment
    pybind11::class_<arb::msegment> msegment(m, "msegment");
    msegment
        .def_readonly("prox", &arb::msegment::prox, "the location and radius of the proximal end.")
        .def_readonly("dist", &arb::msegment::dist, "the location and radius of the distal end.")
        .def_readonly("tag", &arb::msegment::tag, "tag meta-data.");

    // arb::mcable
    pybind11::class_<arb::mcable> cable(m, "cable");
    cable
        .def(pybind11::init(
            [](arb::msize_t bid, double prox, double dist) {
                arb::mcable c{bid, prox, dist};
                if (!test_invariants(c)) {
                    throw pyarb_error("Invalid cable description. Cable segments must have proximal and distal end points in the range [0,1].");
                }
                return c;
            }),
            "branch"_a, "prox"_a, "dist"_a)
        .def_readonly("branch", &arb::mcable::branch,
                "The id of the branch on which the cable lies.")
        .def_readonly("prox", &arb::mcable::prox_pos,
                "The relative position of the proximal end of the cable on its branch ∈ [0,1].")
        .def_readonly("dist", &arb::mcable::dist_pos,
                "The relative position of the distal end of the cable on its branch ∈ [0,1].")
        .def("__str__", [](const arb::mcable& c) { return util::pprintf("{}", c); })
        .def("__repr__", [](const arb::mcable& c) { return util::pprintf("{}", c); });

    //
    // Higher-level data structures (segment_tree, morphology)
    //

    // arb::segment_tree
    pybind11::class_<arb::segment_tree> segment_tree(m, "segment_tree");
    segment_tree
        // constructors
        .def(pybind11::init<>())
        // modifiers
        .def("reserve", &arb::segment_tree::reserve)
        .def("append", [](arb::segment_tree& t, arb::msize_t parent, arb::mpoint prox, arb::mpoint dist, int tag) {
                            return t.append(parent, prox, dist, tag);
                          },
                "parent"_a, "prox"_a, "dist"_a, "tag"_a,
                "Append a segment to the tree.")
        .def("append", [](arb::segment_tree& t, arb::msize_t parent, arb::mpoint dist, int tag) {
                            return t.append(parent, dist, tag);
                          },
                "parent"_a, "dist"_a, "tag"_a,
                "Append a segment to the tree.")
        .def("append",
                [](arb::segment_tree& t, arb::msize_t p, double x, double y, double z, double radius, int tag) {
                    return t.append(p, arb::mpoint{x,y,z,radius}, tag);
                },
                "parent"_a, "x"_a, "y"_a, "z"_a, "radius"_a, "tag"_a,
                "Append a segment to the tree, using the distal location of the parent segment as the proximal end.")
        // properties
        .def_property_readonly("empty", [](const arb::segment_tree& st){return st.empty();},
                "Indicates whether the tree is empty (i.e. whether it has size 0)")
        .def_property_readonly("size", [](const arb::segment_tree& st){return st.size();},
                "The number of segments in the tree.")
        .def_property_readonly("parents", [](const arb::segment_tree& st){return st.parents();},
                "A list with the parent index of each segment.")
        .def_property_readonly("segments", [](const arb::segment_tree& st){return st.segments();},
                "A list of the segments.")
        .def("__str__", [](const arb::segment_tree& s) {
                return util::pprintf("<arbor.segment_tree:\n{}>", s);});

    // Function that creates a morphology from an swc file.
    // Wraps calls to C++ functions arborio::parse_swc() and arborio::load_swc_arbor().
    m.def("load_swc",
        [](std::string fname) {
            std::ifstream fid{fname};
            if (!fid.good()) {
                throw pyarb_error(util::pprintf("can't open file '{}'", fname));
            }
            try {
                auto data = arborio::parse_swc(fid);
                check_trailing(fid, fname);
                return arborio::load_swc_arbor(data);
            }
            catch (arborio::swc_error& e) {
                // Try to produce helpful error messages for SWC parsing errors.
                throw pyarb_error(util::pprintf("error parsing {}: {}", fname, e.what()));
            }
        },
        "filename"_a,
        "Generate a morphology from an SWC file following the rules prescribed by Arbor.\n"
        "Specifically:\n"
        "* Single-segment somas are disallowed. These are usually interpreted as spherical somas\n"
        "  and are a special case. This behavior is not allowed using this SWC loader.\n"
        "* There are no special rules related to somata. They can be one or multiple branches\n"
        "  and other segments can connect anywhere along them.\n"
        "* A segment is always created between a sample and its parent, meaning there\n"
        "  are no gaps in the resulting morphology.");

    m.def("load_swc_allen",
        [](std::string fname, bool no_gaps=false) {
            std::ifstream fid{fname};
            if (!fid.good()) {
                throw pyarb_error(util::pprintf("can't open file '{}'", fname));
            }
            try {
                auto data = arborio::parse_swc(fid);
                check_trailing(fid, fname);
                return arborio::load_swc_allen(data, no_gaps);

            }
            catch (arborio::swc_error& e) {
                // Try to produce helpful error messages for SWC parsing errors.
                throw pyarb_error(
                        util::pprintf("Allen SWC: error parsing {}: {}", fname, e.what()));
            }
        },
        "filename"_a, "no_gaps"_a=false,
        "Generate a morphology from an SWC file following the rules prescribed by AllenDB\n"
        " and Sonata. Specifically:\n"
        "* The first sample (the root) is treated as the center of the soma.\n"
        "* The first morphology is translated such that the soma is centered at (0,0,0).\n"
        "* The first sample has tag 1 (soma).\n"
        "* All other samples have tags 2, 3 or 4 (axon, apic and dend respectively)\n"
        "SONATA prescribes that there should be no gaps, however the models in AllenDB\n"
        "have gaps between the start of sections and the soma. The flag no_gaps can be\n"
        "used to enforce this requirement.\n"
        "\n"
        "Arbor does not support modelling the soma as a sphere, so a cylinder with length\n"
        "equal to the soma diameter is used. The cylinder is centered on the origin, and\n"
        "aligned along the z axis.\n"
        "Axons and apical dendrites are attached to the proximal end of the cylinder, and\n"
        "dendrites to the distal end, with a gap between the start of each branch and the\n"
        "end of the soma cylinder to which it is attached.");

    m.def("load_swc_neuron",
        [](std::string fname) {
            std::ifstream fid{fname};
            if (!fid.good()) {
                throw pyarb_error(util::pprintf("can't open file '{}'", fname));
            }
            try {
                auto data = arborio::parse_swc(fid);
                check_trailing(fid, fname);
                return arborio::load_swc_neuron(data);
            }
            catch (arborio::swc_error& e) {
                // Try to produce helpful error messages for SWC parsing errors.
                throw pyarb_error(
                    util::pprintf("NEURON SWC: error parsing {}: {}", fname, e.what()));
            }
        },
        "filename"_a,
        "Generate a morphology from an SWC file following the rules prescribed by NEURON.\n"
        " Specifically:\n"
        "* The first sample must be a soma sample.\n"
        "* The soma is represented by a series of n≥1 unbranched, serially listed samples.\n"
        "* The soma is constructed as a single cylinder with diameter equal to the piecewise\n"
        "  average diameter of all the segments forming the soma.\n"
        "* A single-sample soma at is constructed as a cylinder with length=diameter.\n"
        "* If a non-soma sample is to have a soma sample as its parent, it must have the\n"
        "  most distal sample of the soma as the parent.\n"
        "* Every non-soma sample that has a soma sample as its parent, attaches to the\n"
        "  created soma cylinder at its midpoint.\n"
        "* If a non-soma sample has a soma sample as its parent, no segment is created\n"
        "  between the sample and its parent, instead that sample is the proximal point of\n"
        "  a new segment, and there is a gap in the morphology (represented electrically as a\n"
        "  zero-resistance wire)\n"
        "* To create a segment with a certain tag, that is to be attached to the soma,\n"
        "  we need at least 2 samples with that tag."
        );

    // arb::morphology

    pybind11::class_<arb::morphology> morph(m, "morphology");
    morph
        // constructors
        .def(pybind11::init(
                [](arb::segment_tree t){
                    return arb::morphology(std::move(t));
                }))
        // morphology's interface is read-only by design, so most of it can
        // be implemented as read-only properties.
        .def_property_readonly("empty",
                [](const arb::morphology& m){return m.empty();},
                "Whether the morphology is empty.")
        .def_property_readonly("num_branches",
                [](const arb::morphology& m){return m.num_branches();},
                "The number of branches in the morphology.")
        .def("branch_parent", &arb::morphology::branch_parent,
                "i"_a, "The parent branch of branch i.")
        .def("branch_children", &arb::morphology::branch_children,
                "i"_a, "The child branches of branch i.")
        .def("branch_segments",
                [](const arb::morphology& m, arb::msize_t i) {
                    return m.branch_segments(i);
                },
                "i"_a, "A list of the segments in branch i, ordered from proximal to distal ends of the branch.")
        .def("__str__",
                [](const arb::morphology& m) {
                    return util::pprintf("<arbor.morphology:\n{}>", m);
                });
}

} // namespace pyarb
