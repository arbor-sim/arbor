#include <fstream>
#include <tuple>

#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

#include <arbor/morph/morphology.hpp>
#include <arbor/morph/place_pwlin.hpp>
#include <arbor/morph/primitives.hpp>
#include <arbor/morph/segment_tree.hpp>

#include <arborio/swcio.hpp>

#include "error.hpp"
#include "strprintf.hpp"

namespace py = pybind11;

namespace pyarb {

static inline bool cable_lt(const arb::mcable& a, const arb::mcable& b) {
    return std::tuple(a.branch, a.prox_pos, a.dist_pos)<std::tuple(b.branch, b.prox_pos, b.dist_pos);
}

void check_trailing(std::istream& in, std::string fname) {
    if (!(in >> std::ws).eof()) {
        throw pyarb_error(util::pprintf("Trailing data found at end of file '{}'", fname));
    }
}

void register_morphology(py::module& m) {
    using namespace py::literals;

    //
    //  primitives: points, segments, locations, cables... etc.
    //

    m.attr("mnpos") = arb::mnpos;

    // arb::mlocation
    py::class_<arb::mlocation> location(m, "location",
        "A location on a cable cell.");
    location
        .def(py::init(
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
        .def(py::self==py::self)
        .def("__str__",
            [](arb::mlocation l) { return util::pprintf("(location {} {})", l.branch, l.pos); })
        .def("__repr__",
            [](arb::mlocation l) { return util::pprintf("(location {} {})", l.branch, l.pos); });

    // arb::mpoint
    py::class_<arb::mpoint> mpoint(m, "mpoint");
    mpoint
        .def(py::init<double, double, double, double>(),
             "x"_a, "y"_a, "z"_a, "radius"_a,
             "Create an mpoint object from parameters x, y, z, and radius, specified in µm.")
        .def(py::init([](py::tuple t) {
                if (py::len(t)!=4) throw std::runtime_error("tuple length != 4");
                return arb::mpoint{t[0].cast<double>(), t[1].cast<double>(), t[2].cast<double>(), t[3].cast<double>()}; }),
             "Create an mpoint object from a tuple (x, y, z, radius), specified in µm.")
        .def_readonly("x", &arb::mpoint::x, "X coordinate [μm].")
        .def_readonly("y", &arb::mpoint::y, "Y coordinate [μm].")
        .def_readonly("z", &arb::mpoint::z, "Z coordinate [μm].")
        .def_readonly("radius", &arb::mpoint::radius,
            "Radius of cable at sample location centered at coordinates [μm].")
        .def(py::self==py::self)
        .def("__str__",
            [](const arb::mpoint& p) {
                return util::pprintf("<arbor.mpoint: x {}, y {}, z {}, radius {}>", p.x, p.y, p.z, p.radius);
            })
        .def("__repr__",
            [](const arb::mpoint& p) {return util::pprintf("{}", p);});

    py::implicitly_convertible<py::tuple, arb::mpoint>();

    // arb::msegment
    py::class_<arb::msegment> msegment(m, "msegment");
    msegment
        .def_readonly("prox", &arb::msegment::prox, "the location and radius of the proximal end.")
        .def_readonly("dist", &arb::msegment::dist, "the location and radius of the distal end.")
        .def_readonly("tag", &arb::msegment::tag, "tag meta-data.");

    // arb::mcable
    py::class_<arb::mcable> cable(m, "cable");
    cable
        .def(py::init(
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
        .def(py::self==py::self)
        .def("__str__", [](const arb::mcable& c) { return util::pprintf("{}", c); })
        .def("__repr__", [](const arb::mcable& c) { return util::pprintf("{}", c); });

    // arb::isometry
    py::class_<arb::isometry> isometry(m, "isometry");
    isometry
        .def(py::init<>(), "Construct a trivial isometry.")
        .def("__call__", [](arb::isometry& iso, arb::mpoint& p) {
                return iso.apply(p);
            },
            "Apply isometry to mpoint argument.")
        .def("__call__", [](arb::isometry& iso, py::tuple t) {
                int len = py::len(t);
                if (len<3) throw std::runtime_error("tuple length < 3");

                arb::mpoint p{t[0].cast<double>(), t[1].cast<double>(), t[2].cast<double>(), 0.};
                p = iso.apply(p);

                py::tuple result(len);
                result[0] = p.x;
                result[1] = p.y;
                result[2] = p.z;
                for (int i = 3; i<len; ++i) {
                    result[i] = t[i];
                }
                return result;
            },
            "Apply isometry to first three components of tuple argument.")
        .def(py::self*py::self)
        .def_static("translate",
            [](double x, double y, double z) { return arb::isometry::translate(x, y, z); },
            "x"_a, "y"_a, "z"_a,
            "Construct a translation isometry from displacements x, y, and z.")
        .def_static("translate",
            [](py::tuple t) {
                if (py::len(t)!=3) throw std::runtime_error("tuple length != 3");
                return arb::isometry::translate(t[0].cast<double>(), t[1].cast<double>(), t[2].cast<double>());
            },
            "Construct a translation isometry from the first three components of a tuple.")
        .def_static("translate",
            [](arb::mpoint p) { return arb::isometry::translate(p.x, p.y, p.z); },
            "Construct a translation isometry from the x, y, and z components of an mpoint.")
        .def_static("rotate",
            [](double theta, double x, double y, double z) { return arb::isometry::rotate(theta, x, y, z); },
            "theta"_a, "x"_a, "y"_a, "z"_a,
            "Construct a rotation isometry of angle theta about the axis in direction (x, y, z).")
        .def_static("rotate",
            [](double theta, py::tuple t) {
                if (py::len(t)!=3) throw std::runtime_error("tuple length != 3");
                return arb::isometry::rotate(theta, t[0].cast<double>(), t[1].cast<double>(), t[2].cast<double>());
            },
            "theta"_a, "axis"_a,
            "Construct a rotation isometry of angle theta about the given axis in the direction described by a tuple.");

    // arb::place_pwlin
    py::class_<arb::place_pwlin> place(m, "place_pwlin");
    place
        .def(py::init<const arb::morphology&, const arb::isometry&>(),
            "morphology"_a, "isometry"_a=arb::isometry{},
            "Construct a piecewise-linear placement object from the given morphology and optional isometry.")
        .def("at", &arb::place_pwlin::at, "location"_a,
            "Return an interpolated mpoint corresponding to the location argument.")
        .def("all_at", &arb::place_pwlin::all_at, "location"_a,
            "Return list of all possible interpolated mpoints corresponding to the location argument.")
        .def("segments",
            [](const arb::place_pwlin& self, std::vector<arb::mcable> cables) {
                std::sort(cables.begin(), cables.end(), cable_lt);
                return self.segments(cables);
            },
            "Return minimal list of full or partial msegments whose union is coterminous "
            "with the extent of the given list of cables.")
        .def("all_segments",
            [](const arb::place_pwlin& self, std::vector<arb::mcable> cables) {
                std::sort(cables.begin(), cables.end(), cable_lt);
                return self.all_segments(cables);
            },
            "Return maximal list of non-overlapping full or partial msegments whose union is coterminous "
            "with the extent of the given list of cables.");

    //
    // Higher-level data structures (segment_tree, morphology)
    //

    // arb::segment_tree
    py::class_<arb::segment_tree> segment_tree(m, "segment_tree");
    segment_tree
        // constructors
        .def(py::init<>())
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
    m.def("load_swc_arbor",
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

    py::class_<arb::morphology> morph(m, "morphology");
    morph
        // constructors
        .def(py::init(
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
