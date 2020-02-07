#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <fstream>

#include <arbor/morph/morphology.hpp>
#include <arbor/morph/primitives.hpp>
#include <arbor/morph/sample_tree.hpp>
#include <arbor/swcio.hpp>

#include "error.hpp"
#include "strprintf.hpp"

namespace pyarb {

void register_morphology(pybind11::module& m) {
    using namespace pybind11::literals;

    //
    //  primitives: points, samples, locations, cables... etc.
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
            "branch"_a, "position"_a,
            "Construct a location specification holding:\n"
            "  branch:   The id of the branch.\n"
            "  position: The relative position (from 0., proximal, to 1., distal) on the branch.\n")
        .def_readonly("branch",  &arb::mlocation::branch,
            "The id of the branch.")
        .def_readonly("position", &arb::mlocation::pos,
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

    // arb::msample
    pybind11::class_<arb::msample> msample(m, "msample");
    msample
        .def(pybind11::init(
            [](arb::mpoint loc, int tag){
                return arb::msample{loc, tag};}),
            "location"_a, "tag"_a)
        .def(pybind11::init(
            [](double x, double y, double z, double r, int tag){
                return arb::msample{{x,y,z,r}, tag};}),
            "x"_a, "y"_a, "z"_a, "radius"_a, "tag"_a,
            "spatial values {x, y, z, radius} are in μm.")
        .def_readonly("loc", &arb::msample::loc,
            "Location of sample.")
        .def_readonly("tag", &arb::msample::tag,
            "Property tag of sample. "
            "If loaded from standard SWC file the following tags are used: soma=1, axon=2, dendrite=3, apical dendrite=4, however arbitrary tags can be used.")
        .def("__str__",
            [](const arb::msample& s) {
                return util::pprintf("<arbor.mpoint: x {}, y {}, z {}, radius {}, tag {}>",
                        s.loc.x, s.loc.y, s.loc.z, s.loc.radius, s.tag); })
        .def("__repr__",
            [](const arb::msample& s) {
                return util::pprintf("{}>", s);});

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
             "branch_id"_a, "prox"_a, "dist"_a)
        .def_readonly("prox_pos", &arb::mcable::prox_pos,
                "The relative position of the proximal end of the cable on its branch ∈ [0,1].")
        .def_readonly("dist_pos", &arb::mcable::dist_pos,
                "The relative position of the distal end of the cable on its branch ∈ [0,1].")
        .def_readonly("branch", &arb::mcable::branch,
                "The id of the branch on which the cable lies.")
        .def("__str__", [](const arb::mcable& c) {
            return util::pprintf("<arbor.cable: branch {}, prox {}, dist {}", c.branch, c.prox_pos, c.dist_pos); })
        .def("__repr__", [](const arb::mcable& c) { return util::pprintf("{}", c); });

    //
    // Higher-level data structures (sample_tree, morphology)
    //

    // arb::sample_tree
    pybind11::class_<arb::sample_tree> sample_tree(m, "sample_tree");
    sample_tree
        // constructors
        .def(pybind11::init<>())
        // modifiers
        .def("reserve", &arb::sample_tree::reserve)
        .def("append", [](arb::sample_tree& t, arb::msample s){return t.append(s);})
        .def("append", [](arb::sample_tree& t, arb::msize_t p, arb::msample s){return t.append(p, s);})
        // properties
        .def_property_readonly("empty", [](const arb::sample_tree& st){return st.empty();},
                "Indicates whether the sample tree is empty (i.e. whether it has size 0)")
        .def_property_readonly("size", [](const arb::sample_tree& st){return st.size();},
                "The number of samples in the sample tree.")
        .def_property_readonly("parents", [](const arb::sample_tree& st){return st.parents();},
                "A list with the parent index of each sample.")
        .def("__str__", [](const arb::sample_tree& s) {
                return util::pprintf("<arbor.sample_tree:\n{}>", s);});

    // Function that creates a sample_tree from an swc file.
    // Wraps calls to C++ functions arb::parse_swc_file() and arb::swc_as_sample_tree().
    m.def("load_swc",
        [](std::string fname) {
            std::ifstream fid{fname};
            if (!fid.good()) {
                throw pyarb_error(util::pprintf("can't open file '{}'", fname));
            }
            try {
                auto records = arb::parse_swc_file(fid);
                arb::swc_canonicalize(records);
                return arb::swc_as_sample_tree(records);
            }
            catch (arb::swc_error& e) {
                // Try to produce helpful error messages for SWC parsing errors.
                throw pyarb_error(
                    util::pprintf("error parsing line {} of '{}': {}.",
                                  e.line_number, fname, e.what()));
            }
        },
        "Load an swc file and convert to a sample_tree.");

    // arb::morphology

    pybind11::class_<arb::morphology> morph(m, "morphology");
    morph
        // constructors
        .def(pybind11::init(
                [](arb::sample_tree t){
                    return arb::morphology(std::move(t));
                }))
        .def(pybind11::init(
                [](arb::sample_tree t, bool spherical_root) {
                    return arb::morphology(std::move(t), spherical_root);
                }), "sample_tree"_a, "spherical_root"_a)
        // morphology's interface is read-only by design, so most of it can
        // be implemented as read-only properties.
        .def_property_readonly("empty",
                [](const arb::morphology& m){return m.empty();},
                "A list with the parent index of each sample.")
        .def_property_readonly("spherical_root",
                [](const arb::morphology& m){return m.spherical_root();},
                "Whether the root of the morphology is spherical.")
        .def_property_readonly("num_branches",
                [](const arb::morphology& m){return m.num_branches();},
                "The number of branches in the morphology.")
        .def_property_readonly("num_samples",
                [](const arb::morphology& m){return m.num_samples();},
                "The number of samples in the morphology.")
        .def_property_readonly("samples",
                [](const arb::morphology& m){return m.samples();},
                "All of the samples in the morphology.")
        .def_property_readonly("sample_parents",
                [](const arb::morphology& m){return m.sample_parents();},
                "The parent indexes of each sample.")
        .def("branch_parent", &arb::morphology::branch_parent,
                "i"_a, "The parent branch of branch i.")
        .def("branch_children", &arb::morphology::branch_children,
                "i"_a, "The child branches of branch i.")
        .def("branch_indexes",
                [](const arb::morphology& m, arb::msize_t i) {
                    auto p = m.branch_indexes(i);
                    return std::vector<arb::msize_t>(p.first, p.second);
                },
                "i"_a, "Range of indexes into the sample points in branch i.")
        .def("__str__",
                [](const arb::morphology& m) {
                    return util::pprintf("<arbor.morphology:\n{}>", m);
                });
}

} // namespace pyarb
