#include <fstream>
#include <tuple>
#include <variant>

#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

#include <arbor/morph/morphology.hpp>
#include <arbor/morph/place_pwlin.hpp>
#include <arbor/morph/primitives.hpp>
#include <arbor/morph/segment_tree.hpp>
#include <arbor/version.hpp>

#include <arborio/label_parse.hpp>
#include <arborio/swcio.hpp>
#include <arborio/neurolucida.hpp>
#include <arborio/neuroml.hpp>

#include "util.hpp"
#include "error.hpp"
#include "proxy.hpp"
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
            "Radius of cable at sample location centred at coordinates [μm].")
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
            "with the extent of the given list of cables.")
        .def("closest",
            [](const arb::place_pwlin& self, double x, double y, double z) {
                auto [l, d] = self.closest(x, y, z);
                return pybind11::make_tuple(l, d);
            },
            "Find the location on the morphology that is closest to a 3d point. "
            "Returns the location and its distance from the point.");

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
        .def("is_fork", &arb::segment_tree::is_fork,
                "i"_a, "True if segment has more than one child.")
        .def("is_terminal", &arb::segment_tree::is_terminal,
                "i"_a, "True if segment has no children.")
        .def("is_root", &arb::segment_tree::is_root,
                "i"_a, "True if segment has no parent.")
        // properties
        .def_property_readonly("empty", [](const arb::segment_tree& st){return st.empty();},
                "Indicates whether the tree is empty (i.e. whether it has size 0)")
        .def_property_readonly("size", [](const arb::segment_tree& st){return st.size();},
                "The number of segments in the tree.")
        .def_property_readonly("parents", [](const arb::segment_tree& st){return st.parents();},
                "A list with the parent index of each segment.")
        .def_property_readonly("segments", [](const arb::segment_tree& st){return st.segments();},
                "A list of the segments.")
        .def("apply_isometry",
             [](const arb::segment_tree& t, const arb::isometry& i) { return arb::apply(t, i); },
             "Apply an isometry to all segments in the tree.")
        .def("split_at",
             [](const arb::segment_tree& t, arb::msize_t id) { return arb::split_at(t, id); },
             "Split into a pair of trees at the given id, such that one tree is the subtree rooted at id and the other is the original tree without said subtree.")
        .def("join_at",
             [](const arb::segment_tree& t, arb::msize_t id, const arb::segment_tree& o) { return arb::join_at(t, id, o); },
             "Join two subtrees at a given id, such that said id becomes the parent of the inserted sub-tree.")
        .def("equivalent",
             [](const arb::segment_tree& t, const arb::segment_tree& o) { return arb::equivalent(t, o); },
             "Two trees are equivalent, but not neccessarily identical, ie they have the same segments and structure.")
        .def("tag_roots",
            [](const arb::segment_tree& t, int tag) { return arb::tag_roots(t, tag); },
            "Get roots of tag region of this segment tree.")
        .def("__str__", [](const arb::segment_tree& s) {
                return util::pprintf("<arbor.segment_tree:\n{}>", s);});

    using morph_or_tree = std::variant<arb::segment_tree, arb::morphology>;

    // Function that creates a morphology/segment_tree from an swc file.
    // Wraps calls to C++ functions arborio::parse_swc() and arborio::load_swc_arbor().
    m.def("load_swc_arbor",
        [](py::object fn, bool raw) -> morph_or_tree {
            try {
                auto contents = util::read_file_or_buffer(fn);
                auto data = arborio::parse_swc(contents);
                if (raw) {
                    return arborio::load_swc_arbor_raw(data);
                }
                return arborio::load_swc_arbor(data);
            }
            catch (arborio::swc_error& e) {
                // Try to produce helpful error messages for SWC parsing errors.
                throw pyarb_error(util::pprintf("Arbor SWC: parse error: {}", e.what()));
            }
        },
        "filename_or_stream"_a,
        pybind11::arg_v("raw", false, "Return a segment tree instead of a fully formed morphology"),
        "Generate a morphology/segment_tree from an SWC file following the rules prescribed by Arbor.\n"
        "Specifically:\n"
        " * Single-segment somas are disallowed.\n"
        " * There are no special rules related to somata. They can be one or multiple branches\n"
        "   and other segments can connect anywhere along them.\n"
        " * A segment is always created between a sample and its parent, meaning there\n"
        "   are no gaps in the resulting morphology.");
    m.def("load_swc_neuron",
        [](py::object fn, bool raw) -> morph_or_tree {
            try {
                auto contents = util::read_file_or_buffer(fn);
                auto data = arborio::parse_swc(contents);
                if (raw) {
                    return arborio::load_swc_neuron_raw(data);
                }
                return arborio::load_swc_neuron(data);
            }
            catch (arborio::swc_error& e) {
                // Try to produce helpful error messages for SWC parsing errors.
                throw pyarb_error(util::pprintf("NEURON SWC: parse error: {}", e.what()));
            }
        },
        "filename_or_stream"_a,
        pybind11::arg_v("raw", false, "Return a segment tree instead of a fully formed morphology"),
        "Generate a morphology from an SWC file following the rules prescribed by NEURON.\n"
        "See the documentation https://docs.arbor-sim.org/en/latest/fileformat/swc.html\n"
        "for a detailed description of the interpretation.");


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
        .def("to_segment_tree", &arb::morphology::to_segment_tree,
                "Convert this morphology to a segment_tree.")
        .def("__str__",
                [](const arb::morphology& m) {
                    return util::pprintf("<arbor.morphology:\n{}>", m);
                });

    // Neurolucida ASCII, or .asc, file format support.

    py::class_<arborio::asc_morphology> asc_morphology(m, "asc_morphology",
            "The morphology and label dictionary meta-data loaded from a Neurolucida ASCII (.asc) file.");
    asc_morphology
        .def_readonly("morphology",
                &arborio::asc_morphology::morphology,
                "The cable cell morphology.")
        .def_readonly("segment_tree",
                &arborio::asc_morphology::segment_tree,
                "The raw segment tree.")
        .def_property_readonly("labels",
            [](const arborio::asc_morphology& m) {return label_dict_proxy(m.labels);},
            "The four canonical regions are labeled 'soma', 'axon', 'dend' and 'apic'.");

    using asc_morph_or_tree = std::variant<arb::segment_tree, arborio::asc_morphology>;

    m.def("load_asc",
        [](py::object fn, bool raw) -> asc_morph_or_tree {
            try {
                auto contents = util::read_file_or_buffer(fn);
                if (raw) {
                    return arborio::parse_asc_string_raw(contents.c_str());
                }
                return arborio::parse_asc_string(contents.c_str());
            }
            catch (std::exception& e) {
                // Try to produce helpful error messages for SWC parsing errors.
                throw pyarb_error(util::pprintf("error loading neurolucida asc file: {}", e.what()));
            }
        },
        "filename_or_stream"_a,
        pybind11::arg_v("raw", false, "Return a segment tree instead of a fully formed morphology"),
        "Load a morphology or segment_tree and meta data from a Neurolucida ASCII .asc file.");

    // arborio::morphology_data
    py::class_<arborio::nml_morphology_data> nml_morph_data(m, "neuroml_morph_data");
    nml_morph_data
        .def_readonly("cell_id",
            &arborio::nml_morphology_data::cell_id,
            "Cell id, or empty if morphology was taken from a top-level <morphology> element.")
        .def_readonly("id",
            &arborio::nml_morphology_data::id,
            "Morphology id.")
        .def_readonly("morphology",
            &arborio::nml_morphology_data::morphology,
            "Morphology constructed from a signle NeuroML <morphology> element.")
        .def("segments",
            [](const arborio::nml_morphology_data& md) {return label_dict_proxy(md.segments);},
            "Label dictionary containing one region expression for each segment id.")
        .def("named_segments",
            [](const arborio::nml_morphology_data& md) {return label_dict_proxy(md.named_segments);},
            "Label dictionary containing one region expression for each name applied to one or more segments.")
        .def("groups",
            [](const arborio::nml_morphology_data& md) {return label_dict_proxy(md.groups);},
            "Label dictionary containing one region expression for each segmentGroup id.")
        .def_readonly("group_segments",
            &arborio::nml_morphology_data::group_segments,
            "Map from segmentGroup ids to their corresponding segment ids.");

    // arborio::neuroml
    py::class_<arborio::neuroml> neuroml(m, "neuroml");
    neuroml
        // constructors
        .def(py::init(
            [](py::object fn) {
                try {
                    auto contents = util::read_file_or_buffer(fn);
                    return arborio::neuroml(contents);
                }
                catch (arborio::neuroml_exception& e) {
                    // Try to produce helpful error messages for NeuroML parsing errors.
                    throw pyarb_error(util::pprintf("NeuroML error: {}", e.what()));
                }
            }),
            "Construct NML morphology from filename or stream.")
        .def("cell_ids",
            [](const arborio::neuroml& nml) {
                try {
                    return nml.cell_ids();
                }
                catch (arborio::neuroml_exception& e) {
                    throw util::pprintf("NeuroML error: {}", e.what());
                }
            },
            "Query top-level cells.")
        .def("morphology_ids",
            [](const arborio::neuroml& nml) {
                try {
                    return nml.morphology_ids();
                }
                catch (arborio::neuroml_exception& e) {
                    throw util::pprintf("NeuroML error: {}", e.what());
                }
            },
            "Query top-level standalone morphologies.")
        .def("morphology",
            [](const arborio::neuroml& nml, const std::string& morph_id, bool spherical) {
                try {
                    using namespace arborio::neuroml_options;
                    return nml.morphology(morph_id, spherical? allow_spherical_root: none);
                }
                catch (arborio::neuroml_exception& e) {
                    throw util::pprintf("NeuroML error: {}", e.what());
                }
            }, "morph_id"_a, "allow_spherical_root"_a=false,
            "Retrieve top-level nml_morph_data associated with morph_id.")
        .def("cell_morphology",
            [](const arborio::neuroml& nml, const std::string& cell_id, bool spherical) {
                try {
                    using namespace arborio::neuroml_options;
                    return nml.cell_morphology(cell_id, spherical? allow_spherical_root: none);
                }
                catch (arborio::neuroml_exception& e) {
                    throw util::pprintf("NeuroML error: {}", e.what());
                }
            }, "cell_id"_a, "allow_spherical_root"_a=false,
            "Retrieve nml_morph_data associated with cell_id.");
}

} // namespace pyarb
