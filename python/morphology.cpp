#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <fstream>

#include <arbor/morph/morphology.hpp>
#include <arbor/morph/primitives.hpp>
#include <arbor/morph/segment_tree.hpp>
#include <arbor/swcio.hpp>

#include "error.hpp"
#include "strprintf.hpp"

namespace pyarb {

arb::segment_tree load_swc_allen(const std::string& fname, bool no_gaps=false) {
        std::ifstream fid{fname};
        if (!fid.good()) {
            throw pyarb_error(util::pprintf("can't open file '{}'", fname));
        }
        try {
            using namespace arb;
            auto records = parse_swc_file(fid);

            // Assert that the file contains at least one sample.
            if (records.empty()) {
                throw pyarb_error("Allen SWC: empty file");
            }

            // assert that root sample has tag 1.
            if (records[0].tag!=1) {
                throw pyarb_error("Allen SWC: the soma record does not have tag 1");
            }

            // Assert that all non-root samples have a tag of 2, 3, or 4.
            auto it = std::find_if(
                        records.begin()+1, records.end(),
                        [](auto& r){return r.tag<2 || r.tag>4;});
            if (it!=records.end()) {
                throw pyarb_error(
                        "Allen SWC: every record must have a tag of 2, 3 or 4, except for the first which must have tag 1");
            }

            // Assert that all samples have the same tag as their parent, except
            // those attached to the soma.
            it = std::find_if(
                        records.begin()+1, records.end(),
                        [&records](auto& r){auto p = r.parent_id; return p && r.tag!=records[p].tag;});
            if (it!=records.end()) {
                throw pyarb_error(
                        "Allen SWC: every record not attached to the soma must have the same tag as its parent");
            }

            // Translate the morphology so that the soma is centered at the origin (0,0,0)
            mpoint sloc{records[0].x, records[0].y, records[0].z, records[0].r};
            for (auto& r: records) {
                r.x -= sloc.x;
                r.y -= sloc.y;
                r.z -= sloc.z;
            }

            segment_tree tree;

            // Model the spherical soma as a cylinder with length=2*radius.
            // The cylinder is centred on the origin, and extended along the y axis.
            double soma_rad = sloc.radius;
            tree.append(mnpos, {0, -soma_rad, 0, soma_rad}, {0, soma_rad, 0, soma_rad}, 1);

            // Build branches off soma.
            std::unordered_map<msize_t, msize_t> pmap;
            std::set<msize_t> unused_samples;
            const auto nrec = records.size();
            for (unsigned i=1; i<nrec; ++i) {
                const auto& r = records[i];
                // If sample i has the root as its parent don't create a segment.
                if (r.parent_id==0) {
                    if (no_gaps) {
                        // Assert that this branch starts on the "surface" of the spherical soma.
                        auto d = std::fabs(soma_rad - std::sqrt(r.x*r.x + r.y*r.y + r.z*r.z));
                        if (d>1e-3) { // 1 nm tolerance
                            throw pyarb_error("Allen SWC: no gaps are allowed between the soma and any axons, dendrites or apical dendrites");
                        }
                    }
                    // This maps axons and apical dendrites to soma.prox, and dendrites to soma.dist.
                    pmap[i] = r.tag==3? 0: mnpos;
                    unused_samples.insert(i);
                    continue;
                }

                const auto p = r.parent_id;
                const auto& prox = records[p];
                const auto& dist = records[i];
                tree.append(pmap.at(p), {prox.x, prox.y, prox.z, prox.r}, {dist.x, dist.y, dist.z, dist.r}, r.tag);
                pmap[i] = tree.size() - 1;
                unused_samples.erase(p);
            }

            if (!unused_samples.empty()) {
                throw pyarb_error("Allen SWC: Every branch must contain at least one segment");
            }

            return tree;
        }
        catch (arb::swc_error& e) {
            // Try to produce helpful error messages for SWC parsing errors.
            throw pyarb_error(
                util::pprintf("Allen SWC: error parsing line {} of '{}': {}",
                              e.line_number, fname, e.what()));
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
                "Indicates whether the sample tree is empty (i.e. whether it has size 0)")
        .def_property_readonly("size", [](const arb::segment_tree& st){return st.size();},
                "The number of samples in the sample tree.")
        .def_property_readonly("parents", [](const arb::segment_tree& st){return st.parents();},
                "A list with the parent index of each sample.")
        .def_property_readonly("segments", [](const arb::segment_tree& st){return st.segments();},
                "A list of the samples.")
        .def("__str__", [](const arb::segment_tree& s) {
                return util::pprintf("<arbor.segment_tree:\n{}>", s);});

    // Function that creates a segment_tree from an swc file.
    // Wraps calls to C++ functions arb::parse_swc_file() and arb::swc_as_segment_tree().
    m.def("load_swc",
        [](std::string fname) {
            std::ifstream fid{fname};
            if (!fid.good()) {
                throw pyarb_error(util::pprintf("can't open file '{}'", fname));
            }
            try {
                auto records = arb::parse_swc_file(fid);
                return arb::swc_as_segment_tree(records);
            }
            catch (arb::swc_error& e) {
                // Try to produce helpful error messages for SWC parsing errors.
                throw pyarb_error(
                    util::pprintf("error parsing line {} of '{}': {}.",
                                  e.line_number, fname, e.what()));
            }
        },
        "Load an swc file and convert to a segment_tree.");

    m.def("load_swc_allen", &load_swc_allen,
            "filename"_a, "no_gaps"_a=false,
            "Generate a segment tree from an SWC file following the rules prescribed by\n"
            "AllenDB and Sonata. Specifically:\n"
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
