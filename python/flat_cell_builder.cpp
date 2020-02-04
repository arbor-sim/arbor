#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <arbor/cable_cell.hpp>
#include <arbor/morph/morphology.hpp>
#include <arbor/morph/primitives.hpp>
#include <arbor/morph/sample_tree.hpp>

#include "conversion.hpp"
#include "error.hpp"
#include "morph_parse.hpp"
#include "s_expr.hpp"
#include "strprintf.hpp"

namespace pyarb {

class flat_cell_builder {
    // The sample tree describing the morphology: constructed on the fly as
    // the cables/spheres are added with add_cable/add_sphere.
    arb::sample_tree tree_;

    // The distal sample id of each cable.
    std::vector<arb::msize_t> cable_distal_id_;

    // The number of unique region names used to label cables as they
    // are added to the cell.
    int tag_count_ = 0;
    // Map from region names to the tag used to identify them.
    std::unordered_map<std::string, int> tag_map_;

    arb::label_dict dict_;

    // The morphology is cached, and only updated on request when it is out of date.
    mutable bool cached_morpho_ = true;
    mutable arb::morphology morpho_;
    mutable std::mutex mutex_;

    // Set on construction and unchanged thereafter.
    // Indicates whether 
    bool spherical_ = false;

public:

    flat_cell_builder() = default;

    arb::msize_t add_sphere(double radius, const char* name) {
        cached_morpho_ = false;
        spherical_ = true;
        if (size()) {
            throw pyarb_error("Add soma to non-empty cell.");
        }
        tree_.append({{0,0,0,radius}, get_tag(name)});
        cable_distal_id_.push_back(0);
        return 0;
    }

    // Add a new branch that is attached to parent.
    // Returns the id of the new cable.
    arb::msize_t add_cable(arb::msize_t parent, double len,
                           double r1, double r2, const char* region, int ncomp)
    {
        using arb::mnpos;

        cached_morpho_ = false;

        if (!test_identifier(region)) {
            throw pyarb_error(util::pprintf("'{}' is not a valid label name.", region));
        }

        // Get tag id of region (add a new tag if region does not already exist).
        int tag = get_tag(region);
        const bool at_root = parent==mnpos;

        // Can't attach a cable to the root on a cell with spherical soma.
        if (at_root && spherical_) {
            throw pyarb_error("Invalid parent id.");
        }
        // Parent id must be in the range [0, size())
        if (!at_root && parent>=size()) {
            throw pyarb_error("Invalid parent id.");
        }

        // Calculating the sample that is the parent of this branch is
        // neccesarily complicated to handle cable segments that are attached
        // to the root.
        arb::msize_t p = at_root? (size()? 0: mnpos): cable_distal_id_[parent];

        double z = at_root? 0:                      // attach to root of non-spherical cell
                   spherical_&&!parent? soma_rad(): // attach to spherical root
                   tree_.samples()[p].loc.z;        // attach to end of a cable

        // Only add a first point at the very beginning of the cable if
        // the cable is not attached to another
        const bool add_first_point = p==arb::mnpos      // attached to the "root"
                                  || (!p && spherical_) // attached to a spherical root
                                  || (r1!=tree_.samples()[p].loc.radius);
                                                        // proximal radius does not match r1
        if (add_first_point) {
            p = tree_.append(p, {{0,0,z,r1}, tag});
        }
        if (ncomp>1) {
            double dz = len/ncomp;
            double dr = (r2-r1)/ncomp;
            for (auto i=1; i<ncomp; ++i) {
                p = tree_.append(p, {{0,0,z+i*dz, r1+i*dr}, tag});
            }
        }
        p = tree_.append(p, {{0,0,z+len,r2}, tag});
        cable_distal_id_.push_back(p);

        return size()-1;
    }

    void add_label(const char* name, const char* description) {
        if (!test_identifier(name)) {
            throw pyarb_error(util::pprintf("'{}' is not a valid label name.", name));
        }

        if (auto result = eval(parse(description)) ) {
            // The description is a region.
            if (result->type()==typeid(arb::region)) {
                if (dict_.locset(name)) {
                    throw pyarb_error("Region name clashes with a locset.");
                }
                auto& reg = arb::util::any_cast<arb::region&>(*result);
                if (auto r = dict_.region(name)) {
                    dict_.set(name, join(std::move(reg), std::move(*r)));
                }
                else {
                    dict_.set(name, std::move(reg));
                }
            }
            else if (result->type()==typeid(arb::locset)) {
                if (dict_.region(name)) {
                    throw pyarb_error("Locset name clashes with a region.");
                }
                auto& loc = arb::util::any_cast<arb::locset&>(*result);
                if (auto l = dict_.locset(name)) {
                    dict_.set(name, sum(std::move(loc), std::move(*l)));
                }
                else {
                    dict_.set(name, std::move(loc));
                }
            }
            else {
                throw pyarb_error("Label describes neither a region nor a locset.");
            }
        }
        else {
            throw pyarb_error(result.error().message);
        }
    }

    const arb::sample_tree& samples() const {
        return tree_;
    }

    std::unordered_map<std::string, std::string> labels() const {
        std::unordered_map<std::string, std::string> map;
        for (auto& r: dict_.regions()) {
            map[r.first] = util::pprintf("{}", r.second);
        }
        for (auto& l: dict_.locsets()) {
            map[l.first] = util::pprintf("{}", l.second);
        }

        return map;
    }

    const arb::morphology& morphology() const {
        const std::lock_guard<std::mutex> guard(mutex_);
        if (!cached_morpho_) {
            morpho_ = arb::morphology(tree_, spherical_);
            cached_morpho_ = true;
        }
        return morpho_;
    }

    arb::cable_cell build() const {
        // Make cable_cell from sample tree and dictionary.
        auto c = arb::cable_cell(morphology(), dict_);
        c.default_parameters.discretization = arb::cv_policy_every_sample{};
        return c;
    }

    private:

    // Get tag id of region with name.
    // Add a new tag if region with that name has not already had a tag associated with it.
    int get_tag(const std::string& name) {
        using arb::reg::tagged;

        // Name is in the map: return the tag.
        auto it = tag_map_.find(name);
        if (it!=tag_map_.end()) {
            return it->second;
        }
        // If the name is not in the map, the next step depends on
        // whether the name is used for a locst, or a region that is
        // not in the map, or is not used at all.

        // Name is a locset: error.
        if (dict_.locset(name)) {
            throw pyarb_error(util::pprintf("'{}' is a label for a locset."));
        }
        // Name is a region: add tag to region definition.
        else if(auto reg = dict_.region(name)) {
            tag_map_[name] = ++tag_count_;
            dict_.set(name, join(*reg, tagged(tag_count_)));
            return tag_count_;
        }
        // Name has not been registerd: make a unique tag and new region.
        else {
            tag_map_[name] = ++tag_count_;
            dict_.set(name, tagged(tag_count_));
            return tag_count_;
        }
    }

    // Only valid if called on a non-empty tree with spherical soma.
    double soma_rad() const {
        return tree_.samples()[0].loc.radius;
    }

    // The number of cable segements (plus one optional soma) used to construct the cell.
    std::size_t size() const {
        return cable_distal_id_.size();
    }
};

void register_flat_builder(pybind11::module& m) {
    using namespace pybind11::literals;

    pybind11::class_<flat_cell_builder> builder(m, "flat_cell_builder");
    builder
        .def(pybind11::init<>())
        .def("add_sphere", &flat_cell_builder::add_sphere,
            "radius"_a, "name"_a)
        .def("add_cable",
                [](flat_cell_builder& b, arb::msize_t p, double len, pybind11::object rad, const char* name, int ncomp) {
                    using pybind11::isinstance;
                    using pybind11::cast;
                    if (auto radius = try_cast<double>(rad) ) {
                        return b.add_cable(p, len, *radius, *radius, name, ncomp);
                    }

                    if (auto radii = try_cast<std::pair<double, double>>(rad)) {
                        return b.add_cable(p, len, radii->first, radii->second, name, ncomp);
                    }
                    else {
                        throw pyarb_error(
                            "Radius parameter is not a scalar (constant branch radius) or "
                            "a tuple (radius at proximal and distal ends respectively).");
                    }
                },
            "parent"_a, "length"_a, "radius"_a, "name"_a, "ncomp"_a=1)
        .def("add_label", &flat_cell_builder::add_label,
            "name"_a, "description"_a)
        .def_property_readonly("samples",
            [](const flat_cell_builder& b) { return b.samples(); })
        .def_property_readonly("labels",
            [](const flat_cell_builder& b) { return b.labels(); })
        .def_property_readonly("morphology",
            [](const flat_cell_builder& b) { return b.morphology(); })
        .def("build", &flat_cell_builder::build);
}

} // namespace pyarb
