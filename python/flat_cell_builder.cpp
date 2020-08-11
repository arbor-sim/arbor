#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <arbor/cable_cell.hpp>
#include <arbor/morph/morphology.hpp>
#include <arbor/morph/primitives.hpp>
#include <arbor/morph/segment_tree.hpp>

#include "conversion.hpp"
#include "error.hpp"
#include "morph_parse.hpp"
#include "s_expr.hpp"
#include "strprintf.hpp"

namespace pyarb {

class flat_cell_builder {
    // The segment tree describing the morphology, constructed additatively with
    // segments that are attached to existing segments using add_cable.
    arb::segment_tree tree_;

    // The number of unique region names used to label cables as they
    // are added to the cell.
    int tag_count_ = 0;
    // Map from region names to the tag used to identify them.
    std::unordered_map<std::string, int> tag_map_;

    std::vector<arb::msize_t> cable_distal_segs_;

    arb::label_dict dict_;

    // The morphology is cached, and only updated on request when it is out of date.
    mutable bool cached_morpho_ = true;
    mutable arb::morphology morpho_;
    mutable std::mutex mutex_;

public:

    flat_cell_builder() = default;


    // Add a new cable that is attached to the last cable added to the cell.
    // Returns the id of the new cable.
    arb::msize_t add_cable(double len,
                           double r1, double r2, const char* region, int ncomp)
    {
        return add_cable(size()? size()-1: arb::mnpos, len, r1, r2, region, ncomp);
    }

    // Add a new cable that is attached to the parent cable.
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

        // Parent id must be in the range [0, size())
        if (!at_root && parent>=size()) {
            throw pyarb_error("Invalid parent id.");
        }

        arb::msize_t p = at_root? mnpos: cable_distal_segs_[parent];

        double z = at_root? 0:                      // attach to root
                   tree_.segments()[p].dist.z;      // attach to end of a cable

        double dz = len/ncomp;
        double dr = (r2-r1)/ncomp;
        for (auto i=0; i<ncomp; ++i) {
            p = tree_.append(p, {0,0,z+i*dz, r1+i*dr}, {0,0,z+(i+1)*dz, r1+(i+1)*dr}, tag);
        }

        cable_distal_segs_.push_back(p);

        return cable_distal_segs_.size()-1;
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

    const arb::segment_tree& segments() const {
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
            morpho_ = arb::morphology(tree_);
            cached_morpho_ = true;
        }
        return morpho_;
    }

    arb::cable_cell build() const {
        auto c = arb::cable_cell(morphology(), dict_);
        c.default_parameters.discretization = arb::cv_policy_every_segment{};
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

    // The number of cable segements used in the cell.
    std::size_t size() const {
        return tree_.size();
    }
};

void register_flat_builder(pybind11::module& m) {
    using namespace pybind11::literals;

    pybind11::class_<flat_cell_builder> builder(m, "flat_cell_builder");
    builder
        .def(pybind11::init<>())
        .def("add_cable",
                [](flat_cell_builder& b, double len, pybind11::object rad, const char* name, int ncomp) {
                    using pybind11::isinstance;
                    using pybind11::cast;
                    if (auto radius = try_cast<double>(rad) ) {
                        return b.add_cable(len, *radius, *radius, name, ncomp);
                    }

                    if (auto radii = try_cast<std::pair<double, double>>(rad)) {
                        return b.add_cable(len, radii->first, radii->second, name, ncomp);
                    }
                    else {
                        throw pyarb_error(
                            "Radius parameter is not a scalar (constant branch radius) or "
                            "a tuple (radius at proximal and distal ends respectively).");
                    }
                },
            "length"_a, "radius"_a, "name"_a, "ncomp"_a=1)
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
        .def_property_readonly("segments",
            [](const flat_cell_builder& b) { return b.segments(); })
        .def_property_readonly("labels",
            [](const flat_cell_builder& b) { return b.labels(); })
        .def_property_readonly("morphology",
            [](const flat_cell_builder& b) { return b.morphology(); })
        .def("build", &flat_cell_builder::build);
}

} // namespace pyarb
