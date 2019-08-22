#pragma once

// Simple recipe classes for use in unit and validation tests.

#include <unordered_map>
#include <vector>

#include <arbor/event_generator.hpp>
#include <arbor/cable_cell.hpp>
#include <arbor/cable_cell_param.hpp>
#include <arbor/recipe.hpp>

namespace arb {

// Common functionality: maintain an unordered map of probe data
// per gid, built with `add_probe()`.

class simple_recipe_base: public recipe {
public:
    simple_recipe_base():
        catalogue_(global_default_catalogue())
    {
        cell_gprop_.catalogue = &catalogue_;
        cell_gprop_.default_parameters = neuron_parameter_defaults;
    }

    cell_size_type num_probes(cell_gid_type i) const override {
        return probes_.count(i)? probes_.at(i).size(): 0;
    }

    virtual probe_info get_probe(cell_member_type probe_id) const override {
        return probes_.at(probe_id.gid).at(probe_id.index);
    }

    virtual void add_probe(cell_gid_type gid, probe_tag tag, util::any address) {
        auto& pvec_ = probes_[gid];

        cell_member_type probe_id{gid, cell_lid_type(pvec_.size())};
        pvec_.push_back({probe_id, tag, std::move(address)});
    }

    util::any get_global_properties(cell_kind k) const override {
        switch (k) {
        case cell_kind::cable:
            return cell_gprop_;
        default:
            return util::any{};
        }
    }

    mechanism_catalogue& catalogue() {
        return catalogue_;
    }

    void add_ion(const std::string& ion_name, int charge, double init_iconc, double init_econc, double init_revpot) {
        cell_gprop_.add_ion(ion_name, charge, init_iconc, init_econc, init_revpot);
    }

    void nernst_ion(const std::string& ion_name) {
        cell_gprop_.default_parameters.reversal_potential_method[ion_name] = "nernst/"+ion_name;
    }

protected:
    std::unordered_map<cell_gid_type, std::vector<probe_info>> probes_;
    cable_cell_global_properties cell_gprop_;
    mechanism_catalogue catalogue_;
};

// Convenience derived recipe class for wrapping n copies of a single
// cell description, with no sources or targets. (Derive the class to
// add sources, targets, connections etc.)
//
// Probes are simply stored in a multimap, keyed by gid and can be manually
// added with 'add_probe()'.
//
// Wrapped description class must be both move- and copy-constructable.

template <cell_kind Kind, typename Description>
class homogeneous_recipe: public simple_recipe_base {
public:
    homogeneous_recipe(cell_size_type n, Description desc):
        n_(n), desc_(std::move(desc))
    {}

    cell_size_type num_cells() const override { return n_; }
    cell_kind get_cell_kind(cell_gid_type) const override { return Kind; }

    util::unique_any get_cell_description(cell_gid_type) const override {
        return util::make_unique_any<Description>(desc_);
    }

protected:
    cell_size_type n_;
    Description desc_;
};

// Recipe for a set of `cable` neurons without connections,
// and probes which can be added by `add_probe()` (similar to above).
//
// Cell descriptions passed to the constructor are cloned.

class cable1d_recipe: public simple_recipe_base {
public:
    template <typename Seq>
    explicit cable1d_recipe(const Seq& cells, bool coalesce = true) {
        for (const auto& c: cells) {
            cells_.emplace_back(c);
        }
        cell_gprop_.coalesce_synapses = coalesce;
    }

    explicit cable1d_recipe(const cable_cell& c, bool coalesce = true) {
        cells_.reserve(1);
        cells_.emplace_back(c);
        cell_gprop_.coalesce_synapses = coalesce;
    }

    cell_size_type num_cells() const override { return cells_.size(); }
    cell_kind get_cell_kind(cell_gid_type) const override { return cell_kind::cable; }

    cell_size_type num_sources(cell_gid_type i) const override {
        return cells_.at(i).detectors().size();
    }

    cell_size_type num_targets(cell_gid_type i) const override {
        return cells_.at(i).synapses().size();
    }

    util::unique_any get_cell_description(cell_gid_type i) const override {
        return util::make_unique_any<cable_cell>(cells_[i]);
    }

protected:
    std::vector<cable_cell> cells_;
};

} // namespace arb

