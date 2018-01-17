#pragma once

// Simple recipe classes for use in unit and validation tests.

#include <unordered_map>
#include <vector>

#include <cell.hpp>
#include <event_generator.hpp>
#include <recipe.hpp>

namespace arb {

// Common functionality: maintain an unordered map of probe data
// per gid, built with `add_probe()`.

class simple_recipe_base: public recipe {
public:
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

    void add_specialized_mechanism(std::string name, specialized_mechanism m) {
        cell_gprop.special_mechs[name] = std::move(m);
    }

    util::any get_global_properties(cell_kind k) const override {
        switch (k) {
        case cable1d_neuron:
            return cell_gprop;
        default:
            return util::any{};
        }
    }

protected:
    std::unordered_map<cell_gid_type, std::vector<probe_info>> probes_;
    cell_global_properties cell_gprop;
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

// Recipe for a set of `cable1d_neuron` neurons without connections,
// and probes which can be added by `add_probe()` (similar to above).
//
// Cell descriptions passed to the constructor are cloned.

class cable1d_recipe: public simple_recipe_base {
public:
    template <typename Seq>
    explicit cable1d_recipe(const Seq& cells) {
        for (const auto& c: cells) {
            cells_.emplace_back(c);
        }
    }

    explicit cable1d_recipe(const cell& c) {
        cells_.reserve(1);
        cells_.emplace_back(c);
    }

    cell_size_type num_cells() const override { return cells_.size(); }
    cell_kind get_cell_kind(cell_gid_type) const override { return cell_kind::cable1d_neuron; }

    cell_size_type num_sources(cell_gid_type i) const override {
        return cells_.at(i).detectors().size();
    }

    cell_size_type num_targets(cell_gid_type i) const override {
        return cells_.at(i).synapses().size();
    }

    util::unique_any get_cell_description(cell_gid_type i) const override {
        return util::make_unique_any<cell>(cells_[i]);
    }

protected:
    std::vector<cell> cells_;
};


} // namespace arb

