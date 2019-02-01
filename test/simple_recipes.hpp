#pragma once

// Simple recipe classes for use in unit and validation tests.

#include <unordered_map>
#include <vector>

#include <arbor/event_generator.hpp>
#include <arbor/mc_cell.hpp>
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
        case cell_kind::cable1d_neuron:
            return cell_gprop_;
        default:
            return util::any{};
        }
    }

    mechanism_catalogue& catalogue() {
        return catalogue_;
    }

protected:
    std::unordered_map<cell_gid_type, std::vector<probe_info>> probes_;
    mc_cell_global_properties cell_gprop_;
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

    explicit cable1d_recipe(const mc_cell& c) {
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
        return util::make_unique_any<mc_cell>(cells_[i]);
    }

protected:
    std::vector<mc_cell> cells_;
};

class gap_recipe_0: public recipe {
public:
    gap_recipe_0() {}

    cell_size_type num_cells() const override {
        return size_;
    }

    arb::util::unique_any get_cell_description(cell_gid_type) const override {
        mc_cell c;
        c.add_soma(20);
        c.add_gap_junction({0,1});
        c.add_gap_junction({0,1});
        return {std::move(c)};
    }

    cell_kind get_cell_kind(cell_gid_type gid) const override {
        return cell_kind::cable1d_neuron;
    }
    std::vector<gap_junction_connection> gap_junctions_on(cell_gid_type gid) const override {
        switch (gid) {
            case 0 :
                return {gap_junction_connection({5, 0}, 0.1)};
            case 2 :
                return {
                        gap_junction_connection({3, 0}, 0.1),
                        gap_junction_connection({7, 0}, 0.1)
                };
            case 3 :
                return {
                        gap_junction_connection({7, 0}, 0.1),
                        gap_junction_connection({2, 0}, 0.1)
                };
            case 5 :
                return {gap_junction_connection({0, 0}, 0.1)};
            case 7 :
                return {
                        gap_junction_connection({3, 0}, 0.1),
                };
            default :
                return {};
        }
    }

private:
    cell_size_type size_ = 12;
};

class gap_recipe_1: public recipe {
public:
    gap_recipe_1() {}

    cell_size_type num_cells() const override {
        return size_;
    }

    arb::util::unique_any get_cell_description(cell_gid_type) const override {
        mc_cell c;
        c.add_soma(20);
        return {std::move(c)};
    }

    cell_kind get_cell_kind(cell_gid_type gid) const override {
        return cell_kind::cable1d_neuron;
    }

private:
    cell_size_type size_ = 12;
};

class gap_recipe_2: public recipe {
public:
    gap_recipe_2() {}

    cell_size_type num_cells() const override {
        return size_;
    }

    arb::util::unique_any get_cell_description(cell_gid_type) const override {
        mc_cell c;
        c.add_soma(20);
        c.add_gap_junction({0,1});
        c.add_gap_junction({0,1});
        c.add_gap_junction({0,1});
        return {std::move(c)};
    }

    cell_kind get_cell_kind(cell_gid_type gid) const override {
        return cell_kind::cable1d_neuron;
    }
    std::vector<gap_junction_connection> gap_junctions_on(cell_gid_type gid) const override {
        switch (gid) {
            case 0 :
                return {
                    gap_junction_connection({2, 0}, 0.1),
                    gap_junction_connection({3, 0}, 0.1),
                    gap_junction_connection({5, 0}, 0.1)
                };
            case 2 :
                return {
                    gap_junction_connection({0, 0}, 0.1),
                    gap_junction_connection({3, 0}, 0.1),
                    gap_junction_connection({5, 0}, 0.1)
                };
            case 3 :
                return {
                    gap_junction_connection({0, 0}, 0.1),
                    gap_junction_connection({2, 0}, 0.1),
                    gap_junction_connection({5, 0}, 0.1)
                };
            case 5 :
                return {
                    gap_junction_connection({2, 0}, 0.1),
                    gap_junction_connection({3, 0}, 0.1),
                    gap_junction_connection({0, 0}, 0.1)
                };
            default :
                return {};
        }
    }

private:
    cell_size_type size_ = 12;
};


} // namespace arb

