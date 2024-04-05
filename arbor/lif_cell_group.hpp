#pragma once

#include <vector>
#include <mutex>

#include <arbor/export.hpp>
#include <arbor/common_types.hpp>
#include <arbor/lif_cell.hpp>
#include <arbor/recipe.hpp>
#include <arbor/sampling.hpp>
#include <arbor/spike.hpp>

#include "sampler_map.hpp"
#include "cell_group.hpp"
#include "label_resolution.hpp"

namespace arb {

// Model parameters of leaky integrate and fire neuron model.
struct ARB_SYMBOL_VISIBLE lif_lowered_cell {
    cell_tag_type source; // Label of source.
    cell_tag_type target; // Label of target.

    // Neuronal parameters.
    double tau_m = 10;    // Membrane potential decaying constant [ms].
    double V_th = 10;     // Firing threshold [mV].
    double C_m = 20;      // Membrane capacitance [pF].
    double E_L = 0;       // Resting potential [mV].
    double E_R = E_L;     // Reset potential [mV].
    double V_m = E_L;     // Initial value of the Membrane potential [mV].
    double t_ref = 2;     // Refractory period [ms].

    lif_lowered_cell() = default;
    lif_lowered_cell(const lif_cell& lif) {
        source = lif.source;
        target = lif.target;

        tau_m = lif.tau_m.value_as(U::ms);
        V_th = lif.V_th.value_as(U::mV);
        C_m = lif.C_m.value_as(U::pF);
        E_L = lif.E_L.value_as(U::mV);
        E_R = lif.E_R.value_as(U::mV);
        V_m = lif.V_m.value_as(U::mV);
        t_ref = lif.t_ref.value_as(U::ms);

        if (std::isnan(V_th)) throw std::out_of_range("V_th must be finite and in [mV]");
        if (std::isnan(tau_m) || tau_m < 0) throw std::out_of_range("tau_m must be positive, finite, and in [ms]");
        if (std::isnan(C_m) || C_m < 0) throw std::out_of_range("C_m must be positive, finite, and in [pF]");
        if (std::isnan(E_L)) throw std::out_of_range("E_L must be finite and in [mV]");
        if (std::isnan(E_R)) throw std::out_of_range("E_R must be finite and in [mV]");
        if (std::isnan(V_m)) throw std::out_of_range("V_m must be finite and in [mV]");
        if (std::isnan(t_ref) || t_ref < 0) throw std::out_of_range("t_ref must be positive, finite, and in [ms]");
    }

    ARB_SERDES_ENABLE(lif_lowered_cell, source, target, tau_m, V_th, C_m, E_L, E_R, V_m, t_ref);
};


struct ARB_ARBOR_API lif_cell_group: public cell_group {
    lif_cell_group() = default;

    // Constructor containing gid of first cell in a group and a container of all cells.
    lif_cell_group(const std::vector<cell_gid_type>& gids, const recipe& rec, cell_label_range& cg_sources, cell_label_range& cg_targets);

    cell_kind get_cell_kind() const override;
    void reset() override;
    void advance(epoch epoch, time_type dt, const event_lane_subrange& events) override;

    virtual const std::vector<spike>& spikes() const override;
    void clear_spikes() override;

    // Sampler association methods below should be thread-safe, as they might be invoked
    // from a sampler call back called from a different cell group running on a different thread.
    void add_sampler(sampler_association_handle, cell_member_predicate, schedule, sampler_function) override;
    void remove_sampler(sampler_association_handle) override;
    void remove_all_samplers() override;

    std::vector<probe_metadata> get_probe_metadata(const cell_address_type&) const override;

    ARB_SERDES_ENABLE(lif_cell_group, gids_, cells_, spikes_, last_time_updated_, next_time_updatable_);

    virtual void t_serialize(serializer& ser, const std::string& k) const override;
    virtual void t_deserialize(serializer& ser, const std::string& k) override;

private:
    enum class lif_probe_kind { voltage };

    struct lif_probe_info {
        cell_address_type addr;
        lif_probe_kind kind;
        lif_probe_metadata metadata;
    };

    // Advances a single cell (lid) with the exact solution (jumps can be arbitrary).
    // Parameter dt is ignored, since we make jumps between two consecutive spikes.
    void advance_cell(time_type tfinal, time_type dt, cell_gid_type lid, const event_lane_subrange& event_lane);

    // List of the gids of the cells in the group.
    std::vector<cell_gid_type> gids_;

    // Cells that belong to this group.
    std::vector<lif_lowered_cell> cells_;

    // Spikes that are generated (not necessarily sorted).
    std::vector<spike> spikes_;

    // Time when the cell was last updated.
    std::vector<time_type> last_time_updated_;
    // Time when the cell was last sampled.
    std::vector<time_type> last_time_sampled_;
    // Time when the cell can _next_ be updated;
    std::vector<time_type> next_time_updatable_;

    // SAFETY: We need to access samplers_ through a mutex since
    //         simulation::add_sampler might be called concurrently.
    std::mutex sampler_mex_;
    sampler_association_map samplers_;

    // LIF probe metadata, precalculated to pass to callbacks
    std::unordered_map<cell_address_type, lif_probe_info> probes_;
};

} // namespace arb
