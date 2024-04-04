#pragma once

#include <vector>
#include <mutex>

#include <arbor/export.hpp>
#include <arbor/common_types.hpp>
#include <arbor/adex_cell.hpp>
#include <arbor/recipe.hpp>
#include <arbor/sampling.hpp>
#include <arbor/spike.hpp>

#include "sampler_map.hpp"
#include "cell_group.hpp"
#include "label_resolution.hpp"

namespace arb {

#define UNIT_OF(x, u) \
    x = adex.x.value_as(arb::units::u); \
    if (!std::isfinite(x)) throw std::domain_error(#x " must be finite and in [" #u "]"); \

struct ARB_SYMBOL_VISIBLE adex_lowered_cell {
    cell_tag_type source; // Label of source
    cell_tag_type target; // Label of target

    // Neuronal parameters.
    double delta; // Steepness parameter [mV]
    double V_th;  // Firing threshold [mV]
    double C_m;   // Membrane capacitance [pF]
    double E_L;   // Resting potential [mV]
    double E_R;   // Reset potential [mV]
    double V_m;   // Initial value of the Membrane potential [mV]
    double t_ref; // Refractory period [ms]
    double g;     // Leak conductivity [uS]
    // Adaption parameters
    double tau;   // Adaption decaying constant [ms]
    double w;     // Initial value for adaption parameter [nA]
    double a;     // Adaption dynamics [uS].
    double b;     // When spikes trigger, increase w by this [nA]

    adex_lowered_cell() { *this = adex_cell{}; };
    adex_lowered_cell(const adex_cell& adex) {
        source = adex.source;
        target = adex.target;

        UNIT_OF(delta, mV);
        UNIT_OF(V_th, mV);
        UNIT_OF(C_m, pF);
        UNIT_OF(E_L, mV);
        UNIT_OF(E_R, mV);
        UNIT_OF(V_m, mV);
        UNIT_OF(t_ref, ms);
        UNIT_OF(g, uS);
        UNIT_OF(tau, ms);
        UNIT_OF(w, nA);
        UNIT_OF(a, uS);
        UNIT_OF(b, nA);

        if (t_ref < 0) throw std::domain_error("t_ref must be positive.");
        if (t_ref < 0) throw std::domain_error("tau must be positive.");
        if (C_m < 0) throw std::domain_error("C_m must be positive.");
    }

    ARB_SERDES_ENABLE(adex_lowered_cell, delta, V_th, V_m, C_m, E_L, E_R, t_ref, g, tau, w, a, b);
};

#undef UNIT_OF

struct ARB_ARBOR_API adex_cell_group: public cell_group {
    adex_cell_group() = default;

    // Constructor containing gid of first cell in a group and a container of all cells.
    adex_cell_group(const std::vector<cell_gid_type>& gids, const recipe& rec, cell_label_range& cg_sources, cell_label_range& cg_targets);

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

    ARB_SERDES_ENABLE(adex_cell_group, current_time_, gids_, cells_, spikes_);

    virtual void t_serialize(serializer& ser, const std::string& k) const override;
    virtual void t_deserialize(serializer& ser, const std::string& k) override;

    static bool backend_supported(backend_kind kind) { return kind == backend_kind::multicore; }

private:
    enum class adex_probe_kind { voltage, adaption };

    struct adex_probe_info {
        adex_probe_kind kind;
        adex_probe_metadata metadata;
    };

    // Advances a single cell (lid) with the exact solution (jumps can be arbitrary).
    // Parameter dt is ignored, since we make jumps between two consecutive spikes.
    void advance_cell(time_type tfinal, time_type dt, cell_gid_type lid, const event_lane_subrange& event_lane);



    // List of the gids of the cells in the group.
    std::vector<cell_gid_type> gids_;

    // Cells that belong to this group.
    std::vector<adex_lowered_cell> cells_;

    // Spikes that are generated (not necessarily sorted).
    std::vector<spike> spikes_;

    // SAFETY: We need to access samplers_ through a mutex since
    //         simulation::add_sampler might be called concurrently.
    std::mutex sampler_mex_;
    sampler_association_map samplers_; //

    // ADEX probe metadata, precalculated to pass to callbacks
    std::unordered_map<cell_address_type, adex_probe_info> probes_;

    // Time of next possible update to model refractory periods.
    std::vector<time_type> next_update_;
    std::vector<time_type> current_time_;
};

} // namespace arb
