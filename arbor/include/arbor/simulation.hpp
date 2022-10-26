#pragma once

#include <array>
#include <memory>
#include <unordered_map>
#include <vector>
#include <functional>

#include <arbor/export.hpp>
#include <arbor/arb_types.hpp>
#include <arbor/common_types.hpp>
#include <arbor/context.hpp>
#include <arbor/domain_decomposition.hpp>
#include <arbor/load_balance.hpp>
#include <arbor/recipe.hpp>
#include <arbor/sampling.hpp>
#include <arbor/schedule.hpp>
#include <arbor/spike.hpp>
#include <arbor/util/handle_set.hpp>

namespace arb {

using spike_export_function = std::function<void(const std::vector<spike>&)>;
using epoch_function = std::function<void(double time, double tfinal)>;

// simulation_state comprises private implementation for simulation class.
class simulation_state;

class simulation_builder;

class ARB_ARBOR_API simulation {
public:
    simulation(const recipe& rec, context ctx, const domain_decomposition& decomp,
               arb_seed_type seed = 0);

    simulation(const recipe& rec,
               context ctx = make_context(),
               std::function<domain_decomposition(const recipe&, context)> balancer = 
                   [](auto& r, auto c) { return partition_load_balance(r, c); },
               arb_seed_type seed = 0):
        simulation(rec, ctx, balancer(rec, ctx)) {}

    simulation(simulation const&) = delete;
    simulation(simulation&&);

    static simulation_builder create(recipe const &);

    void update(const connectivity& rec);

    void reset();

    time_type run(time_type tfinal, time_type dt);

    // Note: sampler functions may be invoked from a different thread than that
    // which called the `run` method.

    sampler_association_handle add_sampler(cell_member_predicate probeset_ids,
        schedule sched, sampler_function f, sampling_policy policy = sampling_policy::lax);

    void remove_sampler(sampler_association_handle);

    void remove_all_samplers();

    // Return probe metadata, one entry per probe associated with supplied probe id,
    // or an empty vector if no local match for probe id.
    std::vector<probe_metadata> get_probe_metadata(cell_member_type probeset_id) const;

    std::size_t num_spikes() const;

    // Set event binning policy on all our groups.
    void set_binning_policy(binning_kind policy, time_type bin_interval);

    // Register a callback that will perform a export of the global
    // spike vector.
    void set_global_spike_callback(spike_export_function = spike_export_function{});

    // Register a callback that will perform a export of the rank local
    // spike vector.
    void set_local_spike_callback(spike_export_function = spike_export_function{});

    // Register a callback that will be called at the end of each epoch, and at the
    // start of the simulation.
    void set_epoch_callback(epoch_function = epoch_function{});

    // Add events directly to targets.
    // Must be called before calling simulation::run, and must contain events that
    // are to be delivered at or after the current simulation time.
    void inject_events(const cse_vector& events);

    ~simulation();

private:
    std::unique_ptr<simulation_state> impl_;
};

// Builder pattern for simulation class to help with construction.
// Simulation constructor arguments can be added through setter functions in any order or left out
// entirely, in which case a sane default is chosen. The build member function instantiates the
// simulation with the current arguments and returns it.
class ARB_ARBOR_API simulation_builder {
public:

    simulation_builder(recipe const& rec) noexcept : rec_{rec} {}

    simulation_builder(simulation_builder&&) = default;
    simulation_builder(simulation_builder const&) = default;

    simulation_builder& set_context(context ctx) noexcept {
        ctx_ = ctx;
        return *this;
    }

    simulation_builder& set_decomposition(domain_decomposition decomp) noexcept {
        balancer_ = [decomp = std::move(decomp)](const recipe&, context) {return decomp; };
        return *this;
    }

    simulation_builder& set_balancer(
        std::function<domain_decomposition(const recipe&, context)> balancer) noexcept {
        balancer_ = std::move(balancer);
        return *this;
    }

    simulation_builder& set_seed(arb_seed_type seed) noexcept {
        seed_ = seed;
        return *this;
    }

    operator simulation() const { return build(); }

    std::unique_ptr<simulation> make_unique() const {
        return std::make_unique<simulation>(build());
    }

private:
    simulation build() const {
        return ctx_ ?
            build(ctx_):
            build(make_context());
    }

    simulation build(context ctx) const {
        return balancer_ ?
            build(ctx, balancer_(rec_, ctx)):
            build(ctx, partition_load_balance(rec_, ctx));
    }

    simulation build(context ctx, domain_decomposition const& decomp) const {
        return simulation(rec_, ctx, decomp, seed_);
    }

private:
    const recipe& rec_;
    context ctx_;
    std::function<domain_decomposition(const recipe&, context)> balancer_;
    arb_seed_type seed_ = 0u;
};

// An epoch callback function that prints out a text progress bar.
ARB_ARBOR_API epoch_function epoch_progress_bar();

} // namespace arb
