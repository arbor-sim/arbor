#pragma once

#include <arbor/assert.hpp>
#include <arbor/common_types.hpp>
#include <arbor/spike_event.hpp>
#include <arbor/schedule.hpp>
#include <arbor/arbexcept.hpp>

namespace arb {

// An `event_generator` generates a sequence of events to be delivered to a
// cell. The sequence of events is always in ascending order, i.e. each event
// will be greater than the event that proceeded it, where events are ordered
// by:
//  - delivery time;
//  - then target id for events with the same delivery time;
//  - then weight for events with the same delivery time and target.
//
// An `event_generator` supports three operations:
//
// `void event_generator::reset()`
//
//     Reset generator state.
//
// `event_seq event_generator::events(time_type to, time_type from)`
//
//     Provide a non-owning view on to the events in the time interval [to,
//     from).
//
// `void set_target_lid(lid)`
//
//     event_generators are constructed on cable_local_label_types comprising a
//     label and a selection policy. These labels need to be resolved to a
//     specific cell_lid_type. This is done externally.
//
// The `event_seq` type is a pair of `spike_event` pointers that provide a view
// onto an internally-maintained contiguous sequence of generated spike event
// objects. This view is valid only for the lifetime of the generator, and is
// invalidated upon a call to `reset` or another call to `events`.
//
// Calls to the `events` method must be monotonic in time: without an
// intervening call to `reset`, two successive calls `events(t0, t1)` and
// `events(t2, t3)` to the same event generator must satisfy
//
//         0 ≤ t0 ≤ t1 ≤ t2 ≤ t3.
//
// `event_generator` objects have value semantics.

using event_seq = std::pair<const spike_event*, const spike_event*>;

// Generate events with a fixed target and weight according to
// a provided time schedule.
struct event_generator {
    event_generator(cell_local_label_type target, float weight, schedule sched):
        weight_(weight), target_(std::move(target)), sched_(std::move(sched))
    {}

    void set_target_lid(cell_lid_type lid) { target_lid_ = lid; }
    const cell_local_label_type& target() const { return target_; }

    void reset() { sched_.reset(); }

    event_seq events(time_type t0, time_type t1) {
        arb_assert(target_lid_ != cell_lid_type(-1));
        const auto& [t_lo, t_hi] = sched_.events(t0, t1);

        events_.clear();
        events_.reserve(t_hi - t_lo);
        for (auto it = t_lo; it != t_hi; ++it) events_.emplace_back(target_lid_, *it, weight_);

        return {events_.data(), events_.data()+events_.size()};
    }

private:
    cell_lid_type target_lid_ = cell_lid_type(-1);
    float weight_ = 0;
    cell_local_label_type target_;
    pse_vector events_;
    schedule sched_;
};

// Simplest generator: just do nothing
inline event_generator empty_generator(cell_local_label_type target, float weight) {
    return event_generator(std::move(target), weight, schedule());
}


// Generate events at integer multiples of dt that lie between tstart and tstop.
inline event_generator regular_generator(cell_local_label_type target,
                                         float weight,
                                         const units::quantity& tstart,
                                         const units::quantity& dt,
                                         const units::quantity& tstop=terminal_time*units::ms) {
    return event_generator(std::move(target), weight, regular_schedule(tstart, dt, tstop));
}

inline event_generator poisson_generator(cell_local_label_type target,
                                         float weight,
                                         const units::quantity& tstart,
                                         const units::quantity& rate_kHz,
                                         seed_type seed = default_seed,
                                         const units::quantity& tstop=terminal_time*units::ms) {
    return event_generator(std::move(target), weight, poisson_schedule(tstart, rate_kHz, seed, tstop));
}


// Generate events from a predefined sorted event sequence.
template<typename S> inline
event_generator explicit_generator(cell_local_label_type target,
                                   float weight,
                                   const S& s) {
    return event_generator(std::move(target), weight, explicit_schedule(s));
}

template<typename S> inline
event_generator explicit_generator_from_milliseconds(cell_local_label_type target,
                                                     float weight,
                                                     const S& s) {
    return event_generator(std::move(target), weight, explicit_schedule_from_milliseconds(s));
}

} // namespace arb

