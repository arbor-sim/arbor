#pragma once

#include <algorithm>
#include <cstdint>
#include <memory>
#include <random>
#include <type_traits>
#include <optional>

#include <arbor/assert.hpp>
#include <arbor/common_types.hpp>
#include <arbor/generic_event.hpp>
#include <arbor/spike_event.hpp>
#include <arbor/schedule.hpp>
#include <arbor/arbexcept.hpp>

namespace arb {

// An `event_generator` generates a sequence of events to be delivered to a cell.
// The sequence of events is always in ascending order, i.e. each event will be
// greater than the event that proceeded it, where events are ordered by:
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
//     Provide a non-owning view on to the events in the time interval
//     [to, from).
//
// `void resolve_label(resolution_function)`
//
//     event_generators are constructed on cable_local_label_types comprising
//     a label and a selection policy. These labels need to be resolved to a
//     specific cell_lid_type. This is done using a resolution_function.
//
// The `event_seq` type is a pair of `spike_event` pointers that
// provide a view onto an internally-maintained contiguous sequence
// of generated spike event objects. This view is valid only for
// the lifetime of the generator, and is invalidated upon a call
// to `reset` or another call to `events`.
//
// Calls to the `events` method must be monotonic in time: without an
// intervening call to `reset`, two successive calls `events(t0, t1)`
// and `events(t2, t3)` to the same event generator must satisfy
// 0 ≤ t0 ≤ t1 ≤ t2 ≤ t3.
//
// `event_generator` objects have value semantics.

using event_seq = std::pair<const spike_event*, const spike_event*>;
using resolution_function = std::function<cell_lid_type(const cell_local_label_type&)>;


// Generate events with a fixed target and weight according to
// a provided time schedule.

struct event_generator {
    event_generator(cell_local_label_type target, float weight, schedule sched):
        target_(std::move(target)), weight_(weight), sched_(std::move(sched))
    {}

    void resolve_label(resolution_function label_resolver) {
        resolved_ = label_resolver(target_);
    }

    void reset() {
        sched_.reset();
    }

    event_seq events(time_type t0, time_type t1) {
        if (!resolved_) throw arbor_internal_error("Unresolved label in event generator.");
        auto tgt = *resolved_;
        auto ts = sched_.events(t0, t1);

        events_.clear();
        events_.reserve(ts.second-ts.first);

        for (auto i = ts.first; i!=ts.second; ++i) {
            events_.push_back(spike_event{tgt, *i, weight_});
        }

        return {events_.data(), events_.data()+events_.size()};
    }

private:
    pse_vector events_;
    cell_local_label_type target_;
    resolution_function label_resolver_;
    std::optional<cell_lid_type> resolved_;
    float weight_;
    schedule sched_;
};

// Simplest generator: just do nothing
inline
event_generator empty_generator(
    cell_local_label_type target,
    float weight)
{
    return event_generator(std::move(target), weight, schedule());
}


// Generate events at integer multiples of dt that lie between tstart and tstop.

inline event_generator regular_generator(
    cell_local_label_type target,
    float weight,
    time_type tstart,
    time_type dt,
    time_type tstop=terminal_time)
{
    return event_generator(std::move(target), weight, regular_schedule(tstart, dt, tstop));
}

template <typename RNG>
inline event_generator poisson_generator(
    cell_local_label_type target,
    float weight,
    time_type tstart,
    time_type rate_kHz,
    const RNG& rng,
    time_type tstop=terminal_time)
{
    return event_generator(std::move(target), weight, poisson_schedule(tstart, rate_kHz, rng, tstop));
}


// Generate events from a predefined sorted event sequence.

template<typename S> inline
event_generator explicit_generator(cell_local_label_type target,
                                   float weight,
                                   const S& s)
{
    return event_generator(std::move(target), weight, explicit_schedule(s));
}

} // namespace arb

