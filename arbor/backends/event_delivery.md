# Back-end event delivery

## Data structures

The lowered cell implementation gathers postsynaptic spike events from its governing
`cell_group` class, and then passes them on to concrete back-end device-specific
classes for on-device delivery.

`backends/event.hpp` contains the concrete classes used to represent event
destinations and event information.

The back-end event management structure is supplied by the corresponding `backend`
class as `backend::multi_event_stream`. It presents a limited public interface to
the lowered cell, and is passed by reference as a parameter to the mechanism
`apply_events` method.

### Target handles

Target handles are used by the lowered cell implementation to identify a particular mechanism
instance that can receive events — ultimately via `net_receive` — and corresponding simulated
cell. The cell information is given as an index into the cell group collection of cells,
and is used to group events by integration domain (we have one domain per cell in each cell
group).

Target handles are represented by the `target_handle` struct, opaque to `mc_cell_group`,
but created in the `fvm_multicell` for each point mechanism (synapse) in the cell group.

### Deliverable events

Events for delivery within the next integration period are staged in the lowered cell
as a vector of `deliverable_event` objects. These comprise an event delivery time,
a `target_handle` describing their destination, and a weight.

### Back-end event streams

`backend::multi_event_stream` represents a set (one per cell/integration domain)
of event streams. There is one `multi_event_stream` per lowered cell.

From the perspective of the lowered cell, it must support the methods below.
In the following, `time` is a `view` or `const_view` of an array with one
element per stream.

*  `void init(const std::vector<deliverable_event>& staged_events)`

   Take a copy of the staged events (which must be ordered by increasing event time)
   and initialize the streams by gathering the events by cell.

*  `bool empty() const`

   Return true if and only if there are no un-retired events left in any stream.

*  `void clear()`

   Retire all events, leaving the `multi_event_stream` in an empty state.

*  `void mark_until_after(const_view time)`

   For all streams, mark events for delivery in the _i_ th stream with event time ≤ _time[i]_.

*  `void event_times_if_before(view time) const`

   For each stream, set _time[i]_ to the time of the next event time in the _i_ th stream
   if such an event exists and has time less than _time[i]_.

*  `void drop_marked_events()`

   Retire all marked events.


## Event delivery and integration timeline

Event delivery is performed as part of the integration loop within the lowered
cell implementation. The interface is provided by the `multi_event_stream`
described above, together with the mechanism method that handles the delivery proper,
`mechanism::deliver_events` and a `backend` static method that computes the
integration step end time before considering any pending events.

For `fvm_multicell` one integration step comprises:

1.  Events for each cell that are due at that cell's corresponding time are
    gathered with `events_.mark_events(time_)` where `time_` is a
    `const_view` of the cell times and `events_` is a reference to the
    `backend::multi_event_stream` object.

2.  Each mechanism is requested to deliver to itself any marked events that
    are associated with that mechanism, via the virtual
    `mechanism::apply_events(backend::multi_event_stream&)` method.

    This action must precede the computation of mechanism current contributions
    with `mechanism::compute_currents()`.

3.  Marked events are discarded with `events_.drop_marked_events()`.

4.  The upper bound on the integration step stop time `time_to_` is
    computed via `backend::update_time_to(time_to_, time_, dt_max_, tfinal_)`,
    as the minimum of the per-cell time `time_` plus `dt_max_` and
    the final integration stop time `tfinal_`.

5.  The integration step stop time `time_to_` is reduced to match any
    pending events on each cell with `events_.event_times_if_before(time_to)`.

6.  The solver matrix is assembled and solved to compute the voltages, using the
    newly computed currents and integration step times.

7.  The mechanism states are updated with `mechanism::advance_state()`.

8.  The cell times `time_` are set to the integration step stop times `time_to_`.

9.  Spike detection for the last integration step is performed via the
    `threshold_watcher_` object.

## Consequences for the integrator

Towards the end of the integration period, an integration step may have a zero _dt_
for one or more cells within the group, and this needs to be handled correctly:

*   Generated mechanism `advance_state()` methods should be numerically correct with
    zero _dt_; a possibility is to guard the integration step with a _dt_ check.

*   Matrix assemble and solve must check for zero _dt_. In the FVM `multicore`
    matrix implementation, zero _dt_ sets the diagonal element to zero and the
    rhs to the voltage; the solve procedure then ignores cells with a zero
    diagonal element.
