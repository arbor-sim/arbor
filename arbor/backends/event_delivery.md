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
cell. The cell information is given as an index into the cell group collection of cells.

Target handles are represented by the `target_handle` struct, opaque to `mc_cell_group`,
but created in the `fvm_multicell` for each point mechanism (synapse) in the cell group.

### Deliverable events

Events for delivery within the next integration period are staged in the lowered cell
as a vector of `deliverable_event` objects. These comprise an event delivery time,
a `target_handle` describing their destination, and a weight.

### Back-end event streams

`backend::multi_event_stream` represents a set of event streams. There is one `multi_event_stream`
per mechanism storage `backend::shared_state::mech_storage`.
From the perspective of the lowered cell, it must support the methods below.

*  `void init(const std::vector<deliverable_event>& staged_events)`

   Take a copy of the staged events (which must be partitioned by the event index (a.k.a. the stream
   index), and ordered by increasing event time within each partition) and initialize the streams.

*  `bool empty() const`

   Return true if and only if there are no un-retired events left in any stream.

*  `size_type n_streams() const`

   Number of partitions/streams.

*  `size_type n_remaining() const`

   Number of remaining un-retired events among all streams.

*  `size_type n_marked() const`

   Number of marked events among all streams.

*  `void clear()`

   Retire all events, leaving the `multi_event_stream` in an empty state.

*  `void mark_until_after(arb_value_type t_until)`

   For all streams, mark events for delivery with event time ≤ `t_until`.

*  `void drop_marked_events()`

   Retire all marked events.


## Event delivery and integration timeline

Event delivery is performed as part of the integration loop within the lowered
cell implementation. The interface is provided by the `multi_event_stream`
described above, together with the mechanism method that handles the delivery proper,
`mechanism::deliver_events` and `backend` methods
`shared_state::register_events`, `shared_state::mark_events` and `shared_state::deliver_events`.
Events are considered as pending when their event time is within one time step of the current time:
`event_time > time - dt/2 && event_time <= time + dt/2`.

For `fvm_multicell` one integration step comprises:

1.  Events for each cell that are due at that cell's corresponding time are
    gathered with `state_>mark_events(step_midpoint)` where `step_midpoint` is
    the current time plus half a time step (upper bound for pending event times). This method, in
    turn, calls the `multi_event_stream::mark_until_after(step_midpoint)`.

2.  Each mechanism is requested to deliver to itself any marked events that
    are associated with that mechanism, via the
    `shared_state::deliver_events(const mechanism&)` method. This method eventually calls the virtual
    `mechanism::apply_events(backend::multi_event_stream&)` method and retires the deliverd events
    using `multi_event_stream::drop_marked_events`.

    This action must precede the computation of mechanism current contributions
    with `mechanism::compute_currents()`.

3.  The solver matrix is assembled and solved to compute the voltages, using the
    newly computed currents and integration step times.

4.  The mechanism states are updated with `mechanism::advance_state()`.

5.  The cell times `time` are set to the integration step stop times `time_to`.

6.  Spike detection for the last integration step is performed via the
    `threshold_watcher_` object.
