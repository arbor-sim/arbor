`multi_event_stream` implements a pop only event queue.
* there are multicore and gpu implementations
* `event_time_if_before` method will modify the `time_to` values used to set time step size for each cell in a
  cable cell group to the time of the next event in their respective event queues.  
    * remove this, because we don't want to modify this value (if at all possible we want to remove it entirely

`mechanism_abi.hpp` contains the `arb_mechanism_ppack` struct: state info provided (read only) to mech implementations.
* remove the folowing arrays and replace with scalars:
    * `const arb_value_type* vec_t;`
    * `arb_value_type* vec_dt;`
* consider that, instead of passing these through pp_pack, we could pass them as arguments to `advance` functions.
* from inspecting the code, the pointers and values in the parameter packs are set on construction, and are not modified during execution
    * of course, the contents of the arrays change!

Where to keep the time and timestep "state"?
    * keep it in the parameter pack.
    * where it might be set/updated according to a value kept in the cell group.

Will it every vary?
    * yes: sub-dt steps will be required at a) the end of epochs b) the end of a `simulation::run` period.
