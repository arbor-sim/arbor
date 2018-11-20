# To Do List For Python Wrappers

- [] documentation
- [] probes and sampling
- [] cell building
    - [] load from swc
    - [] construction by section assembly
    - [] use neuroml terminology
- [] unit tests for all implemented features
- [] demos
    - [] demo with plot of voltage trace
    - [] demo with plot of spikes
    - [] demo with MPI
    - [] demo with GPU
    - [] demo with poisson event generator
    - [] ring demo
    - [] k-way connectivity demo with inhib and excit populations
- [x] MPI support via for mpi4py found correctly by CMake


// TODO:
/*

- extend the python recipe shims to implement
    virtual cell_size_type recipe::num_probes(cell_gid_type)  const { return 0; }
    virtual probe_info recipe::get_probe_cell(cell_member_type)  const { throw... }
- wrap arb::cell_probe_address
    - location + what to sample
    - this is what users can return from the 
- wrap arb::probe_info
    - make a multicompartment probe info

*/

// common_types.hpp

using probe_tag = int;                 // extra contextual information associated with a probe.
using sample_size_type = std::int32_t; // for holding counts and indexes into generated sample data.

// recipe.hpp

struct probe_info {
    cell_member_type id;
    probe_tag tag;
    util::any address; // specific to cell kind of cell `id.gid`.
};

probe_info recipe::get_probe(cell_member_type probe_id);


// mc_cell.hpp

struct cell_probe_address {
    enum probe_kind {
        membrane_voltage, membrane_current
    };
    segment_location location;
    probe_kind kind;
};

// Example recipe implementation
probe_info recipe::get_probe(cell_member_type probe_id) {
    // Get the appropriate kind for measuring voltage.
    cell_probe_address::probe_kind kind = cell_probe_address::membrane_voltage;
    // Measure at the soma.
    arb::segment_location loc(0, 0.0);

    return arb::probe_info{id, kind, cell_probe_address{loc, kind}};
}

/*
   The recipe-shim should return an arbitrary python object that will have cell-kind
   specific description of sampling.
       - this way only cell-kind specific descriptions need be wrapped
       - 

   The shim unwraps and forwards this: care needs to be taken that 
    the simplest implementation requires
        id : given as argument
        kind : provided by user
        location : provided by user

    
    probe_info should be wrapped, then constructed using these three arguments


*/


