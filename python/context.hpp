#pragma once

#include <pybind11/pybind11.h>
#include "conversion.hpp"
#include <arbor/context.hpp>
#include <arborenv/default_env.hpp>

namespace pyarb {

// A Python shim that holds the information that describes an arb::proc_allocation.
struct proc_allocation_shim {
    arb::proc_allocation proc_allocation;
    proc_allocation_shim(arb::proc_allocation pa) : proc_allocation{pa} {}
    proc_allocation_shim(unsigned = 1, pybind11::object = pybind11::none(), bool = false, bool = false);

    // getter and setter (in order to assert when being set)
    void set_gpu_id(pybind11::object);
    void set_num_threads(unsigned);
    void set_bind_procs(bool bp) { proc_allocation.bind_threads = bp; };
    void set_bind_threads(bool bt) { proc_allocation.bind_threads = bt; };

    std::optional<int> get_gpu_id() const { return T2optional(proc_allocation.gpu_id, is_nonneg()); };
    unsigned get_num_threads() const { return proc_allocation.num_threads; };
    bool has_gpu() const { return proc_allocation.has_gpu(); };
    bool get_bind_threads() const { return proc_allocation.bind_threads; };
    bool get_bind_procs() const { return proc_allocation.bind_procs; };
};

// A Python shim that holds the information that describes an arb::context.
struct context_shim {
    arb::context context;
    context_shim(arb::context c): context{c} {}
};

// Public no arg version of make_context_shim that defaults proc_alloc to make_default_proc_allocation_shim
context_shim make_context_shim();

} // namespace pyarb
