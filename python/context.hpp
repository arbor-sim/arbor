#pragma once

#include <pybind11/pybind11.h>
#include <arbor/context.hpp>
#include <arborenv/default_env.hpp>

namespace pyarb {

struct proc_allocation_shim;

// A Python shim that holds the information that describes an arb::proc_allocation.
struct proc_allocation_shim {
    arb::proc_allocation proc_allocation;
    proc_allocation_shim(arb::proc_allocation pa) : proc_allocation{pa} {}
    proc_allocation_shim(unsigned, pybind11::object);
    proc_allocation_shim(): proc_allocation_shim(1, pybind11::none()) {}

    // getter and setter (in order to assert when being set)
    void set_gpu_id(pybind11::object);

    void set_num_threads(unsigned);

    std::optional<int> get_gpu_id() const { return proc_allocation.gpu_id; }
    unsigned get_num_threads() const { return proc_allocation.num_threads; }
    bool has_gpu() const { return bool(proc_allocation.gpu_id); }
};

// A Python shim that holds the information that describes an arb::context.
struct context_shim {
    arb::context context;
    context_shim(arb::context c): context{c} {}
};

context_shim make_context_shim(proc_allocation_shim = arbenv::default_allocation(), pybind11::object = pybind11::none());

} // namespace pyarb
