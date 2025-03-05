#include <iostream>
#include <optional>
#include <sstream>
#include <string>

#include <pybind11/pybind11.h>

#include <arbor/version.hpp>
#include <arbor/arbexcept.hpp>

#include <arborenv/default_env.hpp>

#include "context.hpp"
#include "conversion.hpp"
#include "error.hpp"
#include "strprintf.hpp"

#ifdef ARB_MPI_ENABLED
#include "mpi.hpp"
#endif

namespace pyarb {

// printers
std::ostream& operator<<(std::ostream& o, const context_shim& ctx) {
    auto& c = ctx.context;
    const char* gpu = arb::has_gpu(c)? "True": "False";
    const char* mpi = arb::has_mpi(c)? "True": "False";
    return
        o << "<arbor.context: num_threads " << arb::num_threads(c)
          << ", has_gpu " << gpu
          << ", has_mpi " << mpi
          << ", num_ranks " << arb::num_ranks(c)
          << ">";
}

std::ostream& operator<<(std::ostream& o, const proc_allocation_shim& alloc) {
    return o << "<arbor.proc_allocation: threads " << alloc.get_num_threads() <<
                ", gpu_id " << util::to_string(alloc.get_gpu_id()) << 
                ", bind_threads " << util::to_string(alloc.get_bind_threads()) << 
                ", bind_procs " << util::to_string(alloc.get_bind_procs()) << 
                ">";
}

// proc_alloc getter and setter (in order to assert when being set)
proc_allocation_shim::proc_allocation_shim(unsigned threads, pybind11::object gpu, bool bp, bool bt) {
    set_num_threads(threads);
    set_gpu_id(gpu);
    set_bind_procs(bp);
    set_bind_threads(bt);
}

void proc_allocation_shim::set_gpu_id(pybind11::object gpu) {
    auto gpu_id = py2optional<int>(gpu, "gpu_id must be None, or a non-negative integer", is_nonneg());
    proc_allocation.gpu_id = gpu_id.value_or(-1);
};

void proc_allocation_shim::set_num_threads(unsigned threads) {
    if (0==threads) {
        throw arb::zero_thread_requested_error(threads);
    }
    proc_allocation.num_threads = threads;
};

// generators
context_shim make_context_shim(proc_allocation_shim alloc, pybind11::object mpi, pybind11::object inter) {
#ifndef ARB_GPU_ENABLED
    if (alloc.has_gpu()) {
        throw pyarb_error("Attempt to set an GPU communicator but Arbor is not configured with GPU support.");
    }
#endif
#ifndef ARB_MPI_ENABLED
    if (!mpi.is_none() || !inter.is_none()) {
        throw pyarb_error("Attempt to set an MPI communicator but Arbor is not configured with MPI support.");
    }
#else
    const char* mpi_err_str = "mpi must be None, or a known MPI communicator type. Supported MPI implementations = native"
#ifdef ARB_WITH_MPI4PY
    ", mpi4py.";
#else
    ". Consider installing mpi4py and rebuilding Arbor.";
#endif
    if (mpi.is_none() && !inter.is_none()) {
        throw pyarb_error("Attempted to set an intercommunicator without also providing a intracommunicator.");
    }
    if (can_convert_to_mpi_comm(mpi)) {
        if (can_convert_to_mpi_comm(inter)) {
            return context_shim(arb::make_context(alloc.proc_allocation, convert_to_mpi_comm(mpi), convert_to_mpi_comm(inter)));
        }
        return context_shim(arb::make_context(alloc.proc_allocation, convert_to_mpi_comm(mpi)));
    }
    if (auto c = py2optional<mpi_comm_shim>(mpi, mpi_err_str)) {
        if (auto i = py2optional<mpi_comm_shim>(inter, mpi_err_str)) {
            return context_shim(arb::make_context(alloc.proc_allocation, c->comm, i->comm));
        }
        return context_shim(arb::make_context(alloc.proc_allocation, c->comm));
    } else {
        if (py2optional<mpi_comm_shim>(inter, mpi_err_str)) {
            throw pyarb_error("Attempted to set an intercommunicator without also providing a intracommunicator.");
        }
    }
#endif
    return context_shim{arb::make_context(alloc.proc_allocation)};
}

context_shim make_context_shim() {
    return context_shim{arb::make_context(arbenv::default_allocation())};
};

// pybind
void register_contexts(pybind11::module& m) {
    using namespace std::string_literals;
    using namespace pybind11::literals;

    // proc_allocation
    pybind11::class_<proc_allocation_shim> proc_allocation(m, "proc_allocation",
        "Enumerates the computational resources on a node to be used for simulation.");
    proc_allocation
        .def(pybind11::init<unsigned, pybind11::object, bool, bool>(),
             pybind11::kw_only(),
             "threads"_a=1,
             "gpu_id"_a=pybind11::none(),
             "bind_procs"_a=false,
             "bind_threads"_a=false,
             "Construct an allocation with arguments:\n"
             "  threads:      The number of threads available locally for execution. Must be set to 1 at minimum. 1 by default.\n"
             "  gpu_id:       The identifier of the GPU to use, None by default.\n"
             "  bind_procs:   Create process binding mask.\n"
             "  bind_threads: Create thread binding mask.\n")
        .def_property("threads", &proc_allocation_shim::get_num_threads, &proc_allocation_shim::set_num_threads,
            "The number of threads available locally for execution.")
        .def_property("bind_procs", &proc_allocation_shim::get_bind_procs, &proc_allocation_shim::set_bind_procs,
            "Try to bind MPI procs?")
        .def_property("bind_threads", &proc_allocation_shim::get_bind_threads, &proc_allocation_shim::set_bind_threads,
            "Try to bind threads?")
        .def_property("gpu_id", &proc_allocation_shim::get_gpu_id, &proc_allocation_shim::set_gpu_id,
            "The identifier of the GPU to use.\n"
            "Corresponds to the integer parameter used to identify GPUs in CUDA API calls.")
        .def_property_readonly("has_gpu", &proc_allocation_shim::has_gpu,
            "Whether a GPU is being used (True/False).")
        .def("__str__",  util::to_string<proc_allocation_shim>)
        .def("__repr__", util::to_string<proc_allocation_shim>);

    // context
    pybind11::class_<context_shim, std::shared_ptr<context_shim>> context(m, "context", "An opaque handle for the hardware resources used in a simulation.");
    context
        .def(pybind11::init(
                [](){ return make_context_shim(); }),
             "Construct a local context with proc_allocation = env.default_allocation().\n")
        .def(pybind11::init(
                 [](unsigned threads, pybind11::object gpu, pybind11::object mpi, pybind11::object inter, bool bind_p, bool bind_t){
                     return make_context_shim(proc_allocation_shim(threads, gpu, bind_p, bind_t), mpi, inter);
                 }),
             pybind11::kw_only(),
             "threads"_a=1,
             "gpu_id"_a=pybind11::none(),
             "mpi"_a=pybind11::none(),
             "inter"_a=pybind11::none(),
             "bind_procs"_a=false,
             "bind_threads"_a=false,
             "Construct a context with arguments:\n"
             "  threads: The number of threads available locally for execution. Must be set to 1 at minimum. 1 by default.\n"
             "  gpu_id:  The identifier of the GPU to use, None by default. Only available if arbor.__config__['gpu']!=\"none\".\n"
             "  mpi:     The MPI communicator, None by default. Only available if arbor.__config__['mpi']==True.\n"
             "  inter:   An MPI intercommunicator used to connect to external simulations, None by default. Only available if arbor.__config__['mpi']==True.\n"
             "  bind_procs:   Create process binding mask.\n"
             "  bind_threads: Create thread binding mask.")
        .def(pybind11::init(
                 [](proc_allocation_shim alloc, pybind11::object mpi, pybind11::object inter) {
                     return make_context_shim(alloc, mpi, inter);
                 }),
             "alloc"_a,
             pybind11::kw_only(),
             "mpi"_a=pybind11::none(),
             "inter"_a=pybind11::none(),
             "Construct a context with arguments:\n"
             "  alloc:   The computational resources to be used for the simulation.\n"
             "  mpi:     The MPI communicator, None by default. Only available if arbor.__config__['mpi']==True.\n"
             "  inter:   An MPI intercommunicator used to connect to external simulations, None by default. Only available if arbor.__config__['mpi']==True.\n")
        .def_property_readonly("has_mpi", [](const context_shim& ctx){return arb::has_mpi(ctx.context);},
                               "Whether the context uses MPI for distributed communication.")
        .def_property_readonly("has_gpu", [](const context_shim& ctx){return arb::has_gpu(ctx.context);},
                               "Whether the context has a GPU.")
        .def_property_readonly("threads", [](const context_shim& ctx){return arb::num_threads(ctx.context);},
                               "The number of threads in the context's thread pool.")
        .def_property_readonly("ranks", [](const context_shim& ctx){return arb::num_ranks(ctx.context);},
                               "The number of distributed domains (equivalent to the number of MPI ranks).")
        .def_property_readonly("rank", [](const context_shim& ctx){return arb::rank(ctx.context);},
                               "The numeric id of the local domain (equivalent to MPI rank).")
        .def("__str__", util::to_string<context_shim>)
        .def("__repr__", util::to_string<context_shim>);
}

} // namespace pyarb
