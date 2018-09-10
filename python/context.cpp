#include <sstream>
#include <string>

#include <arbor/context.hpp>
#include <arbor/domain_decomposition.hpp>
#include <arbor/version.hpp>

#include "context.hpp"
#include "strings.hpp"

#include <pybind11/pybind11.h>

#ifdef ARB_MPI_ENABLED
#include "mpi.hpp"
#endif

namespace pyarb {

void register_contexts(pybind11::module& m) {
    using namespace std::string_literals;

    pybind11::class_<arb::local_resources> local_resources(m, "local_resources");
    local_resources
        .def(pybind11::init([](){return arb::get_local_resources();}),
            "By default local_resources has the total number of available threads\n"
            "detected by Arbor, and the total number of GPUs from cudaDeviceCount()")
        .def_readonly("threads", &arb::local_resources::num_threads,
            "The number of threads available locally for execution.")
        .def_readonly("gpus", &arb::local_resources::num_gpus,
            "The number of GPUs available locally for execution.")
        .def("__str__", &local_resources_string)
        .def("__repr__", &local_resources_string);

    pybind11::class_<arb::proc_allocation> proc_allocation(m, "proc_allocation");
    proc_allocation
        .def(pybind11::init<>())
        .def_readwrite("threads", &arb::proc_allocation::num_threads,
            "The number of threads available locally for execution.")
        .def_readwrite("gpu_id",   &arb::proc_allocation::gpu_id,
            "The identifier of the GPU to use.\n"
            "Corresponds to the integer index used to identify GPUs in CUDA API calls.")
        .def_property_readonly("has_gpu", &arb::proc_allocation::has_gpu,
            "Whether a GPU is being used (True/False).")
        .def("__str__", &proc_allocation_string)
        .def("__repr__", &proc_allocation_string);

    pybind11::class_<context_shim> context(m, "context");
    context
        .def(pybind11::init<>(
            [](){return context_shim(arb::make_context());}))
        .def(pybind11::init(
            [](const arb::proc_allocation& alloc){return context_shim(arb::make_context(alloc));}))
#ifdef ARB_MPI_ENABLED
        .def(pybind11::init(
            [](const arb::proc_allocation& alloc, mpi_comm_shim c){return context_shim(arb::make_context(alloc, c.comm));}))
#endif
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
        .def("__str__", [](const context_shim& c){return context_string(c.context);})
        .def("__repr__", [](const context_shim& c){return context_string(c.context);});
}

} // namespace pyarb

