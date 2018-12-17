#include <iostream>

#include <sstream>
#include <string>

#include <arbor/context.hpp>
#include <arbor/domain_decomposition.hpp>
#include <arbor/version.hpp>

#include "context.hpp"
#include "strings.hpp"

#include <pybind11/cast.h>
#include <pybind11/pybind11.h>

#ifdef ARB_MPI_ENABLED
#include "mpi.hpp"
#endif

namespace pyarb {

void register_contexts(pybind11::module& m) {
    using namespace std::string_literals;
    using namespace pybind11::literals;

    pybind11::class_<arb::proc_allocation> proc_allocation(m, "proc_allocation");
    proc_allocation
        .def(pybind11::init<>())
        .def(pybind11::init<int, int>(), "threads"_a, "gpu"_a=-1)
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
        .def(pybind11::init(
            [](int threads, pybind11::object gpu, pybind11::object mpi){
                arb::proc_allocation alloc(threads, gpu.is_none()? -1: pybind11::cast<int>(gpu));
                if (mpi.is_none()) {
                    return context_shim(arb::make_context(alloc));
                }
                auto& c = pybind11::cast<mpi_comm_shim&>(mpi);
                return context_shim(arb::make_context(alloc, c.comm));
            }),
            "threads"_a=1, "gpu"_a=pybind11::none(), "mpi"_a=pybind11::none())
#else
        .def(pybind11::init(
            [](int threads, pybind11::object gpu){
                int gpu_id = gpu.is_none()? -1: pybind11::cast<int>(gpu);
                return context_shim(arb::make_context(arb::proc_allocation(threads, gpu_id)));
            }),
            "threads"_a=1, "gpu"_a=pybind11::none())
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

