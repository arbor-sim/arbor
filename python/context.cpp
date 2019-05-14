#include <iostream>
#include <sstream>
#include <string>

#include <arbor/context.hpp>
#include <arbor/version.hpp>

#include <pybind11/pybind11.h>

#include "context.hpp"
#include "conversion.hpp"
#include "error.hpp"
#include "strings.hpp"

#ifdef ARB_MPI_ENABLED
#include "mpi.hpp"
#endif

namespace pyarb {

namespace {
auto is_nonneg_int = [](auto&& t){ return ((t==int(t) and t>=0)); };
}

// A Python shim that holds the information that describes an arb::proc_allocation.
struct proc_allocation_shim {
    using opt_int = arb::util::optional<int>;

    opt_int gpu_id = {};
    int num_threads = 1;

    proc_allocation_shim(): proc_allocation_shim(1, pybind11::none()) {}

    proc_allocation_shim(int threads, pybind11::object gpu) {
        set_num_threads(threads);
        set_gpu_id(gpu);
    }

    // getter and setter (in order to assert when being set)
    void set_gpu_id(pybind11::object gpu) {
        gpu_id = py2optional<int>(gpu, "gpu id must be None, or a non-negative integer.", is_nonneg_int);
    };

    void set_num_threads(int threads) {
        pyarb::assert_throw(is_nonneg_int(threads), "threads must be a non-negative integer.");
        num_threads = threads;
    };

    const opt_int get_gpu_id()      const { return gpu_id; }
    const int     get_num_threads() const { return num_threads; }
    const bool    has_gpu()         const { return (gpu_id.value_or(-1) >= 0) ? true : false; }

    // helper function to use arb::make_context(arb::proc_allocation)
    arb::proc_allocation allocation() const {
        return arb::proc_allocation(num_threads, gpu_id.value_or(-1));
    }
};

// Helper template for printing C++ optional types in Python.
// Prints either the value, or None if optional value is not set.
template <typename T>
std::string to_string(const arb::util::optional<T>& o) {
    if (!o) return "None";

    std::stringstream s;
    s << *o;
    return s.str();
}

std::string proc_alloc_string(const proc_allocation_shim& a) {
    std::stringstream s;
    s << "<hardware resource allocation: threads " << a.num_threads
      << ", gpu id " << to_string(a.gpu_id);
    s << ">";
    return s.str();
}

// helper functions to wrap arb::make_context
arb::context make_context_shim(const proc_allocation_shim& Alloc) {
    return arb::make_context(Alloc.allocation());
}

template <typename Comm>
arb::context make_context_shim(const proc_allocation_shim& Alloc, Comm comm) {
    return arb::make_context(Alloc.allocation(), comm);
}

void register_contexts(pybind11::module& m) {
    using namespace std::string_literals;
    using namespace pybind11::literals;
    using opt_int = arb::util::optional<int>;
#ifdef ARB_MPI_ENABLED
    using opt_mpi_comm = arb::util::optional<mpi_comm_shim>;
#endif

// proc_allocation
    pybind11::class_<proc_allocation_shim> proc_allocation(m, "proc_allocation",
        "Enumerates the computational resources on a node to be used for simulation.");
    proc_allocation
        .def(pybind11::init<int, pybind11::object>(),
            "threads"_a=1, "gpu_id"_a=pybind11::none(),
            "Construct an allocation with arguments:\n"
            "  threads: The number of threads available locally for execution (defaults to 1).\n"
            "  gpu_id:  The index of the GPU to use (defaults to None for no GPU).\n")
        .def_property("threads", &proc_allocation_shim::get_num_threads, &proc_allocation_shim::set_num_threads,
            "The number of threads available locally for execution.")
        .def_property("gpu_id", &proc_allocation_shim::get_gpu_id, &proc_allocation_shim::set_gpu_id,
            "The identifier of the GPU to use.\n"
            "Corresponds to the integer index used to identify GPUs in CUDA API calls.")
	.def_property_readonly("has_gpu", &proc_allocation_shim::has_gpu,
            "Whether a GPU is being used (True/False).")
        .def("__str__", &proc_alloc_string)
        .def("__repr__", &proc_alloc_string);

// context
    pybind11::class_<context_shim> context(m, "context", "An opaque handle for the hardware resources used in a simulation.");
    context
        .def(pybind11::init<>(
            [](){return context_shim(arb::make_context());}),
            "Construct a local context with one thread, no GPU, no MPI by default.\n"
            )
        .def(pybind11::init(
            [](const proc_allocation_shim& alloc){
                return context_shim(make_context_shim(alloc));
            }),
             "alloc"_a,
             "Construct a local context with argument:\n"
             "  alloc:   The computational resources to be used for the simulation.\n")
#ifdef ARB_MPI_ENABLED
        .def(pybind11::init(
            [](const proc_allocation_shim& alloc, pybind11::object mpi){
                if (mpi.is_none()) {
                    return context_shim(make_context_shim(alloc));
                }
                opt_mpi_comm c = py2optional<mpi_comm_shim>(mpi,
                        "mpi must be None, or an MPI communicator.");
                auto comm = c.value_or(MPI_COMM_WORLD).comm;
                return context_shim(make_context_shim(alloc, comm));
            }),
             "alloc"_a, "mpi"_a=pybind11::none(),
             "Construct a distributed context with arguments:\n"
             "  alloc:   The computational resources to be used for the simulation.\n"
             "  mpi:     The MPI communicator (defaults to None for no MPI).\n")
        .def(pybind11::init(
            [](int threads, pybind11::object gpu, pybind11::object mpi){
                opt_int gpu_id = py2optional<int>(gpu,
                        "gpu id must be None, or a non-negative integer.", is_nonneg_int);
                arb::proc_allocation alloc(threads, gpu_id.value_or(-1));
                if (mpi.is_none()) {
                    return context_shim(arb::make_context(alloc));
                }
                opt_mpi_comm c = py2optional<mpi_comm_shim>(mpi,
                        "mpi must be None, or an MPI communicator.");
                auto comm = c.value_or(MPI_COMM_WORLD).comm;
                return context_shim(arb::make_context(alloc, comm));
            }),
             "threads"_a=1, "gpu_id"_a=pybind11::none(), "mpi"_a=pybind11::none(),
             "Construct a distributed context with arguments:\n"
             "  threads: The number of threads available locally for execution (defaults to 1).\n"
             "  gpu_id:  The index of the GPU to use (defaults to None for no GPU).\n"
             "  mpi:     The MPI communicator (defaults to None for no MPI).\n")
#else
        .def(pybind11::init(
            [](int threads, pybind11::object gpu){
                opt_int gpu_id = py2optional<int>(gpu,
                        "gpu id must be None, or a non-negative integer.", is_nonneg_int);
                return context_shim(arb::make_context(arb::proc_allocation(threads, gpu_id.value_or(-1))));
            }),
             "threads"_a=1, "gpu_id"_a=pybind11::none(),
             "Construct a local context with arguments:\n"
             "  threads: The number of threads available locally for execution (defaults to 1).\n"
             "  gpu_id:  The index of the GPU to use (defaults to None for no GPU).\n")
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
