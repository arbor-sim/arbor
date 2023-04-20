#include <pybind11/pybind11.h>
#include "conversion.hpp"

#include <arborenv/gpu_env.hpp>
#include <arborenv/concurrency.hpp>
#include <arborenv/default_env.hpp>

#include "mpi.hpp"
#include "error.hpp"
#include "context.hpp"

namespace pyarb {

    void register_arborenv(pybind11::module& m) {
        auto s = m.def_submodule("env", "Wrappers for arborenv.");
        s.def("find_private_gpu",
              [] (pybind11::object mpi) {
#ifndef ARB_GPU_ENABLED
                  throw pyarb_error("Private GPU: Arbor is not configured with GPU support.");
#else
#ifndef ARB_MPI_ENABLED
                  throw pyarb_error("Private GPU: Arbor is not configured with MPI.");
#else
                  auto err = "Private GPU: Invalid MPI Communicator.";
                  if (can_convert_to_mpi_comm(mpi)) {
                      return arbenv::find_private_gpu(can_convert_to_mpi_comm(mpi));
                  }
                  else if (auto c = py2optional<mpi_comm_shim>(mpi, err)) {
                      return arbenv::find_private_gpu(c->comm);
                  } else {
                      throw pyarb_error(err);
                  }
#endif
#endif
              },
              "Identify a private GPU id per node, only available if built with GPU and MPI.\n"
              "  mpi:     The MPI communicator.")
        .def("thread_concurrency", []() -> unsigned {return arbenv::thread_concurrency();},
            "Attempts to detect the number of locally available CPU cores. Returns 1 if unable to detect the number of cores. Use with caution in combination with MPI.")
        .def("get_env_num_threads", []() -> unsigned {return arbenv::get_env_num_threads();},
            "Retrieve user-specified number of threads to use from the environment variable ARBENV_NUM_THREADS.")
        .def("default_concurrency", []() -> unsigned {return arbenv::default_concurrency();},
            "Returns number of threads to use from get_env_num_threads(), or else from thread_concurrency() if get_env_num_threads() returns zero.")
        .def("default_gpu", []() -> std::optional<int> {return T2optional(arbenv::default_gpu(), is_nonneg());},
            "Determine GPU id to use from the ARBENV_GPU_ID environment variable, or from the first available GPU id of those detected.")
        .def("default_allocation", []() -> proc_allocation_shim {return proc_allocation_shim{arbenv::default_allocation()};},
            "Attempts to detect the number of locally available CPU cores. Returns 1 if unable to detect the number of cores. Use with caution in combination with MPI.");
    }
}
