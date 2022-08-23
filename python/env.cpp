#include <pybind11/pybind11.h>

#include <arborenv/gpu_env.hpp>

#include "mpi.hpp"
#include "error.hpp"

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
                  auto err = ""Private GPU: Invalid MPI Communicator."";
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
              "  mpi:     The MPI communicator.");
    }
}
