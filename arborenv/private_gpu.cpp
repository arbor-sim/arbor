#ifdef ARB_HAVE_MPI

#include <stdexcept>
#include <numeric>

#include <mpi.h>

#include <arborenv/gpu_env.hpp>
#include "gpu_uuid.hpp"

namespace arbenv {

#ifdef ARB_HAVE_GPU

template <>
int find_private_gpu(MPI_Comm comm) {
    int nranks;
    int rank;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nranks);

    // Helper for testing error status of all MPI ranks.
    // Returns true if any rank passes true.
    auto test_global_error = [comm](bool local_status) -> bool {
        int l = local_status? 1: 0;
        int global_status;
        MPI_Allreduce(&l, &global_status, 1, MPI_INT, MPI_MAX, comm);
        return global_status==1;
    };

    // STEP 1: find list of locally available uuid.

    bool local_error = false;
    std::string msg;
    std::vector<uuid> uuids;
    try {
        uuids = get_gpu_uuids();
    }
    catch (const std::exception& e) {
        msg = e.what();
        local_error = true;
    }

    // STEP 2: mpi test error on any node.

    if (test_global_error(local_error)) {
        if (local_error) {
            throw std::runtime_error("unable to detect the unique id of visible GPUs: " + msg);
        }
        else {
            throw std::runtime_error("unable to detect the unique id of visible GPUs: error on another MPI rank");
        }
    }

    // STEP 3: Gather all uuids to local rank.

    // Gather number of gpus per rank.
    int ngpus = uuids.size();
    std::vector<int> gpus_per_rank(nranks);
    MPI_Allgather(&ngpus, 1, MPI_INT,
                  gpus_per_rank.data(), 1, MPI_INT,
                  comm);

    // Determine partition of gathered uuid list.
    std::vector<int> gpu_partition(nranks+1);
    std::partial_sum(gpus_per_rank.begin(), gpus_per_rank.end(), gpu_partition.begin()+1);

    // Make MPI Datatype for uuid
    MPI_Datatype uuid_mpi_type;
    MPI_Type_contiguous(sizeof(uuid), MPI_BYTE, &uuid_mpi_type);
    MPI_Type_commit(&uuid_mpi_type);

    // Gather all uuid
    std::vector<uuid> global_uuids(gpu_partition.back());
    MPI_Allgatherv(uuids.data(), ngpus, uuid_mpi_type,
                   global_uuids.data(), gpus_per_rank.data(), gpu_partition.data(),
                   uuid_mpi_type, comm);

    // Unregister uuid type.
    MPI_Type_free(&uuid_mpi_type);

    // step 4: find the local GPU
    auto gpu = assign_gpu(global_uuids, gpu_partition, rank);

    if (test_global_error(gpu.error)) {
        throw std::runtime_error(
            "Unable to assign a unique GPU to MPI rank: the CUDA_VISIBLE_DEVICES"
            " environment variable is likely incorrectly configured." );
    }

    return gpu.id;
}

#else

// return -1 -> "no gpu" when compiled without GPU support.
template <>
int find_private_gpu(MPI_Comm comm) {
    return -1;
}

#endif // def ARB_HAVE_GPU

} // namespace arbenv

#endif // def ARB_HAVE_MPI


