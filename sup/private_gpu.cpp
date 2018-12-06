#include <numeric>
#include <iostream>

#include <mpi.h>

#include <sup/gpu.hpp>

#include "gpu_uuid.hpp"

namespace sup {

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

    //std::cout << "rank -- " << rank << " of -- " << nranks << std::endl;


    // STEP 1: find list of locally available uuid.

    bool local_error = false;
    std::vector<uuid> uuids;
    try {
        uuids = get_gpu_uuids();
    }
    catch (const std::exception& e) {
        local_error = true;
    }
    //std::cout << "error " << rank << " -- " << (local_error? "fail": "pass") << std::endl;

    // STEP 2: mpi test error on any node.

    if (test_global_error(local_error)) {
        std::cerr << "error quering devices" << std::endl;
        return -1;
    }

    for (auto id: uuids) std::cout << "-- " << rank << " -- " << id << "\n";

    // STEP 3: Gather all uuids to local rank.

    // Gather number of gpus per rank.
    int ngpus = uuids.size();
    std::vector<int> gpus_per_rank(nranks);
    MPI_Allgather(&ngpus, 1, MPI_INT,
                  gpus_per_rank.data(), 1, MPI_INT,
                  comm);

    //std::cout << "gpu_per_rank["; for (auto i: gpus_per_rank) std::cout << i << " "; std::cout << "]" << std::endl;

    // Determine partition of gathered uuid list.
    std::vector<int> gpu_partition(nranks+1);
    std::partial_sum(gpus_per_rank.begin(), gpus_per_rank.end(), gpu_partition.begin()+1);

    //std::cout << "gpu_part["; for (auto i: gpu_partition) std::cout << i << " "; std::cout << "]" << std::endl;

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

    std::cout << "-- " << rank << " gotta gpu: "
        << "{" << gpu.id << ", " << (gpu.error? "error": "ok") << "}" << std::endl;

    if (test_global_error(gpu.error)) {
        std::cerr << "error determining groups" << std::endl;
        return -1;
    }

    return gpu.id;
}

} // namespace sup
