#pragma once

#include <array>
#include <ostream>
#include <vector>

namespace arbenv {

// Store cudaUUID_t in a byte array for easy type punning and comparison.
// 128 bit uuids are not just for GPUs: they are used in many applications,
// so call the type uuid, instead of a gpu-specific name.
// Interpret uuids in the most common storage format: big-endian.
struct alignas(sizeof(void*)) uuid {
    std::array<unsigned char, 16> bytes =
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
};

// Test GPU uids for equality
bool operator==(const uuid& lhs, const uuid& rhs);

// Strict lexographical ordering of GPU uids
bool operator<(const uuid& lhs, const uuid& rhs);

// Print uuid in big-endian format, e.g. f1fd7811-e4d3-4d54-abb7-efc579fb1e28
std::ostream& operator<<(std::ostream& o, const uuid& id);

// Return the uuid of gpu devices visible to this process
// Throws std::runtime_error if there was an error on any CUDA runtime calls.
std::vector<uuid> get_gpu_uuids();

struct gpu_rank {
    bool error = true;
    int id = -1;

    explicit gpu_rank(int id): error(false), id(id) {}
    gpu_rank() = default;
};

gpu_rank assign_gpu(const std::vector<uuid>& uids, const std::vector<int>&  uid_part, int rank);

} // namespace arbenv
