#include <algorithm>
#include <array>
#include <cstring>
#include <functional>
#include <iomanip>
#include <ios>
#include <numeric>
#include <ostream>
#include <stdexcept>
#include <vector>

#include <cuda_runtime.h>

#include <arbor/util/scope_exit.hpp>
#include "gpu_uuid.hpp"


// CUDA 10 allows GPU uuid to be queried via cudaGetDeviceProperties.
// Previous versions require the CUDA NVML library to get uuid.
//
// ARBENV_USE_NVML will be defined at configuration time if using
// CUDA version 9.

#ifdef ARBENV_USE_NVML
    #include <nvml.h>
#endif

using arb::util::on_scope_exit;

namespace arbenv {

// Test GPU uids for equality
bool operator==(const uuid& lhs, const uuid& rhs) {
    for (auto i=0u; i<lhs.bytes.size(); ++i) {
        if (lhs.bytes[i]!=rhs.bytes[i]) return false;
    }
    return true;
}

// Strict lexographical ordering of GPU uids
bool operator<(const uuid& lhs, const uuid& rhs) {
    for (auto i=0u; i<lhs.bytes.size(); ++i) {
        if (lhs.bytes[i]<rhs.bytes[i]) return true;
        if (lhs.bytes[i]>lhs.bytes[i]) return false;
    }
    return false;
}

std::ostream& operator<<(std::ostream& o, const uuid& id) {
    std::ios old_state(nullptr);
    old_state.copyfmt(o);
    o << std::hex << std::setfill('0');

    bool first = true;
    int ranges[6] = {0, 4, 6, 8, 10, 16};
    for (int i=0; i<5; ++i) {
        if (!first) o << "-";
        for (auto j=ranges[i]; j<ranges[i+1]; ++j) {
            o << std::setw(2) << (int)id.bytes[j];
        }
        first = false;
    }
    o.copyfmt(old_state);
    return o;
}

std::runtime_error make_runtime_error(cudaError_t error_code) {
    return std::runtime_error(
        std::string("cuda runtime error ")
        + cudaGetErrorName(error_code) + ": " + cudaGetErrorString(error_code));
}

#ifndef ARBENV_USE_NVML

// For CUDA 10 and later the uuid of all available GPUs is straightforward
// to obtain by querying cudaGetDeviceProperties for each visible device.
std::vector<uuid> get_gpu_uuids() {
    // Get number of devices.
    int ngpus = 0;
    auto status = cudaGetDeviceCount(&ngpus);
    if (status==cudaErrorNoDevice) {
        // No GPUs detected: return an empty list.
        return {};
    }
    else if (status!=cudaSuccess) {
        throw make_runtime_error(status);
    }

    // Storage for the uuids.
    std::vector<uuid> uuids(ngpus);

    // For each GPU query CUDA runtime API for uuid.
    for (int i=0; i<ngpus; ++i) {
        cudaDeviceProp props;
        status = cudaGetDeviceProperties(&props, i);
        if (status!=cudaSuccess) {
            throw make_runtime_error(status);
        }

        // Copy the bytes from props.uuid to uuids[i].
        auto b = reinterpret_cast<const unsigned char*>(&props.uuid);
        std::copy(b, b+sizeof(uuid), uuids[i].bytes.begin());
    }

    return uuids;
}

#else

std::runtime_error make_runtime_error(nvmlReturn_t error_code) {
    return std::runtime_error(
        std::string("cuda nvml runtime error: ") + nvmlErrorString(error_code));
}

// Split CUDA_VISIBLE_DEVICES variable string into a list of integers.
// The environment variable can have spaces, and the order is important:
// i.e. "0,1" is not the same as "1,0".
//      CUDA_VISIBLE_DEVICES="1,0"
//      CUDA_VISIBLE_DEVICES="0, 1"
// The CUDA run time parses the list until it finds an error, then returns
// the partial list.
// i.e.
//      CUDA_VISIBLE_DEVICES="1, 0, hello" -> {1,0}
//      CUDA_VISIBLE_DEVICES="hello, 1" -> {}
// All non-numeric characters at end of a value appear to be ignored:
//      CUDA_VISIBLE_DEVICES="0a,1" -> {0,1}
//      CUDA_VISIBLE_DEVICES="a0,1" -> {}
// This doesn't try too hard to check for all possible errors.
std::vector<int> parse_visible_devices(std::string str, int ngpu) {
    std::vector<int> values;
    std::istringstream ss(str);
    while (ss) {
        int v;
        if (ss >> v) {
            if (v<0 || v>=ngpu) break;
            values.push_back(v);
            while (ss && ss.get()!=',');
        }
    }
    return values;
}

// Take a uuid string with the format:
//      GPU-f1fd7811-e4d3-4d54-abb7-efc579fb1e28
// And convert to a 16 byte sequence
//
// Assume that the intput string is correctly formatted.
uuid string_to_uuid(char* str) {
    uuid result;
    unsigned n = std::strlen(str);

    // Remove the "GPU" from front of string, and the '-' hyphens, e.g.:
    //      GPU-f1fd7811-e4d3-4d54-abb7-efc579fb1e28
    // becomes
    //      f1fd7811e4d34d54abb7efc579fb1e28
    std::remove_if(str, str+n, [](char c){return !std::isxdigit(c);});

    // Converts a single hex character, i.e. 0123456789abcdef, to int
    // Assumes that input is a valid hex character.
    auto hex_c2i = [](unsigned char c) -> unsigned char {
        c = std::tolower(c);
        return std::isalpha(c)? c-'a'+10: c-'0';
    };

    // Convert pairs of characters into single bytes.
    for (int i=0; i<16; ++i) {
        const char* s = str+2*i;
        result.bytes[i] = (hex_c2i(s[0])<<4) + hex_c2i(s[1]);
    }

    return result;
}

// For CUDA 9 the only way to get gpu uuid is via NVML.
// NVML can be used to query all GPU devices, not just the
// devices that have been made visible to the calling process.
// Hence, there are two steps to finding the uuid of visible devices:
// 1. Query the environment variable CUDA_VISIBLE_DEVICES to
//    determine which devices are locally visible, and to enumerate
//    them correctly.
// 2. Query NVML for the uuid of each visible device.
std::vector<uuid> get_gpu_uuids() {
    // Get number of devices.
    int ngpus = 0;
    auto cuda_status = cudaGetDeviceCount(&ngpus);
    if (cuda_status==cudaErrorNoDevice) return {};
    else if (cuda_status!=cudaSuccess) throw make_runtime_error(cuda_status);

    // Attempt to initialize nvml
    auto nvml_status = nvmlInit();
    const bool nvml_init = (nvml_status==NVML_ERROR_ALREADY_INITIALIZED);
    if (!nvml_init && nvml_status!=NVML_SUCCESS) {
        throw make_runtime_error(nvml_status);
    }
    auto nvml_guard = on_scope_exit([nvml_init](){if (!nvml_init) nvmlShutdown();});

    // store the uuids
    std::vector<uuid> uuids;

    // find the number of available GPUs
    unsigned count = -1;
    nvml_status = nvmlDeviceGetCount(&count);
    if (nvml_status!=NVML_SUCCESS) throw make_runtime_error(nvml_status);

    // Indexes of GPUs available on this rank
    std::vector<int> device_ids;

    // Test if the environment variable CUDA_VISIBLE_DEVICES has been set.
    const char* visible_device_env = std::getenv("CUDA_VISIBLE_DEVICES");
    // If set, attempt to parse the device ids from it.
    if (visible_device_env) {
        // Parse the gpu ids from the environment variable
        device_ids = parse_visible_devices(visible_device_env, count);
        if ((unsigned)ngpus != device_ids.size()) {
            // Mismatch between device count detected by cuda runtime
            // and that set in environment variable.
            throw std::runtime_error(
                "Mismatch between the number of devices in CUDA_VISIBLE_DEVICES"
                " and the number of devices detected by cudaGetDeviceCount.");
        }
    }
    // Not set, so all devices must be available.
    else {
        device_ids.resize(count);
        std::iota(device_ids.begin(), device_ids.end(), 0);
    }

    // For each device id, query NVML for the device's uuid.
    for (int i: device_ids) {
        char buffer[NVML_DEVICE_UUID_BUFFER_SIZE];
        // get handle of gpu with index i
        nvmlDevice_t handle;
        nvml_status = nvmlDeviceGetHandleByIndex(i, &handle);
        if (nvml_status!=NVML_SUCCESS) throw make_runtime_error(nvml_status);

        // get uuid as a string with format GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
        nvml_status = nvmlDeviceGetUUID(handle, buffer, sizeof(buffer));
        if (nvml_status!=NVML_SUCCESS) throw make_runtime_error(nvml_status);

        uuids.push_back(string_to_uuid(buffer));
    }

    return uuids;
}
#endif // ndef ARBENV_USE_NVML

// Compare two sets of uuids
//   1: both sets are identical
//  -1: some common elements
//   0: no common elements
// Each set is described by a pair of iterators.
template <typename I>
int compare_gpu_groups(std::pair<I,I> l, std::pair<I,I> r) {
    auto range_size = [] (auto& rng) { return std::distance(rng.first, rng.second);};
    if (range_size(l)<range_size(r)) {
        std::swap(l, r);
    }

    unsigned count = 0;
    for (auto it=l.first; it!=l.second; ++it) {
        if (std::find(r.first, r.second, *it)!=r.second) ++count;
    }

    // test for complete match
    if (count==range_size(l) && count==range_size(r)) return 1;
    // test for partial match
    if (count) return -1;
    return 0;
}

gpu_rank assign_gpu(const std::vector<uuid>& uids,
                    const std::vector<int>&  uid_part,
                    int rank)
{
    // Determine the number of ranks in MPI communicator
    auto nranks = uid_part.size()-1;

    // Helper that generates the range of gpu id for rank i
    auto make_group = [&] (int i) {
        return std::make_pair(uids.begin()+uid_part[i], uids.begin()+uid_part[i+1]);
    };

    // The list of ranks that share the same GPUs as this rank (including this rank).
    std::vector<int> neighbors;

    // The gpu uid range for this rank
    auto local_gpus = make_group(rank);

    // Find all ranks with the same set of GPUs as this rank.
    for (std::size_t i=0; i<nranks; ++i) {
        auto other_gpus = make_group(i);
        auto match = compare_gpu_groups(local_gpus, other_gpus);
        if (match==1) { // found a match
            neighbors.push_back(i);
        }
        else if (match==-1) { // partial match, which is not permitted
            return {};
        }
        // case where match==0 can be ignored.
    }

    // Determine the position of this rank in the sorted list of ranks.
    int pos_in_group =
        std::distance(
            neighbors.begin(),
            std::find(neighbors.begin(), neighbors.end(), rank));

    // The number of GPUs available to the ranks.
    int ngpu_in_group = std::distance(local_gpus.first, local_gpus.second);

    // Assign GPUs to the first ngpu ranks. If there are more ranks than GPUs,
    // some ranks will not be assigned a GPU (return -1).
    return pos_in_group<ngpu_in_group? gpu_rank(pos_in_group): gpu_rank(-1);
}

} // namespace arbenv
