#include <algorithm>
#include <array>
#include <cstring>
#include <functional>
#include <iomanip>
#include <ios>
#include <numeric>
#include <optional>
#include <ostream>
#include <stdexcept>
#include <vector>

#include <arbor/util/scope_exit.hpp>
#include "gpu_uuid.hpp"
#include "gpu_api.hpp"


#ifdef __linux__
extern "C" {
    #include <unistd.h>
}

std::optional<std::string> get_hostname() {
    // Hostnames can be up to 256 characters in length, however on many systems
    // it is limitted to 64.
    char name[256];
    auto result = gethostname(name, sizeof(name));
    if (result) {
        return std::nullopt;
    }
    return std::string(name);
}
#else
std::optional<std::string> get_hostname() {
    return std::nullopt;
}
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

std::runtime_error make_runtime_error(api_error_type error_code) {
    return std::runtime_error(
        std::string("gpu runtime error ")
        + error_code.name() + ": " + error_code.description());
}

// For CUDA 10 and later the uuid of all available GPUs is straightforward
// to obtain by querying cudaGetDeviceProperties for each visible device.
std::vector<uuid> get_gpu_uuids() {
    // Get number of devices.
    int ngpus = 0;
    auto status = get_device_count(&ngpus);
    if (status.no_device_found()) {
        // No GPUs detected: return an empty list.
        return {};
    }
    else if (!status) {
        throw make_runtime_error(status);
    }

    // Storage for the uuids.
    std::vector<uuid> uuids(ngpus);

    // For each GPU query CUDA runtime API for uuid.
    for (int i=0; i<ngpus; ++i) {
        DeviceProp props;
        status = get_device_properties(&props, i);
        if (!status) {
            throw make_runtime_error(status);
        }

#ifdef ARB_HIP
        // Build a unique string for the device and hash it, then
        // copy the bytes of the has to uuids[i].
        auto host = get_hostname();
        if (!host) throw std::runtime_error("Can't uniquely identify GPUs on the system");
        auto uid = std::hash<std::string>{} (*host + '-' + std::to_string(props.pciBusID) + '-' + std::to_string(props.pciDeviceID));
        auto b = reinterpret_cast<const unsigned char*>(&uid);
        std::copy(b, b+sizeof(std::size_t), uuids[i].bytes.begin());
#else
        // Copy the bytes from props.uuid to uuids[i].
        auto b = reinterpret_cast<const unsigned char*>(&props.uuid);
        std::copy(b, b+sizeof(uuid), uuids[i].bytes.begin());
#endif
    }

    return uuids;
}

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
