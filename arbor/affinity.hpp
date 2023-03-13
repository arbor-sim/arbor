#pragma once

#include <string>
#include <cerrno>

#ifdef ARB_HAVE_HWLOC
#include <hwloc.h>
#endif

#include <arbor/arbexcept.hpp>
#include <arbor/util/scope_exit.hpp>

namespace arb {

enum class affinity_kind {
  thread,
  process
};

struct ARB_SYMBOL_VISIBLE arb_hwloc_error: arbor_exception {
    arb_hwloc_error(const std::string& msg, const std::string& err):
        arbor_exception(
            std::string{"Arbor tried to use HWLOC and an operation failed with an error.\n"
            "Please disable `bind_procs`/`bind_threads` on the current `proc_allocation` and restart.\n"
            "The problematic operation was: "}
            + msg
            + std::string{"\nIt returned this error:\n"}
            + err) {}
};

#ifdef ARB_HAVE_HWLOC

inline
void hwloc(int err, const std::string& msg) {
    if (0 != err) throw arb_hwloc_error(msg, std::string{strerror(err)});
}

inline
void set_affinity(int index, int count, affinity_kind kind) {
    // Create the topology and ensure we don't leak it
    auto topology = hwloc_topology_t{};
    auto guard = util::on_scope_exit([&] { hwloc_topology_destroy(topology); });
    hwloc(hwloc_topology_init(&topology), "Topo init");
    hwloc(hwloc_topology_load(topology), "Topo load");
    // Fetch our current restrictions and apply them to our topology
    hwloc_cpuset_t cpus{};
    hwloc(hwloc_get_cpubind(topology, cpus, HWLOC_CPUBIND_PROCESS), "Get cpuset.");
    hwloc(hwloc_topology_restrict(topology, cpus, 0), "Topo restrict.");
    // Extract the root object describing the full local node
    auto root = hwloc_get_root_obj(topology);
    // Allocate one set per item
    auto cpusets = std::vector<hwloc_cpuset_t>(count, {});
    // Distribute items over topology, giving each of them as much private cache
    // as possible and keeping them locally in number order.
    hwloc(hwloc_distrib(topology,
                        &root, 1,                        // single root for the full machine
                        cpusets.data(), cpusets.size(),  // one cpuset for each thread
                        INT_MAX,                         // maximum available level = Logical Cores
                        0),                              // No flags
          "Distribute");
    if (kind == affinity_kind::thread) {
        // Bind threads to a single PU.
        hwloc(hwloc_bitmap_singlify(cpusets[index]), "Singlify cpuset");
        // Now bind
        hwloc(hwloc_set_cpubind(topology, cpusets[index], HWLOC_CPUBIND_THREAD),
              "Thread binding");
    }
    else if (kind == affinity_kind::process) {
        hwloc(hwloc_set_cpubind(topology, cpusets[index], HWLOC_CPUBIND_PROCESS),
              "Process binding");
    }
    else {
        throw arbor_internal_error{"Unreachable!"};
    }
}

#else

inline void set_affinity(int, int, affinity_kind) { throw arb_feature_disabled{"Binding."}; }

#endif

}
