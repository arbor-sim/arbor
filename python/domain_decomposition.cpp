#include <limits>
#include <string>
#include <sstream>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <arbor/context.hpp>
#include <arbor/domain_decomposition.hpp>
#include <arbor/load_balance.hpp>

#include "context.hpp"
#include "error.hpp"
#include "recipe.hpp"
#include "strprintf.hpp"

namespace pyarb {

std::string gd_string(const arb::group_description& g) {
    return util::pprintf(
        "<arbor.group_description: num_cells {}, gids [{}], {}, {}>",
        g.gids.size(), util::csv(g.gids, 5), g.kind, g.backend);
}

std::string dd_string(const arb::domain_decomposition& d) {
    return util::pprintf(
        "<arbor.domain_decomposition: domain_id {}, num_domains {}, num_local_cells {}, num_global_cells {}, groups {}>",
        d.domain_id, d.num_domains, d.num_local_cells, d.num_global_cells, d.groups.size());
}

std::string ph_string(const arb::partition_hint& h) {
    return util::pprintf(
        "<arbor.partition_hint: cpu_group_size {}, gpu_group_size {}, prefer_gpu {}>",
        h.cpu_group_size, h.gpu_group_size, (h.prefer_gpu == 1) ? "True" : "False");
}

void register_domain_decomposition(pybind11::module& m) {
    using namespace pybind11::literals;

    // Group description
    pybind11::class_<arb::group_description> group_description(m, "group_description",
        "The indexes of a set of cells of the same kind that are grouped together in a cell group.");
    group_description
        .def(pybind11::init<arb::cell_kind, std::vector<arb::cell_gid_type>, arb::backend_kind>(),
            "Construct a group description with cell kind, list of gids, and backend kind.",
            "kind"_a, "gids"_a, "backend"_a)
        .def_readonly("kind", &arb::group_description::kind,
            "The type of cell in the cell group.")
        .def_readonly("gids", &arb::group_description::gids,
            "The list of gids of the cells in the group.")
        .def_readonly("backend", &arb::group_description::backend,
            "The hardware backend on which the cell group will run.")
        .def("__str__",  &gd_string)
        .def("__repr__", &gd_string);

    // Partition hint
    pybind11::class_<arb::partition_hint> partition_hint(m, "partition_hint",
        "Provide a hint on how the cell groups should be partitioned.");
    partition_hint
        .def(pybind11::init<std::size_t, std::size_t, bool>(),
            "cpu_group_size"_a = 1, "gpu_group_size"_a = std::numeric_limits<std::size_t>::max(), "prefer_gpu"_a = true,
            "Construct a partition hint with arguments:\n"
            "  cpu_group_size: The size of cell group assigned to CPU, each cell in its own group by default.\n"
            "                  Must be positive, else set to default value.\n"
            "  gpu_group_size: The size of cell group assigned to GPU, all cells in one group by default.\n"
            "                  Must be positive, else set to default value.\n"
            "  prefer_gpu:     Whether GPU is preferred, True by default.")
        .def_readwrite("cpu_group_size", &arb::partition_hint::cpu_group_size,
                                        "The size of cell group assigned to CPU.")
        .def_readwrite("gpu_group_size", &arb::partition_hint::gpu_group_size,
                                        "The size of cell group assigned to GPU.")
        .def_readwrite("prefer_gpu", &arb::partition_hint::prefer_gpu,
                                        "Whether GPU usage is preferred.")
        .def_property_readonly_static("max_size",  [](pybind11::object) { return arb::partition_hint::max_size; },
                                        "Get the maximum size of cell groups.")
        .def("__str__",  &ph_string)
        .def("__repr__", &ph_string);

    // Domain decomposition
    pybind11::class_<arb::domain_decomposition> domain_decomposition(m, "domain_decomposition",
        "The domain decomposition is responsible for describing the distribution of cells across cell groups and domains.");
    domain_decomposition
        .def(pybind11::init<>())
        .def("gid_domain",
            [](const arb::domain_decomposition& d, arb::cell_gid_type gid) {
                return d.gid_domain(gid);
            },
            "Query the domain id that a cell assigned to (using global identifier gid).",
            "gid"_a)
        .def_readonly("num_domains", &arb::domain_decomposition::num_domains,
            "Number of domains that the model is distributed over.")
        .def_readonly("domain_id", &arb::domain_decomposition::domain_id,
            "The index of the local domain.\n"
            "Always 0 for non-distributed models, and corresponds to the MPI rank for distributed runs.")
        .def_readonly("num_local_cells", &arb::domain_decomposition::num_local_cells,
            "Total number of cells in the local domain.")
        .def_readonly("num_global_cells", &arb::domain_decomposition::num_global_cells,
            "Total number of cells in the global model (sum of num_local_cells over all domains).")
        .def_readonly("groups", &arb::domain_decomposition::groups,
            "Descriptions of the cell groups on the local domain.")
        .def("__str__",  &dd_string)
        .def("__repr__", &dd_string);

    // Partition load balancer
    // The Python recipe has to be shimmed for passing to the function that takes a C++ recipe.
    m.def("partition_load_balance",
        [](std::shared_ptr<py_recipe>& recipe, const context_shim& ctx, arb::partition_hint_map hint_map) {
            try {
                return arb::partition_load_balance(py_recipe_shim(recipe), ctx.context, std::move(hint_map));
            }
            catch (...) {
                py_reset_and_throw();
                throw;
            }
        },
        "Construct a domain_decomposition that distributes the cells in the model described by recipe\n"
        "over the distributed and local hardware resources described by context.\n"
        "Optionally, provide a dictionary of partition hints for certain cell kinds, by default empty.",
        "recipe"_a, "context"_a, "hints"_a=arb::partition_hint_map{});
}

} // namespace pyarb

