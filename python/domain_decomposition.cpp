#include <string>
#include <sstream>

#include <arbor/context.hpp>
#include <arbor/domain_decomposition.hpp>
#include <arbor/load_balance.hpp>

#include "context.hpp"
#include "recipe.hpp"

#include <pybind11/pybind11.h>

namespace pyarb {

std::string group_description_string(const arb::group_description& g) {
    std::stringstream s;
    const auto ncells = g.gids.size();
    s << "<cell group: "
      << ncells << " " << g.kind
      << " on " << g.backend;
    if (ncells==1) {
        s << " gid " << g.gids[0];
    }

    else if (ncells<5) {
        s << ", gids {";
        for (auto i: g.gids) {
            s << i << " ";
        }
        s << "}";
    }
    else {
        s << ", gids {";
        s << g.gids[0] << " " << g.gids[1] << " " << g.gids[2] << " ... " << g.gids.back();
        s << "}";
    }

    s << ">";

    return s.str();
}

void register_domain_decomposition(pybind11::module& m) {
    using namespace pybind11::literals;

    // Group description
    pybind11::class_<arb::group_description> group_description(m, "group_description",
        "The indexes of a set of cells of the same kind that are grouped together in a cell group.");
    group_description
        .def(pybind11::init<arb::cell_kind, std::vector<arb::cell_gid_type>, arb::backend_kind>(),
            "Construct a group description with cell kind, list of gids, and backend kind.")
        .def_readonly("kind", &arb::group_description::kind,
            "The type of cell in the cell group.")
        .def_readonly("gids", &arb::group_description::gids,
            "The gids of the cells in the group in ascending order.")
        .def_readonly("backend", &arb::group_description::backend,
            "The hardware backend on which the cell group will run.")
        .def("__str__",  &group_description_string)
        .def("__repr__", &group_description_string);

    // Domain decomposition
    pybind11::class_<arb::domain_decomposition> domain_decomposition(m, "domain_decomposition");
    domain_decomposition
        .def(pybind11::init<>())
        .def_readwrite("num_domains", &arb::domain_decomposition::num_domains,
            "Number of distrubuted domains.")
        .def_readwrite("domain_id", &arb::domain_decomposition::domain_id,
            "The index of the local domain.")
        .def_readwrite("num_local_cells", &arb::domain_decomposition::num_local_cells,
            "Total number of cells in the local domain.")
        .def_readwrite("num_global_cells", &arb::domain_decomposition::num_global_cells,
            "Total number of cells in the global model (sum over all domains).")
        .def("gid_domain",
            [](const arb::domain_decomposition& d, arb::cell_gid_type gid) {
                return d.gid_domain(gid);
            }, "The domain of cell with global identifier gid.", "gid"_a)
        .def_readwrite("groups", &arb::domain_decomposition::groups,
            "Descriptions of the cell groups on the local domain.");

    // Partition load balancer
    // The Python recipe has to be shimmed for passing to the function that
    // takes a C++ recipe.
    m.def("partition_load_balance",
        [](std::shared_ptr<py_recipe>& r, const context_shim& ctx) {
            return arb::partition_load_balance(py_recipe_shim(r), ctx.context);
        },
        "Simple load balancer.",
        "recipe"_a, "context"_a);
}

} // namespace pyarb

