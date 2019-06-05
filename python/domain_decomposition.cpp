#include <string>
#include <sstream>

#include <pybind11/pybind11.h>

#include <arbor/context.hpp>
#include <arbor/domain_decomposition.hpp>
#include <arbor/load_balance.hpp>

#include "context.hpp"
#include "recipe.hpp"
//#include "strprintf.hpp"

namespace pyarb {

//std::ostream& operator<<(std::ostream& o, const arb::group_description& g) {
//    o << "<cell group: " << ncells
//      << " cells of " << g.kind
//      << " on " << g.backend;
//    if (ncells == 1) {
//        o << ", gid " << g.gids[0];
//    }
//    else if (ncells < 5) {
//        o << ", gids [" << util::csv(o, gids) << "]";
//    }
//    else {
//        o << ", gids [" << util::csv(o, gids, 3) << "]";
//    }
//}

//std::string group_description_string(const arb::group_description& g) {
//    std::stringstream s;
//    const auto ncells = g.gids.size();
//    s << "<cell group: " << ncells << " cells of " << g.kind << " on " << g.backend;
//    if (ncells == 1) {
//        s << " gid " << g.gids[0];
//    }
//
//    else if (ncells < 5) {
//        s << ", gids {";
//        bool first = true;
//        for (auto i: g.gids) {
//            if(!first) {
//                s << " ";
//            }
//            s << i;
//            first = false;
//        }
//        s << "}";
//    }
//    else {
//        s << ", gids {";
//        s << g.gids[0] << " " << g.gids[1] << " " << g.gids[2] << " ... " << g.gids.back();
//        s << "}";
//    }
//
//    s << ">";
//
//    return s.str();
//}

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
            "The gids of the cells in the group in ascending order.")
        .def_readonly("backend", &arb::group_description::backend,
            "The hardware backend on which the cell group will run.");
//        .def("__str__",  util::to_string<arb::group_description>)
//        .def("__repr__", util::to_string<arb::group_description>);

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
        .def("__str__", [](arb::domain_decomposition&){return "<arbor.domain_decomposition>";})
        .def("__repr__", [](arb::domain_decomposition&){return "<arbor.domain_decomposition>";});

    // Partition load balancer
    // The Python recipe has to be shimmed for passing to the function that
    // takes a C++ recipe.
    m.def("partition_load_balance",
        [](std::shared_ptr<py_recipe>& recipe, const context_shim& ctx) {
            return arb::partition_load_balance(py_recipe_shim(recipe), ctx.context);
        },
        "Construct a domain_decomposition that distributes the cells in the model described by recipe\n"
        "over the distributed and local hardware resources described by context.",
        "recipe"_a, "context"_a);
}

} // namespace pyarb

