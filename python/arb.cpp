#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <profiling/meter_manager.hpp>
#include <recipe.hpp>

#include <vector>

namespace pb = pybind11;

namespace arb {
class py_recipe: public arb::recipe {
public:
    using recipe::recipe;

    cell_size_type num_cells() const override {
        PYBIND11_OVERLOAD_PURE(cell_size_type, recipe, num_cells);
    }

    // Cell description type will be specific to cell kind of cell with given gid.
    util::unique_any get_cell_description(cell_gid_type gid) const override {
        PYBIND11_OVERLOAD_PURE(util::unique_any, recipe, get_cell_description);
    }

    cell_kind get_cell_kind(cell_gid_type) const override {
        PYBIND11_OVERLOAD_PURE(cell_kind, recipe, get_cell_kind);
    }

    //cell_size_type num_sources(cell_gid_type) const {return 0;}
    //cell_size_type num_targets(cell_gid_type) const {return 0;}
    //cell_size_type num_probes(cell_gid_type) const {return 0;}

    //std::vector<event_generator_ptr> event_generators(cell_gid_type) const {return {};};

    //std::vector<cell_connection> connections_on(cell_gid_type) const {return {};};
    //probe_info get_probe(cell_member_type probe_id) const {return {};};

    // Global property type will be specific to given cell kind.
    //util::any get_global_properties(cell_kind) const { return util::any{}; };
};

} // namespace arb

arb::util::any foo(int val) {
    return arb::util::any(val);
}

PYBIND11_MODULE(arb, m) {
    m.def("foo", &foo);

    //
    // util types
    //

    //
    // recipes
    //
    pb::class_<arb::recipe, arb::py_recipe> recipe(m, "recipe");
    recipe.def(pb::init<>())
          .def("num_cells", &arb::recipe::num_cells)
          .def("get_cell_description", &arb::recipe::get_cell_description)
          .def("get_cell_kind", &arb::recipe::get_cell_kind);

    //
    // load balancing and domain decomposition
    //

    //
    // models
    //

    //
    // metering
    //
    pb::class_<arb::util::measurement> measurement(m, "measurement",
             "Describes the recording of a single statistic over the course of a simulation,\ngathered by the meter_manager.");
    measurement.def_readwrite("name", &arb::util::measurement::name,
                    "Descriptive label of the measurement, e.g. 'wall time' or 'memory'.")
               .def_readwrite("units", &arb::util::measurement::units,
                    "SI units of the measurement, e.g. s or MiB.")
               .def_readwrite("measurements", &arb::util::measurement::measurements,
                    "A list of measurements, with one entry for each checkpoint.\n"
                    "Each entry is a list of values, with one value for each domain (MPI rank).");

    pb::class_<arb::util::meter_manager> meter_manager(m, "meter_manager");
    meter_manager.def(pb::init<>())
                     .def("start", &arb::util::meter_manager::start)
                     .def("checkpoint", &arb::util::meter_manager::checkpoint);

    // wrap meter_report type such that print(meter_report) works
    pb::class_<arb::util::meter_report> meter_report(m, "meter_report");
    meter_report.def("__str__",
            [] (const arb::util::meter_report& r) {
                std::stringstream s;
                s << r;
                return s.str();
            });

    m.def("make_meter_report", &arb::util::make_meter_report,
          "Generate a meter_report from a set of meters.");
}

