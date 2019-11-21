#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <arbor/benchmark_cell.hpp>
#include <arbor/cable_cell.hpp>
#include <arbor/lif_cell.hpp>
#include <arbor/morph/label_dict.hpp>
#include <arbor/schedule.hpp>
#include <arbor/spike_source_cell.hpp>
#include <arbor/util/unique_any.hpp>

#include "conversion.hpp"
#include "error.hpp"
#include "schedule.hpp"
#include "strprintf.hpp"

namespace pyarb {

// Convert a cell description inside a Python object to a cell
// description in a unique_any, as required by the recipe interface.
//
// Warning: requires that the GIL has been acquired before calling,
// if there is a segmentation fault in the cast or isinstance calls,
// check that the caller has the GIL.
arb::util::unique_any convert_cell(pybind11::object o) {
    using pybind11::isinstance;
    using pybind11::cast;

    pybind11::gil_scoped_acquire guard;
    if (isinstance<arb::spike_source_cell>(o)) {
        return arb::util::unique_any(cast<arb::spike_source_cell>(o));
    }
    if (isinstance<arb::benchmark_cell>(o)) {
        return arb::util::unique_any(cast<arb::benchmark_cell>(o));
    }
    if (isinstance<arb::lif_cell>(o)) {
        return arb::util::unique_any(cast<arb::lif_cell>(o));
    }
    if (isinstance<arb::cable_cell>(o)) {
        return arb::util::unique_any(cast<arb::cable_cell>(o));
    }

    throw pyarb_error("recipe.cell_description returned \""
                      + std::string(pybind11::str(o))
                      + "\" which does not describe a known Arbor cell type");
}

//
//  proxies
//

struct local_param_set_proxy {
    arb::cable_cell_local_parameter_set params;
    void set_membrane_potential(pybind11::object value) {
        params.init_membrane_potential =
            py2optional<double>(value, "membrane potential must be a number.", is_nonneg{});
    }
    void set_temperature(pybind11::object value) {
        params.temperature_K =
            py2optional<double>(value, "temperature in degrees K must be non-negative.", is_nonneg{});
    }
    void set_axial_resistivity(pybind11::object value) {
        params.axial_resistivity =
            py2optional<double>(value, "axial resistivity must be positive.", is_positive{});
    }
    void set_membrane_capacitance(pybind11::object value) {
        params.membrane_capacitance =
            py2optional<double>(value, "membrane capacitance must be positive.", is_positive{});
    }

    auto get_membrane_potential()   const { return params.init_membrane_potential; }
    auto get_temperature()          const { return params.temperature_K; }
    auto get_axial_resistivity()    const { return params.axial_resistivity; }
    auto get_membrane_capacitance() const { return params.axial_resistivity; }

    operator arb::cable_cell_local_parameter_set() const {
        return params;
    }
};

/*
struct param_set_proxy {
    arb::cable_cell_local_parameter_set params;
    void set_membrane_potential(pybind11::object value) {
        params.init_membrane_potential =
            py2optional<double>(value, "membrane potential must be a number.", is_nonneg{});
    }
    void set_temperature(pybind11::object value) {
        params.temperature_K =
            py2optional<double>(value, "temperature in degrees K must be non-negative.", is_nonneg{});
    }
    void set_axial_resistivity(pybind11::object value) {
        params.axial_resistivity =
            py2optional<double>(value, "axial resistivity must be positive.", is_positive{});
    }
    void set_membrane_capacitance(pybind11::object value) {
        params.membrane_capacitance =
            py2optional<double>(value, "membrane capacitance must be positive.", is_positive{});
    }

    auto get_membrane_potential()   const { return params.init_membrane_potential; }
    auto get_temperature()          const { return params.temperature_K; }
    auto get_axial_resistivity()    const { return params.axial_resistivity; }
    auto get_membrane_capacitance() const { return params.axial_resistivity; }

    operator arb::cable_cell_local_parameter_set() const {
        return params;
    }
};
*/

//
// string printers
//

std::string lif_str(const arb::lif_cell& c){
    return util::pprintf(
        "<arbor.lif_cell: tau_m {}, V_th {}, C_m {}, E_L {}, V_m {}, t_ref {}, V_reset {}>",
        c.tau_m, c.V_th, c.C_m, c.E_L, c.V_m, c.t_ref, c.V_reset);
}


std::string mechanism_desc_str(const arb::mechanism_desc& md) {
    return util::pprintf("<arbor.mechanism_desc: name '{}', parameters {}",
            md.name(), util::dictionary_csv(md.values()));
}

std::ostream& operator<<(std::ostream& o, const arb::cable_cell_ion_data& d) {
    return o << util::pprintf("(con_in {}, con_ex {}, rev_pot {})",
            d.init_int_concentration, d.init_ext_concentration, d.init_reversal_potential);
}

std::string ion_data_str(const arb::cable_cell_ion_data& d) {
    return util::pprintf(
        "<arbor.cable_cell_ion_data: con_in {}, con_ex {}, rev_pot {}>",
        d.init_int_concentration, d.init_ext_concentration, d.init_reversal_potential);
}

std::string local_parameter_set_str(const local_param_set_proxy& p) {
    auto s = util::pprintf("<arbor.local_parameter_set: V_m {} (mV), temp {} (K), R_L {} (Ω·cm), C_m {} (F/m²), ion_data {}>",
            p.params.init_membrane_potential, p.params.temperature_K,
            p.params.axial_resistivity, p.params.membrane_capacitance,
            util::dictionary_csv(p.params.ion_data));
    return s;
}

void register_cells(pybind11::module& m) {
    using namespace pybind11::literals;

    // arb::spike_source_cell

    pybind11::class_<arb::spike_source_cell> spike_source_cell(m, "spike_source_cell",
        "A spike source cell, that generates a user-defined sequence of spikes that act as inputs for other cells in the network.");

    spike_source_cell
        .def(pybind11::init<>(
            [](const regular_schedule_shim& sched){
                return arb::spike_source_cell{sched.schedule()};}),
            "schedule"_a, "Construct a spike source cell that generates spikes at regular intervals.")
        .def(pybind11::init<>(
            [](const explicit_schedule_shim& sched){
                return arb::spike_source_cell{sched.schedule()};}),
            "schedule"_a, "Construct a spike source cell that generates spikes at a sequence of user-defined times.")
        .def(pybind11::init<>(
            [](const poisson_schedule_shim& sched){
                return arb::spike_source_cell{sched.schedule()};}),
            "schedule"_a, "Construct a spike source cell that generates spikes at times defined by a Poisson sequence.")
        .def("__repr__", [](const arb::spike_source_cell&){return "<arbor.spike_source_cell>";})
        .def("__str__",  [](const arb::spike_source_cell&){return "<arbor.spike_source_cell>";});

    // arb::benchmark_cell

    pybind11::class_<arb::benchmark_cell> benchmark_cell(m, "benchmark_cell",
        "A benchmarking cell, used by Arbor developers to test communication performance.\n"
        "A benchmark cell generates spikes at a user-defined sequence of time points, and\n"
        "the time taken to integrate a cell can be tuned by setting the realtime_ratio,\n"
        "for example if realtime_ratio=2, a cell will take 2 seconds of CPU time to\n"
        "simulate 1 second.\n");

    benchmark_cell
        .def(pybind11::init<>(
            [](const regular_schedule_shim& sched, double ratio){
                return arb::benchmark_cell{sched.schedule(), ratio};}),
            "schedule"_a, "realtime_ratio"_a=1.0,
            "Construct a benchmark cell that generates spikes at regular intervals.")
        .def(pybind11::init<>(
            [](const explicit_schedule_shim& sched, double ratio){
                return arb::benchmark_cell{sched.schedule(), ratio};}),
            "schedule"_a, "realtime_ratio"_a=1.0,
            "Construct a benchmark cell that generates spikes at a sequence of user-defined times.")
        .def(pybind11::init<>(
            [](const poisson_schedule_shim& sched, double ratio){
                return arb::benchmark_cell{sched.schedule(), ratio};}),
            "schedule"_a, "realtime_ratio"_a=1.0,
            "Construct a benchmark cell that generates spikes at times defined by a Poisson sequence.")
        .def("__repr__", [](const arb::benchmark_cell&){return "<arbor.benchmark_cell>";})
        .def("__str__",  [](const arb::benchmark_cell&){return "<arbor.benchmark_cell>";});

    // arb::lif_cell

    pybind11::class_<arb::lif_cell> lif_cell(m, "lif_cell",
        "A benchmarking cell, used by Arbor developers to test communication performance.");

    lif_cell
        .def(pybind11::init<>())
        .def_readwrite("tau_m",     &arb::lif_cell::tau_m,      "Membrane potential decaying constant [ms].")
        .def_readwrite("V_th",      &arb::lif_cell::V_th,       "Firing threshold [mV].")
        .def_readwrite("C_m",       &arb::lif_cell::C_m,        "Membrane capacitance [pF].")
        .def_readwrite("E_L",       &arb::lif_cell::E_L,        "Resting potential [mV].")
        .def_readwrite("V_m",       &arb::lif_cell::V_m,        "Initial value of the Membrane potential [mV].")
        .def_readwrite("t_ref",     &arb::lif_cell::t_ref,      "Refractory period [ms].")
        .def_readwrite("V_reset",   &arb::lif_cell::V_reset,    "Reset potential [mV].")
        .def("__repr__", &lif_str)
        .def("__str__",  &lif_str);

    //
    // regions and locsets
    //

    // arb::locset

    pybind11::class_<arb::locset> locset(m, "locset");
    locset
        .def(pybind11::init<>(),
            "Default constructor creates empty/null locset")
        .def("__str__", [](const arb::locset& ls) {return util::pprintf("<arbor.locset: {}>", ls);})
        .def("__repr__", [](const arb::locset& ls) {return util::pprintf("{}", ls);});

    m.def("ls_location",  &arb::ls::location, "location"_a, "An explicit location on a morphology.");
    m.def("ls_sample",    &arb::ls::sample,   "sample_id"_a, "The location of a specific sample on the morphology.");
    m.def("ls_terminal",  &arb::ls::terminal, "The terminal points on a morphology.");
    m.def("ls_root",      &arb::ls::root,     "The root of a morphology, equivalent to location(0,0).");
    m.def("ls_nil",       &arb::ls::nil,      "Empty set of locations.");
    // expose 2, 3, 4 and 5 way joins
    m.def("join", [](arb::locset l, arb::locset r){return join(l,r);});
    m.def("join", [](arb::locset r0, arb::locset r1, arb::locset r2){return join(r0, r1, r2);});
    m.def("join", [](arb::locset r0, arb::locset r1, arb::locset r2, arb::locset r3){return join(r0, r1, r2, r3);});
    m.def("join", [](arb::locset r0, arb::locset r1, arb::locset r2, arb::locset r3, arb::locset r4){return join(r0, r1, r2, r3, r4);});
    // expose 2, 3, 4 and 5 way sums
    m.def("sum", [](arb::locset l, arb::locset r){return sum(l,r);});
    m.def("sum", [](arb::locset r0, arb::locset r1, arb::locset r2){return sum(r0, r1, r2);});
    m.def("sum", [](arb::locset r0, arb::locset r1, arb::locset r2, arb::locset r3){return sum(r0, r1, r2, r3);});
    m.def("sum", [](arb::locset r0, arb::locset r1, arb::locset r2, arb::locset r3, arb::locset r4){return sum(r0, r1, r2, r3, r4);});

    // arb::region

    pybind11::class_<arb::region> region(m, "region");
    region
        .def(pybind11::init<>(),
            "Default constructor creates empty/null region")
        .def("__str__", [](const arb::region& r) {return util::pprintf("<arbor.region: {}>", r);})
        .def("__repr__", [](const arb::region& r) {return util::pprintf("{}", r);});

    m.def("reg_all",    &arb::reg::all,     "Region describing the entire morphology.");
    m.def("reg_tag",    &arb::reg::tagged,  "Part of morphology that was tagged with tag.", "tag"_a);
    m.def("reg_branch", &arb::reg::branch,  "The branch on a morphology with id.", "id"_a);
    m.def("reg_cable",  &arb::reg::cable,   "An explicit cable section on a morphology.", "cable"_a);
    m.def("reg_nil",    &arb::reg::nil,     "Empty region.");
    // expose 2, 3, 4 and 5 way joins
    m.def("join", [](arb::region l, arb::region r){return join(l,r);});
    m.def("join", [](arb::region r0, arb::region r1, arb::region r2){return join(r0, r1, r2);});
    m.def("join", [](arb::region r0, arb::region r1, arb::region r2, arb::region r3){return join(r0, r1, r2, r3);});
    m.def("join", [](arb::region r0, arb::region r1, arb::region r2, arb::region r3, arb::region r4){return join(r0, r1, r2, r3, r4);});
    // expose 2, 3, 4 and 5 way intersections
    m.def("intersect", [](arb::region l, arb::region r){return intersect(l,r);});
    m.def("intersect", [](arb::region r0, arb::region r1, arb::region r2){return intersect(r0, r1, r2);});
    m.def("intersect", [](arb::region r0, arb::region r1, arb::region r2, arb::region r3){return intersect(r0, r1, r2, r3);});
    m.def("intersect", [](arb::region r0, arb::region r1, arb::region r2, arb::region r3, arb::region r4){return intersect(r0, r1, r2, r3, r4);});

    // arb::label_dict

    pybind11::class_<arb::label_dict> label_dict(m, "label_dict");
    label_dict
        .def(pybind11::init<>())
        .def("set",
            [](arb::label_dict& l, const std::string& name, arb::locset ls) {
                l.set(name, std::move(ls));})
        .def("set",
            [](arb::label_dict& l, const std::string& name, arb::region reg) {
                l.set(name, std::move(reg));})
        .def("size", &arb::label_dict::size,
             "Rhe number of labels in the dictionary.")
        .def("regions", &arb::label_dict::regions,
             "Returns a dictionary mapping region names to their definitions.")
        .def("locsets", &arb::label_dict::locsets,
             "Returns a dictionary mapping locset names to their definitions.")
        .def("region",
             [](const arb::label_dict& d, const std::string& n) {
                auto reg = d.region(n);
                return reg? *reg: arb::reg::nil();
             },
             "name"_a, "Returns the region with label name. Returns the empty region nil if there is no region with that label.")
        .def("locset",
             [](const arb::label_dict& d, const std::string& n) {
                auto ls = d.locset(n);
                return ls? *ls: arb::ls::nil();
             },
             "name"_a, "Returns the locset with label name. Returns the empty locset nil if there is no locset with that label.");

    //
    // Data structures used to describe mechanisms, electrical properties,
    // gap junction properties, etc.
    //

    // arb::mechanism_desc
    pybind11::class_<arb::mechanism_desc> mechanism_desc(m, "mechanism_desc");
    mechanism_desc
        .def(pybind11::init<>())
        .def(pybind11::init([](const char* n) {return arb::mechanism_desc{n};}))
        // allow construction of a description with parameters provided in a dictionary:
        //      mech = arbor.mechanism_desc('mech_name', {'param1': 1.2, 'param2': 3.14})
        .def(pybind11::init(
            [](const char* name, std::unordered_map<std::string, double> params) {
                arb::mechanism_desc md(name);
                for (const auto& p: params) md.set(p.first, p.second);
                return md;
            }))
        .def("set",
            [](arb::mechanism_desc& md, std::string key, double value) {
                md.set(key, value);
            },
            "key"_a, "value"_a)
        .def_property_readonly("name", [](const arb::mechanism_desc& md) {return md.name();})
        .def_property_readonly("values", [](const arb::mechanism_desc& md) {return md.values();})
        .def("__repr__", &mechanism_desc_str)
        .def("__str__",  &mechanism_desc_str);

    // arb::cable_cell_ion_data
    pybind11::class_<arb::cable_cell_ion_data> ion_data(m, "ion_data");
    ion_data
        .def(pybind11::init(
            [](double ic, double ec, double rp) {
                return arb::cable_cell_ion_data{ic, ec, rp};
            }), "intern_con"_a, "extern_con"_a, "rev_pot"_a)
        .def_readonly("intern_con",
            &arb::cable_cell_ion_data::init_int_concentration,
            "Initial internal concentration of ion species.")
        .def_readonly("extern_con",
            &arb::cable_cell_ion_data::init_ext_concentration,
            "Initial external concentration of ion species.")
        .def_readonly("rev_pot",
            &arb::cable_cell_ion_data::init_reversal_potential,
            "Initial reversal potential of ion species.")
        .def("__repr__", &ion_data_str)
        .def("__str__",  &ion_data_str);

    // arb::cable_cell_local_parameter_set
    pybind11::class_<local_param_set_proxy> local_cable_params(m, "local_parameter_set");
    local_cable_params
        .def(pybind11::init<>())
        .def_property("temperature_K",
            &local_param_set_proxy::get_temperature,
            &local_param_set_proxy::set_temperature,
            "Temperature in degrees Kelvin.")
        .def_property("axial_resistivity",
            &local_param_set_proxy::get_axial_resistivity,
            &local_param_set_proxy::set_axial_resistivity,
            "Axial resistivity in Ω·cm.")
        .def_property("init_membrane_potential",
            &local_param_set_proxy::get_membrane_potential,
            &local_param_set_proxy::set_membrane_potential,
            "Initial membrane potential in mV.")
        .def_property("membrane_capacitance",
            &local_param_set_proxy::get_membrane_capacitance,
            &local_param_set_proxy::set_membrane_capacitance,
            "Membrane capacitance in F/m².")
        .def_property_readonly("ion_data",
                [](const local_param_set_proxy& p) {
                    return p.params.ion_data;
                })
        .def("set_ion",
                [](local_param_set_proxy& p, std::string ion, arb::cable_cell_ion_data data) {
                    p.params.ion_data[std::move(ion)] = data;
                },
            "name"_a, "props"_a,
            "Set properties of an ion species with name.")
        .def("__repr__", &local_parameter_set_str)
        .def("__str__",  &local_parameter_set_str);

    //pybind11::class_<local_param_set_proxy> cable_params(m, "local_parameter_set");

    // arb::cable_cell
    pybind11::class_<arb::cable_cell> cable_cell(m, "cable_cell");
    cable_cell
        .def(pybind11::init(
            [](const arb::morphology& m, const arb::label_dict& labels, bool cfd) {
                return arb::cable_cell(m, labels, cfd);
            }), "morphology"_a, "labels"_a, "compartments_from_discretization"_a=true)
        .def_property_readonly("num_branches", [](const arb::morphology& m) {return m.num_branches();})
        .def("paint",
            [](arb::cable_cell& c, const char* region, const arb::mechanism_desc& d) {
                c.paint(region, d);
            },
            "region"_a, "mechanism"_a,
            "Associate a mechanism with a region.")
        .def("paint",
            [](arb::cable_cell& c, const char* region, const char* mech_name) {
                c.paint(region, mech_name);
            },
            "region"_a, "mechanism_name"_a,
            "Associate a mechanism with a region.")
        .def("paint",
            [](arb::cable_cell& c, const char* region, const local_param_set_proxy& p) {
                c.paint(region, (arb::cable_cell_local_parameter_set)p);
            },
            "region"_a, "mechanism"_a,
            "Associate a set of properties with a region. These properties will override the the global or cell-wide default values on the specific region.")
        .def("__repr__", [](const arb::cable_cell&){return "<arbor.cable_cell>";})
        .def("__str__",  [](const arb::cable_cell&){return "<arbor.cable_cell>";});
}

} // namespace pyarb
