#include <algorithm>
#include <any>
#include <cstddef>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <arborio/cv_policy_parse.hpp>
#include <arborio/label_parse.hpp>

#include <arbor/benchmark_cell.hpp>
#include <arbor/cable_cell.hpp>
#include <arbor/lif_cell.hpp>
#include <arbor/cv_policy.hpp>
#include <arbor/morph/cv_data.hpp>
#include <arbor/morph/label_dict.hpp>
#include <arbor/morph/locset.hpp>
#include <arbor/morph/region.hpp>
#include <arbor/morph/segment_tree.hpp>
#include <arbor/schedule.hpp>
#include <arbor/spike_source_cell.hpp>
#include <arbor/util/any_cast.hpp>
#include <arbor/util/unique_any.hpp>

#include "arbor/cable_cell_param.hpp"
#include "arbor/cv_policy.hpp"
#include "conversion.hpp"
#include "error.hpp"
#include "proxy.hpp"
#include "pybind11/cast.h"
#include "pybind11/stl.h"
#include "pybind11/pytypes.h"
#include "schedule.hpp"
#include "strprintf.hpp"

namespace pyarb {

template <typename T>
std::string to_string(const T& t) {
    std::stringstream ss;
    ss << t;
    return ss.str();
}

// This isn't pretty. Partly because the information in the global parameters
// is all over the place.
template <>
std::string to_string(const arb::cable_cell_global_properties& props) {
    std::string s = "{arbor.cable_global_properties";

    const auto& D = props.default_parameters;
    const auto& I = D.ion_data;
    // name, valence, int_con, ext_con, rev_pot, rev_pot_method
    s += "\n  ions: {";
    for (auto& ion: props.ion_species) {
        if (!I.count(ion.first)) {
            s += util::pprintf("\n    {name: '{}', valence: {}, int_con: None, ext_con: None, rev_pot: None, rev_pot_method: None}",
                    ion.first, ion.second);
        }
        else {
            auto& props = I.at(ion.first);
            std::string method = D.reversal_potential_method.count(ion.first)?
                "'"+D.reversal_potential_method.at(ion.first).name()+"'": "None";
            s += util::pprintf("\n    {name: '{}', valence: {}, int_con: {}, ext_con: {}, rev_pot: {}, rev_pot_method: {}}",
                    ion.first, ion.second,
                    props.init_int_concentration,
                    props.init_ext_concentration,
                    props.init_reversal_potential,
                    method);
        }
    }
    s += "}\n";
    s += util::pprintf("  parameters: {Vm: {}, cm: {}, rL: {}, tempK: {}}\n",
            D.init_membrane_potential, D.membrane_capacitance,
            D.axial_resistivity, D.temperature_K);
    s += "}";
    return s;
}

//
// cv_policy helpers
//

arb::cv_policy make_cv_policy_single(const std::string& reg) {
    return arb::cv_policy_single(arborio::parse_region_expression(reg).unwrap());
}

arb::cv_policy make_cv_policy_explicit(const std::string& locset, const std::string& reg) {
    return arb::cv_policy_explicit(arborio::parse_locset_expression(locset).unwrap(), arborio::parse_region_expression(reg).unwrap());
}

arb::cv_policy make_cv_policy_every_segment(const std::string& reg) {
    return arb::cv_policy_every_segment(arborio::parse_region_expression(reg).unwrap());
}

arb::cv_policy make_cv_policy_fixed_per_branch(unsigned cv_per_branch, const std::string& reg) {
    return arb::cv_policy_fixed_per_branch(cv_per_branch, arborio::parse_region_expression(reg).unwrap());
}

arb::cv_policy make_cv_policy_max_extent(double cv_length, const std::string& reg) {
    return arb::cv_policy_max_extent(cv_length, arborio::parse_region_expression(reg).unwrap());
}

// Helper for finding a mechanism description in a Python object.
// Allows rev_pot_method to be specified with string or mechanism_desc
std::optional<arb::mechanism_desc> maybe_method(pybind11::object method) {
    if (!method.is_none()) {
        if (auto m=try_cast<std::string>(method)) {
            return *m;
        }
        else if (auto m=try_cast<arb::mechanism_desc>(method)) {
            return *m;
        }
        else {
            throw std::runtime_error(util::pprintf("invalid rev_pot_method: {}", method));
        }
    }
    return {};
}

//
// string printers
//

std::string lif_str(const arb::lif_cell& c){
    return util::pprintf(
        "<arbor.lif_cell: tau_m {}, V_th {}, C_m {}, E_L {}, V_m {}, t_ref {}>",
        c.tau_m, c.V_th, c.C_m, c.E_L, c.V_m, c.t_ref);
}


std::string mechanism_desc_str(const arb::mechanism_desc& md) {
    return util::pprintf("mechanism('{}', {})",
            md.name(), util::dictionary_csv(md.values()));
}

std::string scaled_density_desc_str(const arb::scaled_mechanism<arb::density>& p) {
    return util::pprintf("({}, {})",
            mechanism_desc_str(p.t_mech.mech), util::dictionary_csv(p.scale_expr));
}

void register_cells(pybind11::module& m) {
    using namespace pybind11::literals;
    using std::optional;

    // arb::spike_source_cell

    pybind11::class_<arb::spike_source_cell> spike_source_cell(m, "spike_source_cell",
        "A spike source cell, that generates a user-defined sequence of spikes that act as inputs for other cells in the network.");

    spike_source_cell
        .def(pybind11::init<>(
            [](arb::cell_tag_type source_label, const regular_schedule_shim& sched){
                return arb::spike_source_cell{std::move(source_label), sched.schedule()};}),
            "source_label"_a, "schedule"_a,
            "Construct a spike source cell with a single source labeled 'source_label'.\n"
            "The cell generates spikes on 'source_label' at regular intervals.")
        .def(pybind11::init<>(
            [](arb::cell_tag_type source_label, const explicit_schedule_shim& sched){
                return arb::spike_source_cell{std::move(source_label), sched.schedule()};}),
            "source_label"_a, "schedule"_a,
            "Construct a spike source cell with a single source labeled 'source_label'.\n"
            "The cell generates spikes on 'source_label' at a sequence of user-defined times.")
        .def(pybind11::init<>(
            [](arb::cell_tag_type source_label, const poisson_schedule_shim& sched){
                return arb::spike_source_cell{std::move(source_label), sched.schedule()};}),
            "source_label"_a, "schedule"_a,
            "Construct a spike source cell with a single source labeled 'source_label'.\n"
            "The cell generates spikes on 'source_label' at times defined by a Poisson sequence.")
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
            [](arb::cell_tag_type source_label, arb::cell_tag_type target_label, const regular_schedule_shim& sched, double ratio){
                return arb::benchmark_cell{std::move(source_label), std::move(target_label), sched.schedule(), ratio};}),
            "source_label"_a, "target_label"_a,"schedule"_a, "realtime_ratio"_a=1.0,
            "Construct a benchmark cell that generates spikes on 'source_label' at regular intervals.\n"
            "The cell has one source labeled 'source_label', and one target labeled 'target_label'.")
        .def(pybind11::init<>(
            [](arb::cell_tag_type source_label, arb::cell_tag_type target_label, const explicit_schedule_shim& sched, double ratio){
                return arb::benchmark_cell{std::move(source_label), std::move(target_label),sched.schedule(), ratio};}),
            "source_label"_a, "target_label"_a, "schedule"_a, "realtime_ratio"_a=1.0,
            "Construct a benchmark cell that generates spikes on 'source_label' at a sequence of user-defined times.\n"
            "The cell has one source labeled 'source_label', and one target labeled 'target_label'.")
        .def(pybind11::init<>(
            [](arb::cell_tag_type source_label, arb::cell_tag_type target_label, const poisson_schedule_shim& sched, double ratio){
                return arb::benchmark_cell{std::move(source_label), std::move(target_label), sched.schedule(), ratio};}),
            "source_label"_a, "target_label"_a, "schedule"_a, "realtime_ratio"_a=1.0,
            "Construct a benchmark cell that generates spikeson 'source_label' at times defined by a Poisson sequence.\n"
            "The cell has one source labeled 'source_label', and one target labeled 'target_label'.")
        .def("__repr__", [](const arb::benchmark_cell&){return "<arbor.benchmark_cell>";})
        .def("__str__",  [](const arb::benchmark_cell&){return "<arbor.benchmark_cell>";});

    // arb::lif_cell

    pybind11::class_<arb::lif_cell> lif_cell(m, "lif_cell",
        "A leaky integrate-and-fire cell.");

    lif_cell
        .def(pybind11::init<>(
            [](arb::cell_tag_type source_label, arb::cell_tag_type target_label){
                return arb::lif_cell(std::move(source_label), std::move(target_label));}),
            "source_label"_a, "target_label"_a,
            "Construct a lif cell with one source labeled 'source_label', and one target labeled 'target_label'.")
        .def_readwrite("tau_m", &arb::lif_cell::tau_m,
            "Membrane potential decaying constant [ms].")
        .def_readwrite("V_th", &arb::lif_cell::V_th,
            "Firing threshold [mV].")
        .def_readwrite("C_m", &arb::lif_cell::C_m,
            "Membrane capacitance [pF].")
        .def_readwrite("E_L", &arb::lif_cell::E_L,
            "Resting potential [mV].")
        .def_readwrite("E_R", &arb::lif_cell::E_R,
            "Reset potential [mV].")
        .def_readwrite("V_m", &arb::lif_cell::V_m,
            "Initial value of the Membrane potential [mV].")
        .def_readwrite("t_ref", &arb::lif_cell::t_ref,
            "Refractory period [ms].")
        .def_readwrite("source", &arb::lif_cell::source,
            "Label of the single build-in source on the cell.")
        .def_readwrite("target", &arb::lif_cell::target,
            "Label of the single build-in target on the cell.")
        .def("__repr__", &lif_str)
        .def("__str__",  &lif_str);

    // arb::label_dict

    pybind11::class_<label_dict_proxy> label_dict(m, "label_dict",
        "A dictionary of labelled region and locset definitions, with a\n"
        "unique label assigned to each definition.");
    label_dict
        .def(pybind11::init<>(),
             "Create an empty label dictionary.")
        .def(pybind11::init<const std::unordered_map<std::string, std::string>&>(),
            "Initialize a label dictionary from a dictionary with string labels as keys,"
            " and corresponding definitions as strings.")
        .def(pybind11::init<const label_dict_proxy&>(),
            "Initialize a label dictionary from another one")
        .def(pybind11::init([](pybind11::iterator& it) {
                label_dict_proxy ld;
                for (; it != pybind11::iterator::sentinel(); ++it) {
                    const auto tuple = it->cast<pybind11::sequence>();
                    const auto key   = tuple[0].cast<std::string>();
                    const auto value = tuple[1].cast<std::string>();
                    ld.set(key, value);
                }
                return ld;
            }),
            "Initialize a label dictionary from an iterable of key, definition pairs")
        .def("add_swc_tags",
             [](label_dict_proxy& l) { return l.add_swc_tags(); },
             "Add standard SWC tagged regions.\n"
             " - soma: (tag 1)\n"
             " - axon: (tag 2)\n"
             " - dend: (tag 3)\n"
             " - apic: (tag 4)")
        .def("__setitem__",
            [](label_dict_proxy& l, const char* name, const char* desc) {
                l.set(name, desc);})
        .def("__getitem__",
            [](label_dict_proxy& l, const char* name) {
                if (auto v = l.getitem(name)) return v.value();
                throw pybind11::key_error(name);
            })
        .def("__len__", &label_dict_proxy::size)
        .def("__iter__",
            [](const label_dict_proxy &ld) {
                return pybind11::make_key_iterator(ld.cache.begin(), ld.cache.end());},
            pybind11::keep_alive<0, 1>())
        .def("__contains__",
             [](const label_dict_proxy &ld, const char* name) {
                 return ld.contains(name);})
        .def("keys",
            [](const label_dict_proxy &ld) {
                return pybind11::make_key_iterator(ld.cache.begin(), ld.cache.end());},
            pybind11::keep_alive<0, 1>())
        .def("items",
             [](const label_dict_proxy &ld) {
                 return pybind11::make_iterator(ld.cache.begin(), ld.cache.end());},
             pybind11::keep_alive<0, 1>())
        .def("values",
             [](const label_dict_proxy &ld) {
                 return pybind11::make_value_iterator(ld.cache.begin(), ld.cache.end());
             },
             pybind11::keep_alive<0, 1>())
        .def("append", [](label_dict_proxy& l, const label_dict_proxy& other, const char* prefix) {
                l.import(other, prefix);
            },
            "other"_a, "The label_dict to be imported"
            "prefix"_a="", "optional prefix appended to the region and locset labels",
            "Import the entries of a another label dictionary with an optional prefix.")
        .def("update", [](label_dict_proxy& l, const label_dict_proxy& other) {
                l.import(other);
            },
            "other"_a, "The label_dict to be imported"
            "Import the entries of a another label dictionary.")
        .def_readonly("regions", &label_dict_proxy::regions,
             "The region definitions.")
        .def_readonly("locsets", &label_dict_proxy::locsets,
             "The locset definitions.")
        .def("__repr__", [](const label_dict_proxy& d){return d.to_string();})
        .def("__str__",  [](const label_dict_proxy& d){return d.to_string();});

    // arb::cv_policy wrappers

    pybind11::class_<arb::cv_policy> cv_policy(m, "cv_policy",
            "Describes the rules used to discretize (compartmentalise) a cable cell morphology.");
    cv_policy
        .def(pybind11::init([](const std::string& expression) { return arborio::parse_cv_policy_expression(expression).unwrap(); }),
            "expression"_a, "A valid CV policy expression")
        .def_property_readonly("domain",
                               [](const arb::cv_policy& p) {return util::pprintf("{}", p.domain());},
                               "The domain on which the policy is applied.")
        .def(pybind11::self + pybind11::self)
        .def(pybind11::self | pybind11::self)
        .def("__repr__", [](const arb::cv_policy& p) {
            std::stringstream ss;
            ss << p;
            return ss.str();
        })
        .def("__str__", [](const arb::cv_policy& p) {
            std::stringstream ss;
            ss << p;
            return ss.str();
        });

    m.def("cv_policy_explicit",
          &make_cv_policy_explicit,
          "locset"_a, "the locset describing the desired CV boundaries",
          "domain"_a="(all)", "the domain to which the policy is to be applied",
          "Policy to create compartments at explicit locations.");

    m.def("cv_policy_single",
          &make_cv_policy_single,
          "domain"_a="(all)", "the domain to which the policy is to be applied",
          "Policy to create one compartment per component of a region.");

    m.def("cv_policy_every_segment",
          &make_cv_policy_every_segment,
          "domain"_a="(all)", "the domain to which the policy is to be applied",
          "Policy to create one compartment per component of a region.");

    m.def("cv_policy_max_extent",
          &make_cv_policy_max_extent,
          "length"_a, "the maximum CV length",
          "domain"_a="(all)", "the domain to which the policy is to be applied",
          "Policy to use as many CVs as required to ensure that no CV has a length longer than a given value.");

    m.def("cv_policy_fixed_per_branch",
          &make_cv_policy_fixed_per_branch,
          "n"_a, "the number of CVs per branch",
          "domain"_a="(all)", "the domain to which the policy is to be applied",
          "Policy to use the same number of CVs for each branch.");

    // arb::cell_cv_data
    pybind11::class_<arb::cell_cv_data> cell_cv_data(m, "cell_cv_data",
            "Provides information on the CVs representing the discretization of a cable-cell.");
    cell_cv_data
            .def_property_readonly("num_cv", [](const arb::cell_cv_data& data){return data.size();},
                 "Return the number of CVs in the cell.")
            .def("cables",
                 [](const arb::cell_cv_data& d, unsigned index) {
                    if (index >= d.size()) throw pybind11::index_error("index out of range");
                    return d.cables(index);
                 },
                 "index"_a, "Return a list of cables representing the CV at the given index.")
            .def("children",
                 [](const arb::cell_cv_data& d, unsigned index) {
                     if (index >= d.size()) throw pybind11::index_error("index out of range");
                     return d.children(index);
                 },
                 "index"_a,
                 "Return a list of indices of the CVs representing the children of the CV at the given index.")
            .def("parent",
                 [](const arb::cell_cv_data& d, unsigned index) {
                     if (index >= d.size()) throw pybind11::index_error("index out of range");
                     return d.parent(index);
                 },
                 "index"_a,
                 "Return the index of the CV representing the parent of the CV at the given index.")
            .def("__str__",  [](const arb::cell_cv_data& p){return "<arbor.cell_cv_data>";})
            .def("__repr__", [](const arb::cell_cv_data& p){return "<arbor.cell_cv_data>";});

    m.def("cv_data", [](const arb::cable_cell& cell) { return arb::cv_data(cell);},
          "cell"_a, "the cable cell",
          "Returns a cell_cv_data object representing the CVs comprising the cable-cell according to the "
          "discretization policy provided in the decor of the cell. Returns None if no CV-policy was provided "
          "in the decor."
          );

    m.def("intersect_region",
          [](const char* reg, const arb::cell_cv_data& cvs, const std::string& integrate_along) {
              bool integrate_area;
              if (integrate_along == "area") integrate_area = true;
              else if (integrate_along == "length") integrate_area = false;
              else throw pyarb_error(util::pprintf("{} does not name a valid integration axis. "
                                                   "Only 'area' and 'length' are supported)", integrate_along));

              auto object_vec = arb::intersect_region(arborio::parse_region_expression(reg).unwrap(), cvs, integrate_area);
              auto tuple_vec = std::vector<pybind11::tuple>(object_vec.size());
              std::transform(object_vec.begin(), object_vec.end(), tuple_vec.begin(),
                             [](const auto& t)  { return pybind11::make_tuple(t.idx, t.proportion); });
              return tuple_vec;
          },
          "reg"_a,  "A region on a cell",
          "data"_a, "The data representing the CVs of the cell.",
          "integrate_along"_a, "the axis of integration used to determine the proportion of the CV belonging to the region.",
          "Returns a list of [index, proportion] tuples identifying the CVs present in the region.\n"
          "`index` is the index of the CV in the cell_cv_data object provided as an argument.\n"
          "`proportion` is the proportion of the CV (itegrated by area or length) included in the region."
    );

    pybind11::class_<arb::init_membrane_potential> membrane_potential(m, "membrane_potential", "Setting the initial membrane voltage.");
    membrane_potential
        .def(pybind11::init([](double v) -> arb::init_membrane_potential { return {v}; }))
        .def("__repr__", [](const arb::init_membrane_potential& d){return "Vm=" + std::to_string(d.value);});

    pybind11::class_<arb::membrane_capacitance> membrane_capacitance(m, "membrane_capacitance", "Setting the membrane capacitance.");
    membrane_capacitance
        .def(pybind11::init([](double v) -> arb::membrane_capacitance { return {v}; }))
        .def("__repr__", [](const arb::membrane_capacitance& d){return "Cm=" + std::to_string(d.value);});

    pybind11::class_<arb::temperature_K> temperature_K(m, "temperature_K", "Setting the temperature.");
    temperature_K
        .def(pybind11::init([](double v) -> arb::temperature_K { return {v}; }))
        .def("__repr__", [](const arb::temperature_K& d){return "T=" + std::to_string(d.value);});

    pybind11::class_<arb::axial_resistivity> axial_resistivity(m, "axial_resistivity", "Setting the axial resistivity.");
    axial_resistivity
        .def(pybind11::init([](double v) -> arb::axial_resistivity { return {v}; }))
        .def("__repr__", [](const arb::axial_resistivity& d){return "Ra" + std::to_string(d.value);});

    pybind11::class_<arb::init_reversal_potential> reversal_potential(m, "reversal_potential", "Setting the initial reversal potential.");
    reversal_potential
        .def(pybind11::init([](const std::string& i, double v) -> arb::init_reversal_potential { return {i, v}; }))
        .def("__repr__", [](const arb::init_reversal_potential& d){return "e" + d.ion + "=" + std::to_string(d.value);});

    pybind11::class_<arb::init_int_concentration> int_concentration(m, "int_concentration", "Setting the initial internal ion concentration.");
    int_concentration
        .def(pybind11::init([](const std::string& i, double v) -> arb::init_int_concentration { return {i, v}; }))
        .def("__repr__", [](const arb::init_int_concentration& d){return d.ion + "i" + "=" + std::to_string(d.value);});

    pybind11::class_<arb::init_ext_concentration> ext_concentration(m, "ext_concentration", "Setting the initial external ion concentration.");
    ext_concentration
        .def(pybind11::init([](const std::string& i, double v) -> arb::init_ext_concentration { return {i, v}; }))
        .def("__repr__", [](const arb::init_ext_concentration& d){return d.ion + "o" + "=" + std::to_string(d.value);});

    pybind11::class_<arb::ion_diffusivity> ion_diffusivity(m, "ion_diffusivity", "Setting the ion diffusivity.");
    ion_diffusivity
        .def(pybind11::init([](const std::string& i, double v) -> arb::ion_diffusivity { return {i, v}; }))
        .def("__repr__", [](const arb::ion_diffusivity& d){return "D" + d.ion + "=" + std::to_string(d.value);});

    pybind11::class_<arb::density> density(m, "density", "For painting a density mechanism on a region.");
    density
        .def(pybind11::init([](const std::string& name) {return arb::density(name);}))
        .def(pybind11::init([](arb::mechanism_desc mech) {return arb::density(mech);}))
        .def(pybind11::init([](const std::string& name, const std::unordered_map<std::string, double>& params) {return arb::density(name, params);}))
        .def(pybind11::init([](arb::mechanism_desc mech, const std::unordered_map<std::string, double>& params) {return arb::density(mech, params);}))
        .def_readonly("mech", &arb::density::mech, "The underlying mechanism.")
        .def("__repr__", [](const arb::density& d){return "<arbor.density " + mechanism_desc_str(d.mech) + ">";})
        .def("__str__", [](const arb::density& d){return "<arbor.density " + mechanism_desc_str(d.mech) + ">";});

    pybind11::class_<arb::voltage_process> voltage_process(m, "voltage_process", "For painting a voltage_process mechanism on a region.");
    voltage_process
        .def(pybind11::init([](const std::string& name) {return arb::voltage_process(name);}))
        .def(pybind11::init([](arb::mechanism_desc mech) {return arb::voltage_process(mech);}))
        .def(pybind11::init([](const std::string& name, const std::unordered_map<std::string, double>& params) {return arb::voltage_process(name, params);}))
        .def(pybind11::init([](arb::mechanism_desc mech, const std::unordered_map<std::string, double>& params) {return arb::voltage_process(mech, params);}))
        .def_readonly("mech", &arb::voltage_process::mech, "The underlying mechanism.")
        .def("__repr__", [](const arb::voltage_process& d){return "<arbor.voltage_process " + mechanism_desc_str(d.mech) + ">";})
        .def("__str__", [](const arb::voltage_process& d){return "<arbor.voltage_process " + mechanism_desc_str(d.mech) + ">";});

    // arb::scaled_mechanism<arb::density>

    pybind11::class_<arb::scaled_mechanism<arb::density>> scaled_mechanism(
        m, "scaled_mechanism", "For painting a scaled density mechanism on a region.");
    scaled_mechanism
        .def(pybind11::init(
            [](arb::density dens) { return arb::scaled_mechanism<arb::density>(std::move(dens)); }))
        .def(pybind11::init(
            [](arb::density dens, const std::unordered_map<std::string, std::string>& scales) {
                auto s = arb::scaled_mechanism<arb::density>(std::move(dens));
                for (const auto& it: scales) {
                    s.scale(it.first, arborio::parse_iexpr_expression(it.second).unwrap());
                }
                return s;
            }))
        .def(
            "scale",
            [](arb::scaled_mechanism<arb::density>& s, std::string name, const std::string& ex) {
                s.scale(std::move(name), arborio::parse_iexpr_expression(ex).unwrap());
                return s;
            },
            pybind11::arg("name"),
            pybind11::arg("ex"),
            "Add a scaling expression to a parameter.")
        .def("__repr__",
            [](const arb::scaled_mechanism<arb::density>& d) {
                return "<arbor.scaled_mechanism<density> " + scaled_density_desc_str(d) + ">";
            })
        .def("__str__", [](const arb::scaled_mechanism<arb::density>& d) {
            return "<arbor.scaled_mechanism<density> " + scaled_density_desc_str(d) + ">";
        });

    // arb::synapse

    pybind11::class_<arb::synapse> synapse(m, "synapse", "For placing a synaptic mechanism on a locset.");
    synapse
        .def(pybind11::init([](const std::string& name) {return arb::synapse(name);}))
        .def(pybind11::init([](arb::mechanism_desc mech) {return arb::synapse(mech);}))
        .def(pybind11::init([](const std::string& name, const std::unordered_map<std::string, double>& params) {return arb::synapse(name, params);}))
        .def(pybind11::init([](arb::mechanism_desc mech, const std::unordered_map<std::string, double>& params) {return arb::synapse(mech, params);}))
        .def_readonly("mech", &arb::synapse::mech, "The underlying mechanism.")
        .def("__repr__", [](const arb::synapse& s){return "<arbor.synapse " + mechanism_desc_str(s.mech) + ">";})
        .def("__str__", [](const arb::synapse& s){return "<arbor.synapse " + mechanism_desc_str(s.mech) + ">";});

    // arb::junction

    pybind11::class_<arb::junction> junction(m, "junction", "For placing a gap-junction mechanism on a locset.");
    junction
        .def(pybind11::init([](const std::string& name) {return arb::junction(name);}))
        .def(pybind11::init([](arb::mechanism_desc mech) {return arb::junction(mech);}))
        .def(pybind11::init([](const std::string& name, const std::unordered_map<std::string, double>& params) {return arb::junction(name, params);}))
        .def(pybind11::init([](arb::mechanism_desc mech, const std::unordered_map<std::string, double>& params) {return arb::junction(mech, params);}))
        .def_readonly("mech", &arb::junction::mech, "The underlying mechanism.")
        .def("__repr__", [](const arb::junction& j){return "<arbor.junction " + mechanism_desc_str(j.mech) + ">";})
        .def("__str__", [](const arb::junction& j){return "<arbor.junction " + mechanism_desc_str(j.mech) + ">";});

    // arb::i_clamp

    pybind11::class_<arb::i_clamp> i_clamp(m, "iclamp",
        "A current clamp for injecting a DC or fixed frequency current governed by a piecewise linear envelope.");
    i_clamp
        .def(pybind11::init(
                [](double ts, double dur, double cur, double frequency, double phase) {
                    return arb::i_clamp::box(ts, dur, cur, frequency, phase);
                }), "tstart"_a, "duration"_a, "current"_a, pybind11::kw_only(), "frequency"_a=0, "phase"_a=0,
                "Construct finite duration current clamp, constant amplitude")
        .def(pybind11::init(
                [](double cur, double frequency, double phase) {
                    return arb::i_clamp{cur, frequency, phase};
                }), "current"_a, pybind11::kw_only(), "frequency"_a=0, "phase"_a=0,
                "Construct constant amplitude current clamp")
        .def(pybind11::init(
                [](std::vector<std::pair<double, double>> envl, double frequency, double phase) {
                    arb::i_clamp clamp;
                    for (const auto& p: envl) {
                        clamp.envelope.push_back({p.first, p.second});
                    }
                    clamp.frequency = frequency;
                    clamp.phase = phase;
                    return clamp;
                }), "envelope"_a, pybind11::kw_only(), "frequency"_a=0, "phase"_a=0,
                "Construct current clamp according to (time, amplitude) linear envelope")
        .def_property_readonly("envelope",
                [](const arb::i_clamp& obj) {
                    std::vector<std::pair<double, double>> envl;
                    for (const auto& p: obj.envelope) {
                        envl.push_back({p.t, p.amplitude});
                    }
                    return envl;
                },
                "List of (time [ms], amplitude [nA]) points comprising the piecewise linear envelope")
        .def_readonly("frequency", &arb::i_clamp::frequency, "Oscillation frequency (kHz), zero implies DC stimulus.")
        .def_readonly("phase", &arb::i_clamp::phase, "Oscillation initial phase (rad)")
        .def("__repr__", [](const arb::i_clamp& c) {
            return util::pprintf("<arbor.iclamp: frequency {} Hz>", c.frequency);})
        .def("__str__", [](const arb::i_clamp& c) {
            return util::pprintf("<arbor.iclamp: frequency {} Hz>", c.frequency);});

    // arb::threshold_detector
    pybind11::class_<arb::threshold_detector> detector(m, "threshold_detector",
            "A spike detector, generates a spike when voltage crosses a threshold. Can be used as source endpoint for an arbor.connection.");
    detector
        .def(pybind11::init(
            [](double thresh) { return arb::threshold_detector{thresh}; }),
            "threshold"_a, "Voltage threshold of spike detector [mV]")
        .def_readonly("threshold", &arb::threshold_detector::threshold, "Voltage threshold of spike detector [mV]")
        .def("__repr__", [](const arb::threshold_detector& d){
            return util::pprintf("<arbor.threshold_detector: threshold {} mV>", d.threshold);})
        .def("__str__", [](const arb::threshold_detector& d){
            return util::pprintf("(threshold_detector {})", d.threshold);});

    // arb::cable_cell_global_properties
    pybind11::class_<arb::cable_cell_ion_data> ion_data(m, "ion_data");
    ion_data
        .def_readonly("internal_concentration", &arb::cable_cell_ion_data::init_int_concentration,  "Internal concentration.")
        .def_readonly("external_concentration", &arb::cable_cell_ion_data::init_ext_concentration,  "External concentration.")
        .def_readonly("diffusivity",            &arb::cable_cell_ion_data::diffusivity,             "Diffusivity.")
        .def_readonly("reversal_concentration", &arb::cable_cell_ion_data::init_reversal_potential, "Reversal potential.");

    struct ion_settings {
        int charge = 0;
        std::optional<double> internal_concentration;
        std::optional<double> external_concentration;
        std::optional<double> diffusivity;
        std::optional<double> reversal_potential;
        std::string reversal_potential_method = "const";
    };

    pybind11::class_<ion_settings> py_ion_data(m, "ion_settings");
    ion_data
        .def_property_readonly("charge",                    [](const ion_settings& s) { return s.charge; },                    "Valence.")
        .def_property_readonly("internal_concentration",    [](const ion_settings& s) { return s.internal_concentration; },    "Internal concentration.")
        .def_property_readonly("external_concentration",    [](const ion_settings& s) { return s.external_concentration; },    "External concentration.")
        .def_property_readonly("diffusivity",               [](const ion_settings& s) { return s.diffusivity; },               "Diffusivity.")
        .def_property_readonly("reversal_potential",        [](const ion_settings& s) { return s.reversal_potential; },        "Reversal potential.")
        .def_property_readonly("reversal_potential_method", [](const ion_settings& s) { return s.reversal_potential_method; }, "Reversal potential method.");

    pybind11::class_<arb::cable_cell_global_properties> gprop(m, "cable_global_properties");
    gprop
        .def(pybind11::init<>())
        .def(pybind11::init<const arb::cable_cell_global_properties&>())
        .def("check", [](const arb::cable_cell_global_properties& props) {
                arb::check_global_properties(props);},
                "Test whether all default parameters and ion species properties have been set.")
        .def_readwrite("coalesce_synapses",  &arb::cable_cell_global_properties::coalesce_synapses,
                "Flag for enabling/disabling linear syanpse coalescing.")
        // set cable properties
        .def_property("membrane_potential",
                      [](const arb::cable_cell_global_properties& props) { return props.default_parameters.init_membrane_potential; },
                      [](arb::cable_cell_global_properties& props, double u) { props.default_parameters.init_membrane_potential = u; })
        .def_property("membrane_voltage_limit",
                      [](const arb::cable_cell_global_properties& props) { return props.membrane_voltage_limit_mV; },
                      [](arb::cable_cell_global_properties& props, std::optional<double> u) { props.membrane_voltage_limit_mV = u; })
        .def_property("membrane_capacitance",
                      [](const arb::cable_cell_global_properties& props) { return props.default_parameters.membrane_capacitance; },
                      [](arb::cable_cell_global_properties& props, double u) { props.default_parameters.membrane_capacitance = u; })
        .def_property("temperature",
                      [](const arb::cable_cell_global_properties& props) { return props.default_parameters.temperature_K; },
                      [](arb::cable_cell_global_properties& props, double u) { props.default_parameters.temperature_K = u; })
        .def_property("axial_resisitivity",
                      [](const arb::cable_cell_global_properties& props) { return props.default_parameters.axial_resistivity; },
                      [](arb::cable_cell_global_properties& props, double u) { props.default_parameters.axial_resistivity = u; })
        .def("set_property",
            [](arb::cable_cell_global_properties& props,
               optional<double> Vm, optional<double> cm,
               optional<double> rL, optional<double> tempK)
            {
                if (Vm) props.default_parameters.init_membrane_potential = Vm;
                if (cm) props.default_parameters.membrane_capacitance=cm;
                if (rL) props.default_parameters.axial_resistivity=rL;
                if (tempK) props.default_parameters.temperature_K=tempK;
            },
            pybind11::arg_v("Vm",    pybind11::none(), "initial membrane voltage [mV]."),
            pybind11::arg_v("cm",    pybind11::none(), "membrane capacitance [F/m²]."),
            pybind11::arg_v("rL",    pybind11::none(), "axial resistivity [Ω·cm]."),
            pybind11::arg_v("tempK", pybind11::none(), "temperature [Kelvin]."),
            "Set global default values for cable and cell properties.")
        // add/modify ion species
        .def("unset_ion",
             [](arb::cable_cell_global_properties& props, const char* ion) {
                 props.ion_species.erase(ion);
                 props.default_parameters.ion_data.erase(ion);
                 props.default_parameters.reversal_potential_method.erase(ion);
             },
             "Remove ion species from properties.")
        .def("set_ion",
            [](arb::cable_cell_global_properties& props, const char* ion,
               optional<double> valence, optional<double> int_con,
               optional<double> ext_con, optional<double> rev_pot,
               pybind11::object method, optional<double> diff)
            {
                if (!props.ion_species.count(ion) && !valence) {
                    throw std::runtime_error(util::pprintf("New ion species: '{}', missing valence", ion));
                }
                if (valence) props.ion_species[ion] = *valence;

                auto& data = props.default_parameters.ion_data[ion];
                if (int_con) data.init_int_concentration  = *int_con;
                if (ext_con) data.init_ext_concentration  = *ext_con;
                if (rev_pot) data.init_reversal_potential = *rev_pot;
                if (diff)    data.diffusivity             = *diff;

                if (auto m = maybe_method(method)) {
                    props.default_parameters.reversal_potential_method[ion] = *m;
                }
            },
            pybind11::arg("ion"),
            pybind11::arg_v("valence", pybind11::none(), "valence of the ion species."),
            pybind11::arg_v("int_con", pybind11::none(), "initial internal concentration [mM]."),
            pybind11::arg_v("ext_con", pybind11::none(), "initial external concentration [mM]."),
            pybind11::arg_v("rev_pot", pybind11::none(), "reversal potential [mV]."),
            pybind11::arg_v("method",  pybind11::none(), "method for calculating reversal potential."),
            pybind11::arg_v("diff",    pybind11::none(), "diffusivity [m^2/s]."),
            "Set the global default properties of ion species named 'ion'.\n"
            "There are 3 ion species predefined in arbor: 'ca', 'na' and 'k'.\n"
            "If 'ion' in not one of these ions it will be added to the list, making it\n"
            "available to mechanisms. The user has to provide the valence of a previously\n"
            "undefined ion the first time this function is called with it as an argument.\n"
            "Species concentrations and reversal potential can be overridden on\n"
            "specific regions using the paint interface, while the method for calculating\n"
            "reversal potential is global for all compartments in the cell, and can't be\n"
             "overriden locally.")
        .def_property_readonly("ion_data",
                      [](const arb::cable_cell_global_properties& props) { return props.default_parameters.ion_data; })
        .def_property_readonly("ion_valence",
                      [](const arb::cable_cell_global_properties& props) { return props.ion_species; })
        .def_property_readonly("ion_reversal_potential",
                      [](const arb::cable_cell_global_properties& props) { return props.default_parameters.reversal_potential_method; })
        .def_property_readonly("ions",
                               [](arb::cable_cell_global_properties& g) {
                                   std::unordered_map<std::string, ion_settings> result;
                                   for (const auto& [k, v]: g.ion_species) {
                                       auto& ion = result[k];
                                       ion.charge = v;
                                       auto& data = g.default_parameters.ion_data;
                                       if (data.count(k)) {
                                           auto& i = data.at(k);
                                           ion.diffusivity = i.diffusivity;
                                           ion.external_concentration = i.init_ext_concentration;
                                           ion.internal_concentration = i.init_int_concentration;
                                           ion.reversal_potential     = i.init_reversal_potential;
                                       }
                                       auto& revpot = g.default_parameters.reversal_potential_method;
                                       if (revpot.count(k)) {
                                           ion.reversal_potential_method = revpot.at(k).name();
                                       }
                                   }
                                   return result;
                               },
                               "Return a view of all ion settings.")
        .def_readwrite("catalogue",
                       &arb::cable_cell_global_properties::catalogue,
                       "The mechanism catalogue.")
        .def("__str__", [](const arb::cable_cell_global_properties& p){return to_string(p);});

    m.def("neuron_cable_properties", []() {
        arb::cable_cell_global_properties prop;
        prop.default_parameters = arb::neuron_parameter_defaults;
        return prop;
    },
    "default NEURON cable_global_properties");

    // arb::decor

    pybind11::class_<arb::decor> decor(m, "decor",
            "Description of the decorations to be applied to a cable cell, that is the painted,\n"
            "placed and defaulted properties, mecahanisms, ion species etc.");
    decor
        .def(pybind11::init<>())
        .def(pybind11::init<const arb::decor&>())
        // Set cell-wide default values for properties
        .def("set_property",
            [](arb::decor& d,
               optional<double> Vm, optional<double> cm,
               optional<double> rL, optional<double> tempK)
            {
                if (Vm) d.set_default(arb::init_membrane_potential{*Vm});
                if (cm) d.set_default(arb::membrane_capacitance{*cm});
                if (rL) d.set_default(arb::axial_resistivity{*rL});
                if (tempK) d.set_default(arb::temperature_K{*tempK});
                return d;
            },
            pybind11::arg_v("Vm",    pybind11::none(), "initial membrane voltage [mV]."),
            pybind11::arg_v("cm",    pybind11::none(), "membrane capacitance [F/m²]."),
            pybind11::arg_v("rL",    pybind11::none(), "axial resistivity [Ω·cm]."),
            pybind11::arg_v("tempK", pybind11::none(), "temperature [Kelvin]."),
            "Set default values for cable and cell properties. These values can be overridden on specific regions using the paint interface.")
        // modify parameters for an ion species.
        .def("set_ion",
            [](arb::decor& d, const char* ion,
               optional<double> int_con, optional<double> ext_con,
               optional<double> rev_pot, pybind11::object method,
               optional<double> diff)
            {
                if (int_con) d.set_default(arb::init_int_concentration{ion, *int_con});
                if (ext_con) d.set_default(arb::init_ext_concentration{ion, *ext_con});
                if (rev_pot) d.set_default(arb::init_reversal_potential{ion, *rev_pot});
                if (diff)    d.set_default(arb::ion_diffusivity{ion, *diff});
                if (auto m = maybe_method(method)) d.set_default(arb::ion_reversal_potential_method{ion, *m});
                return d;
            },
            pybind11::arg("ion"),
            pybind11::arg_v("int_con", pybind11::none(), "initial internal concentration [mM]."),
            pybind11::arg_v("ext_con", pybind11::none(), "initial external concentration [mM]."),
            pybind11::arg_v("rev_pot", pybind11::none(), "reversal potential [mV]."),
            pybind11::arg_v("method",  pybind11::none(), "method for calculating reversal potential."),
            pybind11::arg_v("diff",    pybind11::none(), "diffusivity [m^2/s]."),
            "Set the properties of ion species named 'ion' that will be applied\n"
            "by default everywhere on the cell. Species concentrations and reversal\n"
            "potential can be overridden on specific regions using the paint interface, \n"
            "while the method for calculating reversal potential is global for all\n"
            "compartments in the cell, and can't be overriden locally.")
        .def("paintings",
            [](arb::decor& dec) {
                std::vector<std::tuple<std::string, arb::paintable>> result;
                for (const auto& [k, v]: dec.paintings()) {
                    result.emplace_back(to_string(k), v);
                }
                return result;
            },
            "Return a view of all painted items.")
        .def("placements",
            [](arb::decor& dec) {
                std::vector<std::tuple<std::string, arb::placeable, std::string>> result;
                for (const auto& [k, v, t]: dec.placements()) {
                    result.emplace_back(to_string(k), v, t);
                }
                return result;
            },
            "Return a view of all placed items.")
        .def("defaults",
            [](arb::decor& dec) {
                return dec.defaults().serialize();
            },
            "Return a view of all defaults.")
        // Paint mechanisms.
        .def("paint",
            [](arb::decor& dec, const char* region, const arb::density& mechanism) {
                return dec.paint(arborio::parse_region_expression(region).unwrap(), mechanism);
            },
            "region"_a, "mechanism"_a,
            "Associate a density mechanism with a region.")
        .def("paint",
            [](arb::decor& dec, const char* region, const arb::voltage_process& mechanism) {
                return dec.paint(arborio::parse_region_expression(region).unwrap(), mechanism);
            },
            "region"_a, "mechanism"_a,
            "Associate a voltage process mechanism with a region.")
        .def("paint",
            [](arb::decor& dec, const char* region, const arb::scaled_mechanism<arb::density>& mechanism) {
                dec.paint(arborio::parse_region_expression(region).unwrap(), mechanism);
            },
            "region"_a, "mechanism"_a,
            "Associate a scaled density mechanism with a region.")
        // Paint membrane/static properties.
        .def("paint",
            [](arb::decor& dec,
               const char* region,
               optional<double> Vm, optional<double> cm,
               optional<double> rL, optional<double> tempK)
            {
                auto r = arborio::parse_region_expression(region).unwrap();
                if (Vm) dec.paint(r, arb::init_membrane_potential{*Vm});
                if (cm) dec.paint(r, arb::membrane_capacitance{*cm});
                if (rL) dec.paint(r, arb::axial_resistivity{*rL});
                if (tempK) dec.paint(r, arb::temperature_K{*tempK});
                return dec;
            },
            pybind11::arg("region"),
            pybind11::arg_v("Vm",    pybind11::none(), "initial membrane voltage [mV]."),
            pybind11::arg_v("cm",    pybind11::none(), "membrane capacitance [F/m²]."),
            pybind11::arg_v("rL",    pybind11::none(), "axial resistivity [Ω·cm]."),
            pybind11::arg_v("tempK", pybind11::none(), "temperature [Kelvin]."),
            "Set cable properties on a region.")
        // Paint ion species initial conditions on a region.
        .def("paint",
            [](arb::decor& dec, const char* region, const char* name,
               optional<double> int_con, optional<double> ext_con,
               optional<double> rev_pot, optional<double> diff) {
                auto r = arborio::parse_region_expression(region).unwrap();
                if (int_con) dec.paint(r, arb::init_int_concentration{name, *int_con});
                if (ext_con) dec.paint(r, arb::init_ext_concentration{name, *ext_con});
                if (rev_pot) dec.paint(r, arb::init_reversal_potential{name, *rev_pot});
                if (diff)    dec.paint(r, arb::ion_diffusivity{name, *diff});
                return dec;
            },
            "region"_a, pybind11::kw_only(), "ion_name"_a,
            pybind11::arg_v("int_con", pybind11::none(), "Initial internal concentration [mM]"),
            pybind11::arg_v("ext_con", pybind11::none(), "Initial external concentration [mM]"),
            pybind11::arg_v("rev_pot", pybind11::none(), "Initial reversal potential [mV]"),
            pybind11::arg_v("diff",    pybind11::none(), "Diffusivity [m^2/s]"),
            "Set ion species properties conditions on a region.")
        // Place synapses
        .def("place",
            [](arb::decor& dec, const char* locset, const arb::synapse& mechanism, const char* label_name) {
                return dec.place(arborio::parse_locset_expression(locset).unwrap(), mechanism, label_name);
            },
            "locations"_a, "synapse"_a, "label"_a,
            "Place one instance of 'synapse' on each location in 'locations'."
            "The group of synapses has the label 'label', used for forming connections between cells.")
        // Place gap junctions.
        .def("place",
            [](arb::decor& dec, const char* locset, const arb::junction& mechanism, const char* label_name) {
                return dec.place(arborio::parse_locset_expression(locset).unwrap(), mechanism, label_name);
            },
            "locations"_a, "junction"_a, "label"_a,
            "Place one instance of 'junction' on each location in 'locations'."
            "The group of junctions has the label 'label', used for forming gap-junction connections between cells.")
        // Place current clamp stimulus.
        .def("place",
            [](arb::decor& dec, const char* locset, const arb::i_clamp& stim, const char* label_name) {
                return dec.place(arborio::parse_locset_expression(locset).unwrap(), stim, label_name);
            },
            "locations"_a, "iclamp"_a, "label"_a,
            "Add a current stimulus at each location in locations."
            "The group of current stimuli has the label 'label'.")
        // Place spike detector.
        .def("place",
            [](arb::decor& dec, const char* locset, const arb::threshold_detector& d, const char* label_name) {
                return dec.place(arborio::parse_locset_expression(locset).unwrap(), d, label_name);
            },
            "locations"_a, "detector"_a, "label"_a,
            "Add a voltage spike detector at each location in locations."
            "The group of spike detectors has the label 'label', used for forming connections between cells.")
        .def("discretization",
            [](arb::decor& dec, const arb::cv_policy& p) { return dec.set_default(p); },
            pybind11::arg("policy"),
             "A cv_policy used to discretise the cell into compartments for simulation")
        .def("discretization",
            [](arb::decor& dec, const std::string& p) {
                return dec.set_default(arborio::parse_cv_policy_expression(p).unwrap());
            },
            pybind11::arg("policy"),
            "An s-expression string representing a cv_policy used to discretise the "
            "cell into compartments for simulation");

    // arb::cable_cell

    pybind11::class_<arb::cable_cell> cable_cell(m, "cable_cell",
        "Represents morphologically-detailed cell models, with morphology represented as a\n"
        "tree of one-dimensional cable segments.");
    cable_cell
        .def(pybind11::init(
            [](const arb::morphology& m, const arb::decor& d, const std::optional<label_dict_proxy>& l) {
                if (l) return arb::cable_cell(m, d, l->dict);
                return arb::cable_cell(m, d);
            }),
            "morphology"_a,
             "decor"_a,
             pybind11::arg_v("labels", pybind11::none(), "Labels"),
            "Construct with a morphology, decor, and label dictionary.")
        .def(pybind11::init(
            [](const arb::segment_tree& t, const arb::decor& d, const std::optional<label_dict_proxy>& l) {
                if (l) return arb::cable_cell({t}, d, l->dict);
                return arb::cable_cell({t}, d);
            }),
            "segment_tree"_a,
             "decor"_a,
             pybind11::arg_v("labels", pybind11::none(), "Labels"),
            "Construct with a morphology derived from a segment tree, decor, and label dictionary.")
        .def_property_readonly("num_branches",
            [](const arb::cable_cell& c) {return c.morphology().num_branches();},
            "The number of unbranched cable sections in the morphology.")
        // Get locations associated with a locset label.
        .def("locations",
            [](arb::cable_cell& c, const char* label) {return c.concrete_locset(arborio::parse_locset_expression(label).unwrap());},
            "label"_a, "The locations of the cell morphology for a locset label.")
        // Get cables associated with a region label.
        .def("cables",
            [](arb::cable_cell& c, const char* label) {return c.concrete_region(arborio::parse_region_expression(label).unwrap()).cables();},
            "label"_a, "The cable segments of the cell morphology for a region label.")
        // Stringification
        .def("__repr__", [](const arb::cable_cell&){return "<arbor.cable_cell>";})
        .def("__str__",  [](const arb::cable_cell&){return "<arbor.cable_cell>";});
}

} // namespace pyarb
