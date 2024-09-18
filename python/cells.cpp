#include <algorithm>
#include <optional>
#include <string>
#include <utility>

#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <pybind11/cast.h>
#include <pybind11/pytypes.h>

#include <arborio/cv_policy_parse.hpp>
#include <arborio/label_parse.hpp>

#include <arbor/benchmark_cell.hpp>
#include <arbor/cable_cell.hpp>
#include <arbor/lif_cell.hpp>
#include <arbor/cv_policy.hpp>
#include <arbor/cable_cell_param.hpp>
#include <arbor/morph/cv_data.hpp>
#include <arbor/morph/label_dict.hpp>
#include <arbor/morph/locset.hpp>
#include <arbor/morph/region.hpp>
#include <arbor/morph/segment_tree.hpp>
#include <arbor/schedule.hpp>
#include <arbor/spike_source_cell.hpp>
#include <arbor/util/any_cast.hpp>
#include <arbor/util/unique_any.hpp>
#include <arbor/cv_policy.hpp>

#include "conversion.hpp"
#include "error.hpp"
#include "label_dict.hpp"
#include "schedule.hpp"
#include "strprintf.hpp"
#include "util.hpp"

namespace pyarb {

namespace U = arb::units;

namespace py = pybind11;

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

struct ion_settings {
    int charge = 0;
    std::optional<double> internal_concentration;
    std::optional<double> external_concentration;
    std::optional<double> diffusivity;
    std::optional<double> reversal_potential;
    std::string reversal_potential_method = "const";
};

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
std::optional<arb::mechanism_desc> maybe_method(py::object method) {
    if (!method.is_none()) {
        if (auto m = try_cast<std::string>(method)) {
            return m;
        }
        else if (auto m=try_cast<arb::mechanism_desc>(method)) {
            return m;
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
        "<arbor.lif_cell: tau_m {}, V_th {}, C_m {}, E_L {}, E_R {}, V_m {}, t_ref {}>",
        U::to_string(c.tau_m), U::to_string(c.V_th), U::to_string(c.C_m),
        U::to_string(c.E_L), U::to_string(c.E_R), U::to_string(c.V_m), U::to_string(c.t_ref));
}


std::string mechanism_desc_str(const arb::mechanism_desc& md) {
    return util::pprintf("mechanism('{}', {})",
            md.name(), util::dictionary_csv(md.values()));
}

std::string scaled_density_desc_str(const arb::scaled_mechanism<arb::density>& p) {
    return util::pprintf("({}, {})",
            mechanism_desc_str(p.t_mech.mech), util::dictionary_csv(p.scale_expr));
}

// Argument to construct a paintable object.
using Q = U::quantity;
using QnS = std::tuple<U::quantity, std::string>;
using paintable_arg = std::variant<Q, QnS>;

std::tuple<Q, arb::iexpr> value_and_scale(const paintable_arg& arg) {
    if (std::holds_alternative<Q>(arg)) {
        return {std::get<Q>(arg), 1};
    }
    else {
        const auto& [val, scale] = std::get<QnS>(arg);
        return {val, arborio::parse_iexpr_expression(scale).unwrap()};
    }
}

void register_cells(py::module& m) {
    using namespace py::literals;
    using std::optional;

    py::class_<arb::spike_source_cell> spike_source_cell(m, "spike_source_cell",
        "A spike source cell, that generates a user-defined sequence of spikes that act as inputs for other cells in the network.");
    py::class_<arb::cell_cv_data> cell_cv_data(m, "cell_cv_data",
            "Provides information on the CVs representing the discretization of a cable-cell.");
    py::class_<arb::benchmark_cell> benchmark_cell(m, "benchmark_cell",
                                                   "A benchmarking cell, used by Arbor developers to test communication performance.\n"
                                                   "A benchmark cell generates spikes at a user-defined sequence of time points, and\n"
                                                   "the time taken to integrate a cell can be tuned by setting the realtime_ratio,\n"
                                                   "for example if realtime_ratio=2, a cell will take 2 seconds of CPU time to\n"
                                                   "simulate 1 second.\n");
    py::class_<arb::lif_cell> lif_cell(m, "lif_cell", "A leaky integrate-and-fire cell.");
    py::class_<arb::cv_policy> cv_policy(m, "cv_policy", "Describes the rules used to discretize (compartmentalise) a cable cell morphology.");
    py::class_<ion_settings> py_ion_data(m, "ion_settings");
    py::class_<arb::cable_cell_global_properties> gprop(m, "cable_global_properties");
    py::class_<arb::decor> decor(m, "decor",
                                 "Description of the decorations to be applied to a cable cell, that is the painted,\n"
                                 "placed and defaulted properties, mecahanisms, ion species etc.");
    py::class_<arb::cable_cell> cable_cell(m, "cable_cell",
                                           "Represents morphologically-detailed cell models, with morphology represented as a\n"
                                           "tree of one-dimensional cable segments.");
    py::class_<arb::init_membrane_potential> membrane_potential(m, "membrane_potential", "Setting the initial membrane voltage.");
    py::class_<arb::ion_reversal_potential_method> revpot_method(m, "reversal_potential_method", "Describes the mechanism used to compute eX for ion X.");
    py::class_<arb::membrane_capacitance> membrane_capacitance(m, "membrane_capacitance", "Setting the membrane capacitance.");
    py::class_<arb::temperature> temperature_K(m, "temperature", "Setting the temperature.");
    py::class_<arb::axial_resistivity> axial_resistivity(m, "axial_resistivity", "Setting the axial resistivity.");
    py::class_<arb::init_reversal_potential> reversal_potential(m, "reversal_potential", "Setting the initial reversal potential.");
    py::class_<arb::init_int_concentration> int_concentration(m, "int_concentration", "Setting the initial internal ion concentration.");
    py::class_<arb::init_ext_concentration> ext_concentration(m, "ext_concentration", "Setting the initial external ion concentration.");
    py::class_<arb::ion_diffusivity> ion_diffusivity(m, "ion_diffusivity", "Setting the ion diffusivity.");
    py::class_<arb::density> density(m, "density", "For painting a density mechanism on a region.");
    py::class_<arb::voltage_process> voltage_process(m, "voltage_process", "For painting a voltage_process mechanism on a region.");
    py::class_<arb::scaled_mechanism<arb::density>> scaled_mechanism(m, "scaled_mechanism", "For painting a scaled density mechanism on a region.");
    py::class_<arb::cable_cell_ion_data> ion_data(m, "ion_data");
    py::class_<arb::threshold_detector> detector(m, "threshold_detector", "A spike detector, generates a spike when voltage crosses a threshold. Can be used as source endpoint for an arbor.connection.");
    py::class_<arb::synapse> synapse(m, "synapse", "For placing a synaptic mechanism on a locset.");
    py::class_<arb::junction> junction(m, "junction", "For placing a gap-junction mechanism on a locset.");
    py::class_<arb::i_clamp> i_clamp(m, "iclamp", "A current clamp for injecting a DC or fixed frequency current governed by a piecewise linear envelope.");

    spike_source_cell
        .def(py::init<>(
            [](arb::cell_tag_type source_label, const regular_schedule_shim& sched){
                return arb::spike_source_cell{std::move(source_label), sched.schedule()};}),
            "source_label"_a, "schedule"_a,
            "Construct a spike source cell with a single source labeled 'source_label'.\n"
            "The cell generates spikes on 'source_label' at regular intervals.")
        .def(py::init<>(
            [](arb::cell_tag_type source_label, const explicit_schedule_shim& sched){
                return arb::spike_source_cell{std::move(source_label), sched.schedule()};}),
            "source_label"_a, "schedule"_a,
            "Construct a spike source cell with a single source labeled 'source_label'.\n"
            "The cell generates spikes on 'source_label' at a sequence of user-defined times.")
        .def(py::init<>(
            [](arb::cell_tag_type source_label, const poisson_schedule_shim& sched){
                return arb::spike_source_cell{std::move(source_label), sched.schedule()};}),
            "source_label"_a, "schedule"_a,
            "Construct a spike source cell with a single source labeled 'source_label'.\n"
            "The cell generates spikes on 'source_label' at times defined by a Poisson sequence.")
        .def("__repr__", [](const arb::spike_source_cell&){return "<arbor.spike_source_cell>";})
        .def("__str__",  [](const arb::spike_source_cell&){return "<arbor.spike_source_cell>";});

    benchmark_cell
        .def(py::init<>(
            [](arb::cell_tag_type source_label, arb::cell_tag_type target_label, const regular_schedule_shim& sched, double ratio){
                return arb::benchmark_cell{std::move(source_label), std::move(target_label), sched.schedule(), ratio};}),
            "source_label"_a, "target_label"_a,"schedule"_a, "realtime_ratio"_a=1.0,
            "Construct a benchmark cell that generates spikes on 'source_label' at regular intervals.\n"
            "The cell has one source labeled 'source_label', and one target labeled 'target_label'.")
        .def(py::init<>(
            [](arb::cell_tag_type source_label, arb::cell_tag_type target_label, const explicit_schedule_shim& sched, double ratio){
                return arb::benchmark_cell{std::move(source_label), std::move(target_label),sched.schedule(), ratio};}),
            "source_label"_a, "target_label"_a, "schedule"_a, "realtime_ratio"_a=1.0,
            "Construct a benchmark cell that generates spikes on 'source_label' at a sequence of user-defined times.\n"
            "The cell has one source labeled 'source_label', and one target labeled 'target_label'.")
        .def(py::init<>(
            [](arb::cell_tag_type source_label, arb::cell_tag_type target_label, const poisson_schedule_shim& sched, double ratio){
                return arb::benchmark_cell{std::move(source_label), std::move(target_label), sched.schedule(), ratio};}),
            "source_label"_a, "target_label"_a, "schedule"_a, "realtime_ratio"_a=1.0,
            "Construct a benchmark cell that generates spikeson 'source_label' at times defined by a Poisson sequence.\n"
            "The cell has one source labeled 'source_label', and one target labeled 'target_label'.")
        .def("__repr__", [](const arb::benchmark_cell&){return "<arbor.benchmark_cell>";})
        .def("__str__",  [](const arb::benchmark_cell&){return "<arbor.benchmark_cell>";});

    lif_cell
        .def(py::init<>(
            [](arb::cell_tag_type source_label,
               arb::cell_tag_type target_label,
               std::optional<U::quantity> tau_m,
               std::optional<U::quantity> V_th,
               std::optional<U::quantity> C_m,
               std::optional<U::quantity> E_L,
               std::optional<U::quantity> E_R,
               std::optional<U::quantity> V_m,
               std::optional<U::quantity> t_ref) {
                auto cell = arb::lif_cell{std::move(source_label), std::move(target_label)};
                if (tau_m) cell.tau_m = *tau_m;
                if (V_th) cell.V_th = *V_th;
                if (C_m) cell.C_m = *C_m;
                if (E_L) cell.E_L = *E_L;
                if (E_R) cell.E_R = *E_R;
                if (V_m) cell.V_m = *V_m;
                if (t_ref) cell.t_ref = *t_ref;
                return cell;
            }),
            "source_label"_a, "target_label"_a,
             py::kw_only(), "tau_m"_a=py::none(), "V_th"_a=py::none(), "C_m"_a=py::none(), "E_L"_a=py::none(), "E_R"_a=py::none(), "V_m"_a=py::none(), "t_ref"_a=py::none(),
             "Construct a lif cell with one source labeled 'source_label', and one target labeled 'target_label'."
             "Can optionally take physical parameters:\n"
             " * tau_m: Membrane potential decaying constant [ms].\n"
             " * V_th: Firing threshold [mV].\n"
             " * C_m: Membrane capacitance [pF].\n"
             " * E_L: Resting potential [mV].\n"
             " * E_R: Reset potential [mV].\n"
             " * V_m: Initial value of the Membrane potential [mV].\n"
             " * t_ref: Refractory period [ms].")
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

    // arb::cv_policy wrappers
    cv_policy
        .def(py::init([](const std::string& expression) { return arborio::parse_cv_policy_expression(expression).unwrap(); }),
            "expression"_a, "A valid CV policy expression")
        .def_property_readonly("domain",
                               [](const arb::cv_policy& p) {return util::pprintf("{}", p.domain());},
                               "The domain on which the policy is applied.")
        .def(py::self + py::self)
        .def(py::self | py::self)
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
    cell_cv_data
    .def_property_readonly("num_cv", [](const arb::cell_cv_data& data){return data.size();},
                 "Return the number of CVs in the cell.")
            .def("cables",
                 [](const arb::cell_cv_data& d, unsigned index) {
                    if (index >= d.size()) throw py::index_error("index out of range");
                    return d.cables(index);
                 },
                 "index"_a, "Return a list of cables representing the CV at the given index.")
            .def("children",
                 [](const arb::cell_cv_data& d, unsigned index) {
                     if (index >= d.size()) throw py::index_error("index out of range");
                     return d.children(index);
                 },
                 "index"_a,
                 "Return a list of indices of the CVs representing the children of the CV at the given index.")
            .def("parent",
                 [](const arb::cell_cv_data& d, unsigned index) {
                     if (index >= d.size()) throw py::index_error("index out of range");
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
              auto tuple_vec = std::vector<py::tuple>(object_vec.size());
              std::transform(object_vec.begin(), object_vec.end(), tuple_vec.begin(),
                             [](const auto& t)  { return py::make_tuple(t.idx, t.proportion); });
              return tuple_vec;
          },
          "reg"_a,  "A region on a cell",
          "data"_a, "The data representing the CVs of the cell.",
          "integrate_along"_a, "the axis of integration used to determine the proportion of the CV belonging to the region.",
          "Returns a list of [index, proportion] tuples identifying the CVs present in the region.\n"
          "`index` is the index of the CV in the cell_cv_data object provided as an argument.\n"
          "`proportion` is the proportion of the CV (itegrated by area or length) included in the region."
    );

    membrane_potential
        .def(py::init([](const U::quantity& v) -> arb::init_membrane_potential { return {v }; }))
        .def("__repr__",
             [](const arb::init_membrane_potential& d){
                 return "Vm=" + to_string(d.value) + " scale=" + to_string(d.scale);});

    revpot_method
        .def(py::init([](const std::string& ion,
                         const arb::mechanism_desc& d) -> arb::ion_reversal_potential_method {
            return {ion, d};
        }))
        .def("__repr__", [](const arb::ion_reversal_potential_method& d) {
            return "ion" + d.ion + " method=" + d.method.name();});

    membrane_capacitance
        .def(py::init([](const U::quantity& v) -> arb::membrane_capacitance { return {v}; }))
        .def("__repr__", [](const arb::membrane_capacitance& d){return "Cm=" + to_string(d.value);});

    temperature_K
        .def(py::init([](const U::quantity& v) -> arb::temperature { return {v}; }))
        .def("__repr__", [](const arb::temperature& d){return "T=" + to_string(d.value);});

    axial_resistivity
        .def(py::init([](const U::quantity& v) -> arb::axial_resistivity { return {v}; }))
        .def("__repr__", [](const arb::axial_resistivity& d){return "Ra" + to_string(d.value);});

    reversal_potential
        .def(py::init([](const std::string& i, const U::quantity& v) -> arb::init_reversal_potential { return {i, v}; }))
        .def("__repr__", [](const arb::init_reversal_potential& d){return "e" + d.ion + "=" + to_string(d.value);});

    int_concentration
        .def(py::init([](const std::string& i, const U::quantity& v) -> arb::init_int_concentration { return {i, v}; }))
        .def("__repr__", [](const arb::init_int_concentration& d){return d.ion + "i" + "=" + to_string(d.value);});

    ext_concentration
        .def(py::init([](const std::string& i, const U::quantity& v) -> arb::init_ext_concentration { return {i, v}; }))
        .def("__repr__", [](const arb::init_ext_concentration& d){return d.ion + "o" + "=" + to_string(d.value);});

    ion_diffusivity
        .def(py::init([](const std::string& i, const U::quantity& v) -> arb::ion_diffusivity { return {i, v}; }))
        .def("__repr__", [](const arb::ion_diffusivity& d){return "D" + d.ion + "=" + to_string(d.value);});

    density
        .def(py::init([](const std::string& name) {return arb::density(name);}))
        .def(py::init([](arb::mechanism_desc mech) {return arb::density(mech);}))
        .def(py::init([](const std::string& name, const std::unordered_map<std::string, double>& params) {return arb::density(name, params);}))
        .def(py::init([](arb::mechanism_desc mech, const std::unordered_map<std::string, double>& params) {return arb::density(mech, params);}))
        .def(py::init([](const std::string& name, py::kwargs parms) {return arb::density(name, util::dict_to_map<double>(parms));}))
        .def(py::init([](arb::mechanism_desc mech, py::kwargs params) {return arb::density(mech, util::dict_to_map<double>(params));}))
        .def_readonly("mech", &arb::density::mech, "The underlying mechanism.")
        .def("__repr__", [](const arb::density& d){return "<arbor.density " + mechanism_desc_str(d.mech) + ">";})
        .def("__str__", [](const arb::density& d){return "<arbor.density " + mechanism_desc_str(d.mech) + ">";});

    voltage_process
        .def(py::init([](const std::string& name) {return arb::voltage_process(name);}))
        .def(py::init([](arb::mechanism_desc mech) {return arb::voltage_process(mech);}))
        .def(py::init([](const std::string& name, const std::unordered_map<std::string, double>& params) {return arb::voltage_process(name, params);}))
        .def(py::init([](arb::mechanism_desc mech, const std::unordered_map<std::string, double>& params) {return arb::voltage_process(mech, params);}))
        .def(py::init([](arb::mechanism_desc mech, py::kwargs params) {return arb::voltage_process(mech, util::dict_to_map<double>(params));}))
        .def(py::init([](const std::string& name, py::kwargs parms) {return arb::voltage_process(name, util::dict_to_map<double>(parms));}))
        .def_readonly("mech", &arb::voltage_process::mech, "The underlying mechanism.")
        .def("__repr__", [](const arb::voltage_process& d){return "<arbor.voltage_process " + mechanism_desc_str(d.mech) + ">";})
        .def("__str__", [](const arb::voltage_process& d){return "<arbor.voltage_process " + mechanism_desc_str(d.mech) + ">";});

    scaled_mechanism
        .def(py::init(
            [](arb::density dens) { return arb::scaled_mechanism<arb::density>(std::move(dens)); }))
        .def(py::init(
            [](arb::density dens, const std::unordered_map<std::string, std::string>& scales) {
                auto s = arb::scaled_mechanism<arb::density>(std::move(dens));
                for (const auto& [k, v]: scales) {
                    s.scale(k, arborio::parse_iexpr_expression(v).unwrap());
                }
                return s;
            }))
        .def(py::init(
            [](arb::density dens, py::kwargs scales) {
                auto s = arb::scaled_mechanism<arb::density>(std::move(dens));
                for (const auto& [k, v]: util::dict_to_map<std::string>(scales)) {
                    s.scale(k, arborio::parse_iexpr_expression(v).unwrap());
                }
                return s;
            }))
        .def(
            "scale",
            [](arb::scaled_mechanism<arb::density>& s, std::string name, const std::string& ex) {
                s.scale(std::move(name), arborio::parse_iexpr_expression(ex).unwrap());
                return s;
            },
            py::arg("name"),
            py::arg("ex"),
            "Add a scaling expression to a parameter.")
        .def("__repr__",
            [](const arb::scaled_mechanism<arb::density>& d) {
                return "<arbor.scaled_mechanism<density> " + scaled_density_desc_str(d) + ">";
            })
        .def("__str__", [](const arb::scaled_mechanism<arb::density>& d) {
            return "<arbor.scaled_mechanism<density> " + scaled_density_desc_str(d) + ">";
        });

    synapse
        .def(py::init([](const std::string& name) {return arb::synapse(name);}))
        .def(py::init([](arb::mechanism_desc mech) {return arb::synapse(mech);}))
        .def(py::init([](const std::string& name, const std::unordered_map<std::string, double>& params) {return arb::synapse(name, params);}))
        .def(py::init([](arb::mechanism_desc mech, const std::unordered_map<std::string, double>& params) {return arb::synapse(mech, params);}))
        .def(py::init([](const std::string& name, py::kwargs parms) {return arb::synapse(name, util::dict_to_map<double>(parms));}))
        .def(py::init([](arb::mechanism_desc mech, py::kwargs params) {return arb::synapse(mech, util::dict_to_map<double>(params));}))
        .def_readonly("mech", &arb::synapse::mech, "The underlying mechanism.")
        .def("__repr__", [](const arb::synapse& s){return "<arbor.synapse " + mechanism_desc_str(s.mech) + ">";})
        .def("__str__", [](const arb::synapse& s){return "<arbor.synapse " + mechanism_desc_str(s.mech) + ">";});

    junction
        .def(py::init([](const std::string& name) {return arb::junction(name);}))
        .def(py::init([](arb::mechanism_desc mech) {return arb::junction(mech);}))
        .def(py::init([](const std::string& name, const std::unordered_map<std::string, double>& params) {return arb::junction(name, params);}))
        .def(py::init([](const std::string& name, py::kwargs parms) {return arb::junction(name, util::dict_to_map<double>(parms));}))
        .def(py::init([](arb::mechanism_desc mech, const std::unordered_map<std::string, double>& params) {return arb::junction(mech, params);}))
        .def(py::init([](arb::mechanism_desc mech, py::kwargs params) {return arb::junction(mech, util::dict_to_map<double>(params));}))
        .def_readonly("mech", &arb::junction::mech, "The underlying mechanism.")
        .def("__repr__", [](const arb::junction& j){return "<arbor.junction " + mechanism_desc_str(j.mech) + ">";})
        .def("__str__", [](const arb::junction& j){return "<arbor.junction " + mechanism_desc_str(j.mech) + ">";});

    i_clamp
        .def(py::init(
                [](const U::quantity& ts,
                   const U::quantity& dur,
                   const U::quantity& cur,
                   const U::quantity& frequency,
                   const U::quantity& phase) {
                    return arb::i_clamp::box(ts, dur, cur, frequency, phase);
                }),
             "tstart"_a, "duration"_a, "current"_a,
             py::kw_only(), py::arg_v("frequency", 0*U::kHz, "0.0*arbor.units.kHz"), py::arg_v("phase", 0*U::rad, "0.0*arbor.units.rad"),
             "Construct finite duration current clamp, constant amplitude")
        .def(py::init(
                [](const U::quantity& cur,
                   const U::quantity& frequency,
                   const U::quantity& phase) {
                    return arb::i_clamp{cur, frequency, phase};
                }),
             "current"_a,
             py::kw_only(), py::arg_v("frequency", 0*U::kHz, "0.0*arbor.units.kHz"), py::arg_v("phase", 0*U::rad, "0.0*arbor.units.rad"),
             "Construct constant amplitude current clamp")
        .def(py::init(
                [](std::vector<std::pair<const U::quantity&, const U::quantity&>> envl,
                   const U::quantity& frequency,
                   const U::quantity& phase) {
                    std::vector<arb::i_clamp::envelope_point> env;
                    for (const auto& [t, a]: envl) env.push_back({t, a});
                    return arb::i_clamp{env, frequency, phase};
                }),
             "envelope"_a,
             py::kw_only(), py::arg_v("frequency", 0*U::kHz, "0.0*arbor.units.kHz"), py::arg_v("phase", 0*U::rad, "0.0*arbor.units.rad"),
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
            return util::pprintf("<arbor.iclamp: frequency {} kHz>", c.frequency);})
        .def("__str__", [](const arb::i_clamp& c) {
            return util::pprintf("<arbor.iclamp: frequency {} kHz>", c.frequency);});

    detector
        .def(py::init(
            [](const U::quantity& thresh) { return arb::threshold_detector{thresh}; }),
            "threshold"_a, "Voltage threshold of spike detector [mV]")
        .def_readonly("threshold", &arb::threshold_detector::threshold, "Voltage threshold of spike detector [mV]")
        .def("__repr__", [](const arb::threshold_detector& d){
            return util::pprintf("<arbor.threshold_detector: threshold {} mV>", d.threshold);})
        .def("__str__", [](const arb::threshold_detector& d){
            return util::pprintf("(threshold_detector {})", d.threshold);});

    ion_data
        .def_readonly("internal_concentration", &arb::cable_cell_ion_data::init_int_concentration,  "Internal concentration.")
        .def_readonly("external_concentration", &arb::cable_cell_ion_data::init_ext_concentration,  "External concentration.")
        .def_readonly("diffusivity",            &arb::cable_cell_ion_data::diffusivity,             "Diffusivity.")
        .def_readonly("reversal_concentration", &arb::cable_cell_ion_data::init_reversal_potential, "Reversal potential.");

    ion_data
        .def_property_readonly("charge",                    [](const ion_settings& s) { return s.charge; },                    "Valence.")
        .def_property_readonly("internal_concentration",    [](const ion_settings& s) { return s.internal_concentration; },    "Internal concentration.")
        .def_property_readonly("external_concentration",    [](const ion_settings& s) { return s.external_concentration; },    "External concentration.")
        .def_property_readonly("diffusivity",               [](const ion_settings& s) { return s.diffusivity; },               "Diffusivity.")
        .def_property_readonly("reversal_potential",        [](const ion_settings& s) { return s.reversal_potential; },        "Reversal potential.")
        .def_property_readonly("reversal_potential_method", [](const ion_settings& s) { return s.reversal_potential_method; }, "Reversal potential method.");

    gprop
        .def(py::init<>())
        .def(py::init<const arb::cable_cell_global_properties&>())
        .def("check", [](const arb::cable_cell_global_properties& props) {
                arb::check_global_properties(props);},
                "Test whether all default parameters and ion species properties have been set.")
        .def_readwrite("coalesce_synapses",  &arb::cable_cell_global_properties::coalesce_synapses,
                "Flag for enabling/disabling linear syanpse coalescing.")
        .def_property("membrane_voltage_limit",
                      [](const arb::cable_cell_global_properties& props) { return props.membrane_voltage_limit_mV; },
                      [](arb::cable_cell_global_properties& props, std::optional<double> u) { props.membrane_voltage_limit_mV = u; })
        // set cable properties
        .def_property_readonly("membrane_potential",
                               [](const arb::cable_cell_global_properties& props) { return props.default_parameters.init_membrane_potential; })
        .def_property_readonly("membrane_capacitance",
                      [](const arb::cable_cell_global_properties& props) { return props.default_parameters.membrane_capacitance; })
        .def_property_readonly("temperature",
                      [](const arb::cable_cell_global_properties& props) { return props.default_parameters.temperature_K; })
        .def_property_readonly("axial_resistivity",
                               [](const arb::cable_cell_global_properties& props) { return props.default_parameters.axial_resistivity; })
        .def("set_property",
            [](arb::cable_cell_global_properties& props,
               optional<U::quantity> Vm, optional<U::quantity> cm,
               optional<U::quantity> rL, optional<U::quantity> tempK) {
                if (Vm) props.default_parameters.init_membrane_potential=Vm.value().value_as(U::mV);
                if (cm) props.default_parameters.membrane_capacitance=cm.value().value_as(U::F/U::m2);
                if (rL) props.default_parameters.axial_resistivity=rL.value().value_as(U::Ohm*U::cm);
                if (tempK) props.default_parameters.temperature_K=tempK.value().value_as(U::Kelvin);
            },
             "Vm"_a=py::none(), "cm"_a=py::none(), "rL"_a=py::none(), "tempK"_a=py::none(),
             "Set global default values for cable and cell properties.\n"
             " * Vm:    initial membrane voltage [mV].\n"
             " * cm:    membrane capacitance [F/m²].\n"
             " * rL:    axial resistivity [Ω·cm].\n"
             " * tempK: temperature [Kelvin].\n"
             "These values can be overridden on specific regions using the paint interface.")
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
                optional<int> valence, optional<U::quantity> int_con,
                optional<U::quantity> ext_con, optional<U::quantity> rev_pot,
                py::object method, optional<U::quantity> diff) {
                 if (!props.ion_species.count(ion) && !valence) {
                     throw std::runtime_error(util::pprintf("New ion species: '{}', missing valence", ion));
                 }
                 if (valence) props.ion_species[ion] = *valence;

                 auto& data = props.default_parameters.ion_data[ion];
                 if (int_con) data.init_int_concentration  = int_con.value().value_as(U::mM);
                 if (ext_con) data.init_ext_concentration  = ext_con.value().value_as(U::mM);
                 if (rev_pot) data.init_reversal_potential = rev_pot.value().value_as(U::mV);
                 if (diff)    data.diffusivity             = diff.value().value_as(U::m2/U::s);

                 if (auto m = maybe_method(method)) {
                     props.default_parameters.reversal_potential_method[ion] = *m;
                 }
             },
             "ion"_a, "valence"_a=py::none(), "int_con"_a=py::none(), "ext_con"_a=py::none(), "rev_pot"_a=py::none(), "method"_a =py::none(), "diff"_a=py::none(),
             "Set the global default properties of ion species named 'ion'.\n"
             " * valence: valence of the ion species [e].\n"
             " * int_con: initial internal concentration [mM].\n"
             " * ext_con: initial external concentration [mM].\n"
             " * rev_pot: reversal potential [mV].\n"
             " * method:  mechanism for calculating reversal potential.\n"
             " * diff:   diffusivity [m^2/s].\n"
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

    decor
        .def(py::init<>())
        .def(py::init<const arb::decor&>())
        // Set cell-wide default values for properties
        .def("set_property",
             [](arb::decor& d,
                optional<U::quantity> Vm, optional<U::quantity> cm,
                optional<U::quantity> rL, optional<U::quantity> tempK) {
                 if (Vm) d.set_default(arb::init_membrane_potential{*Vm});
                 if (cm) d.set_default(arb::membrane_capacitance{*cm});
                 if (rL) d.set_default(arb::axial_resistivity{*rL});
                 if (tempK) d.set_default(arb::temperature{*tempK});
                 return d;
             },
             "Vm"_a=py::none(), "cm"_a=py::none(), "rL"_a=py::none(), "tempK"_a=py::none(),
             "Set default values for cable and cell properties:\n"
             " * Vm:    initial membrane voltage [mV].\n"
             " * cm:    membrane capacitance [F/m²].\n"
             " * rL:    axial resistivity [Ω·cm].\n"
             " * tempK: temperature [Kelvin].\n"
             "These values can be overridden on specific regions using the paint interface.")
        // modify parameters for an ion species.
        .def("set_ion",
             [](arb::decor& d, const char* ion,
                optional<U::quantity> int_con, optional<U::quantity> ext_con,
                optional<U::quantity> rev_pot, py::object method,
                optional<U::quantity> diff) {
                 if (int_con) d.set_default(arb::init_int_concentration{ion, *int_con});
                 if (ext_con) d.set_default(arb::init_ext_concentration{ion, *ext_con});
                 if (rev_pot) d.set_default(arb::init_reversal_potential{ion, *rev_pot});
                 if (diff)    d.set_default(arb::ion_diffusivity{ion, *diff});
                 if (auto m = maybe_method(method)) d.set_default(arb::ion_reversal_potential_method{ion, *m});
                 return d;
             },
             "ion"_a, "int_con"_a=py::none(), "ext_con"_a=py::none(), "rev_pot"_a=py::none(), "method"_a =py::none(), "diff"_a=py::none(),
             "Set the cell-level properties of ion species named 'ion'.\n"
             " * int_con: initial internal concentration [mM].\n"
             " * ext_con: initial external concentration [mM].\n"
             " * rev_pot: reversal potential [mV].\n"
             " * method:  mechanism for calculating reversal potential.\n"
             " * diff:    diffusivity [m^2/s].\n"
             "There are 3 ion species predefined in arbor: 'ca', 'na' and 'k'.\n"
             "If 'ion' in not one of these ions it will be added to the list, making it\n"
             "available to mechanisms. The user has to provide the valence of a previously\n"
             "undefined ion the first time this function is called with it as an argument.\n"
             "Species concentrations and reversal potential can be overridden on\n"
             "specific regions using the paint interface, while the method for calculating\n"
             "reversal potential is global for all compartments in the cell, and can't be\n"
             "overriden locally.")
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
                    result.emplace_back(to_string(k), v, dec.tag_of(t));
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
               optional<paintable_arg> Vm, optional<paintable_arg> cm,
               optional<paintable_arg> rL, optional<paintable_arg> tempK) {
                auto reg = arborio::parse_region_expression(region).unwrap();
                if (Vm) {
                    const auto& [v, s] = value_and_scale(*Vm);
                    dec.paint(reg, arb::init_membrane_potential{v, s});
                }
                if (cm) {
                    const auto& [v, s] = value_and_scale(*cm);
                    dec.paint(reg, arb::membrane_capacitance{v, s});
                }
                if (rL) {
                    const auto& [v, s] = value_and_scale(*rL);
                    dec.paint(reg, arb::axial_resistivity{v, s});
                }
                if (tempK) {
                    const auto& [v, s] = value_and_scale(*tempK);
                    dec.paint(reg, arb::temperature{v, s});
                }
                return dec;
            },
            "region"_a, "Vm"_a=py::none(), "cm"_a=py::none(), "rL"_a=py::none(), "tempK"_a=py::none(),
            "Set cable properties on a region.\n"
             "Set global default values for cable and cell properties.\n"
             " * Vm:    initial membrane voltage [mV].\n"
             " * cm:    membrane capacitance [F/m²].\n"
             " * rL:    axial resistivity [Ω·cm].\n"
             " * tempK: temperature [Kelvin].\n"
             "Each value can be given as a plain quantity or a tuple of (quantity, 'scale') where scale is an iexpr.")
        // Paint ion species initial conditions on a region.
        .def("paint",
            [](arb::decor& dec, const char* region, const char* name,
               optional<paintable_arg> int_con, optional<paintable_arg> ext_con,
               optional<paintable_arg> rev_pot, optional<paintable_arg> diff) {
                auto r = arborio::parse_region_expression(region).unwrap();
                if (int_con) {
                    const auto& [v, s] = value_and_scale(*int_con);
                    dec.paint(r, arb::init_int_concentration{name, v, s});
                }
                if (ext_con) {
                    const auto& [v, s] = value_and_scale(*ext_con);
                    dec.paint(r, arb::init_ext_concentration{name, v, s});
                }
                if (rev_pot) {
                    const auto& [v, s] = value_and_scale(*rev_pot);
                    dec.paint(r, arb::init_reversal_potential{name, v, s});
                }
                if (diff) {
                    const auto& [v, s] = value_and_scale(*diff);
                    dec.paint(r, arb::ion_diffusivity{name, v, s});
                }
                return dec;
            },
            "region"_a, py::kw_only(), "ion"_a, "int_con"_a=py::none(), "ext_con"_a=py::none(), "rev_pot"_a=py::none(), "diff"_a=py::none(),
            "Set ion species properties conditions on a region.\n"
             " * int_con: initial internal concentration [mM].\n"
             " * ext_con: initial external concentration [mM].\n"
             " * rev_pot: reversal potential [mV].\n"
             " * method:  mechanism for calculating reversal potential.\n"
             " * diff:   diffusivity [m^2/s].\n"
             "Each value can be given as a plain quantity or a tuple of (quantity, 'scale') where scale is an iexpr.\n")
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
            py::arg("policy"),
             "A cv_policy used to discretise the cell into compartments for simulation")
        .def("discretization",
            [](arb::decor& dec, const std::string& p) {
                return dec.set_default(arborio::parse_cv_policy_expression(p).unwrap());
            },
            py::arg("policy"),
            "An s-expression string representing a cv_policy used to discretise the "
            "cell into compartments for simulation");

    cable_cell
        .def(py::init(
            [](const arb::morphology& m, const arb::decor& d, const std::optional<label_dict_proxy>& l) {
                if (l) return arb::cable_cell(m, d, l->dict);
                return arb::cable_cell(m, d);
            }),
            "morphology"_a, "decor"_a, "labels"_a=py::none(),
            "Construct with a morphology, decor, and label dictionary.")
        .def(py::init(
            [](const arb::segment_tree& t, const arb::decor& d, const std::optional<label_dict_proxy>& l) {
                if (l) return arb::cable_cell({t}, d, l->dict);
                return arb::cable_cell({t}, d);
            }),
            "segment_tree"_a, "decor"_a, "labels"_a=py::none(),
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
