#include <any>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <arborio/label_parse.hpp>

#include <arbor/cable_cell.hpp>
#include <arbor/load_balance.hpp>
#include <arbor/recipe.hpp>
#include <arbor/simulation.hpp>
#include <arbor/util/any_cast.hpp>

#include "event_generator.hpp"
#include "error.hpp"
#include "strprintf.hpp"
#include "proxy.hpp"

using arb::util::any_cast;

namespace pyarb {

//
// Implementation of sampling infrastructure for Python single cell models.
//
// Note that only voltage sampling is currently supported, which simplifies the
// code somewhat.
//

// Stores the location and sampling frequency for a probe in a single cell model.
struct probe_site {
    arb::locset locset;     // Location of sample on morphology.
    double frequency;       // Sampling frequency [kHz].
    arb::cell_tag_type tag; // Tag = unique name
};

// Stores a single trace, which can be queried and viewed by the user at the end
// of a simulation run.
struct trace {
    std::string variable;           // Name of the variable being recorded.
    arb::mlocation loc;             // Sample site on morphology.
    std::vector<arb::time_type> t;  // Sample times [ms].
    std::vector<double> v;          // Sample values [units specific to sample variable].
};

// Callback provided to sampling API that records into a trace variable.
struct trace_callback {
    std::vector<trace>& traces_;
    const std::unordered_map<arb::mlocation, size_t>& locmap_;


    trace_callback(std::vector<trace>& ts, const std::unordered_map<arb::mlocation, size_t>& ls): traces_(ts), locmap_(ls) {}

    void operator()(arb::probe_metadata md, std::size_t n, const arb::sample_record* recs) {
        // Push each (time, value) pair from the last epoch into trace_.
        auto* loc = any_cast<const arb::mlocation*>(md.meta);
        if (locmap_.count(*loc)) {
            auto& trace = traces_[locmap_.at(*loc)];
            for (std::size_t i=0; i<n; ++i) {
                if (auto p = any_cast<const double*>(recs[i].data)) {
                    trace.t.push_back(recs[i].time);
                    trace.v.push_back(*p);
                }
                else {
                    throw std::runtime_error("unexpected sample type");
                }
            }
        }
    }
};

// Used internally by the single cell model to expose model information to the
// arb::simulation API when a model is instantiated.
// Model descriptors, i.e. the cable_cell and probes, are instantiated
// in the single_cell_model by user calls. The recipe is generated lazily, just
// before simulation construction, so the recipe can use const references to all
// of the model descriptors.
struct single_cell_recipe: arb::recipe {
    const arb::cable_cell& cell_;
    const std::vector<probe_site>& probes_;
    const arb::cable_cell_global_properties& gprop_;
    const std::vector<arb::event_generator>& event_generators_;

    single_cell_recipe(
            const arb::cable_cell& c,
            const std::vector<probe_site>& probes,
            const arb::cable_cell_global_properties& props,
            const std::vector<arb::event_generator>& event_generators):
        cell_(c), probes_(probes), gprop_(props), event_generators_(event_generators)
    {}

    virtual arb::cell_size_type num_cells() const override {
        return 1;
    }

    virtual arb::util::unique_any get_cell_description(arb::cell_gid_type gid) const override {
        return cell_;
    }

    virtual arb::cell_kind get_cell_kind(arb::cell_gid_type gid) const override {
        return arb::cell_kind::cable;
    }

    // connections and event generators

    virtual std::vector<arb::cell_connection> connections_on(arb::cell_gid_type gid) const override {
        return {};
    }

    virtual std::vector<arb::event_generator> event_generators(arb::cell_gid_type) const override {
        return event_generators_;
    }

    // probes

    virtual std::vector<arb::probe_info> get_probes(arb::cell_gid_type gid) const override {
        // For now only voltage can be selected for measurement.
        std::vector<arb::probe_info> pinfo;
        for (auto& p: probes_) pinfo.push_back({arb::cable_probe_membrane_voltage{p.locset}, p.tag});
        return pinfo;
    }

    // gap junctions

    virtual std::vector<arb::gap_junction_connection> gap_junctions_on(arb::cell_gid_type) const override {
        return {}; // No gap junctions on a single cell model.
    }

    virtual std::any get_global_properties(arb::cell_kind kind) const override {
        return gprop_;
    }
};

class single_cell_model {
    arb::cable_cell cell_;
    arb::context ctx_;
    bool run_ = false;

    std::vector<probe_site> probes_;
    std::vector<arb::event_generator> event_generators_;
    std::unique_ptr<arb::simulation> sim_;
    std::vector<double> spike_times_;
    std::vector<trace> traces_;
    std::unordered_map<arb::mlocation, size_t> locmap_;

public:
    arb::cable_cell_global_properties gprop;

    single_cell_model(arb::cable_cell c):
        cell_(std::move(c)), ctx_(arb::make_context()) {
        gprop.default_parameters = arb::neuron_parameter_defaults;
        gprop.catalogue = arb::global_default_catalogue();
    }

    // example use:
    //      m.probe('voltage', arbor.location(2,0.5))
    //      m.probe('voltage', '(location 2 0.5)')
    //      m.probe('voltage', 'term')

    void probe(const std::string& what, const arb::locset& where, const arb::cell_tag_type& tag, double frequency) {
        if (what != "voltage") {
            throw pyarb_error(
                util::pprintf("{} does not name a valid variable to trace (currently only 'voltage' is supported)", what));
        }
        if (frequency<=0) {
            throw pyarb_error(
                util::pprintf("sampling frequency is not greater than zero", what));
        }
        probes_.push_back({where, frequency, tag});
    }

    void event_generator(const arb::event_generator& event_generator) {
        event_generators_.push_back(event_generator);
    }

    void run(double tfinal, double dt) {
        single_cell_recipe rec(cell_, probes_, gprop, event_generators_);

        auto domdec = arb::partition_load_balance(rec, ctx_);

        sim_ = std::make_unique<arb::simulation>(rec, ctx_, domdec);

        // Add probes
        for (arb::cell_lid_type i=0; i<probes_.size(); ++i) {
            const auto& p = probes_[i];
            auto sched = arb::regular_schedule(1.0/p.frequency);
            for (const auto& site: cell_.concrete_locset(p.locset)) {
                locmap_.insert({site, traces_.size()});
                traces_.push_back({"voltage", site, {}, {}});
            }
            sim_->add_sampler(arb::one_probe({0, p.tag}), sched, trace_callback(traces_, locmap_));
        }

        // Set callback that records spike times.
        sim_->set_global_spike_callback(
            [this](const std::vector<arb::spike>& spikes) {
                for (auto& s: spikes) {
                    spike_times_.push_back(s.time);
                }
            });

        sim_->run(tfinal, dt);

        run_ = true;
    }

    const arb::cable_cell& cable_cell() const {
        return cell_;
    }

    const std::vector<double>& spike_times() const {
        return spike_times_;
    }

    const std::vector<trace>& traces() const { return traces_; }
};

void register_single_cell(pybind11::module& m) {
    using namespace pybind11::literals;
    pybind11::class_<trace> tr(m, "trace", "Values and meta-data for a sample-trace on a single cell model.");
    tr
        .def_readonly("variable", &trace::variable, "Name of the variable being recorded.")
        .def_readonly("location", &trace::loc, "Location on cell morphology.")
        .def_readonly("time",    &trace::t, "Time stamps of samples [ms].")
        .def_readonly("value",   &trace::v, "Sample values.")
        .def("__str__", [](const trace& tr) {return util::pprintf("(trace \"{}\" {})", tr.variable, tr.loc);})
        .def("__repr__", [](const trace& tr) {return util::pprintf("(trace \"{}\" {})", tr.variable, tr.loc);});

    pybind11::class_<single_cell_model> model(m, "single_cell_model",
        "Wrapper for simplified description, and execution, of single cell models.");
    model
        .def(pybind11::init([](const arb::segment_tree& m,
                               const arb::decor& d,
                               const label_dict_proxy& l) -> single_cell_model {
            return single_cell_model(arb::cable_cell({m}, d, l.dict));
        }),
             "tree"_a, "decor"_a, "labels"_a=arb::decor{},
             "Build single cell model from cable cell components")
        .def(pybind11::init([](const arb::morphology& m,
                               const arb::decor& d,
                               const label_dict_proxy& l) -> single_cell_model {
            return single_cell_model(arb::cable_cell(m, d, l.dict));
        }),
             "morph"_a, "decor"_a, "labels"_a=arb::decor{},
             "Build single cell model from cable cell components")
        .def(pybind11::init<arb::cable_cell>(),
            "cell"_a, "Initialise a single cell model for a cable cell.")
        .def("run",
             &single_cell_model::run,
             "tfinal"_a,
             "dt"_a = 0.025,
             "Run model from t=0 to t=tfinal ms.")
        .def("probe",
             [](single_cell_model& m, const char* what, const char* where, const char* tag, double frequency) {
                 m.probe(what, arborio::parse_locset_expression(where).unwrap(), tag, frequency);},
             "what"_a, "where"_a, "tag"_a, "frequency"_a,
             "Sample a variable on the cell.\n"
             " what:      Name of the variable to record (currently only 'voltage').\n"
             " where:     Location on cell morphology at which to sample the variable.\n"
             " tag:       Unique name for this probe.\n"
             " frequency: The target frequency at which to sample [kHz].")
        .def("probe",
             [](single_cell_model& m, const char* what, const arb::mlocation& where, const char* tag, double frequency) {
                 m.probe(what, where, tag, frequency);},
             "what"_a, "where"_a, "tag"_a, "frequency"_a,
             "Sample a variable on the cell.\n"
             " what:      Name of the variable to record (currently only 'voltage').\n"
             " where:     Location on cell morphology at which to sample the variable.\n"
             " tag:       Unique name for this probe.\n"
             " frequency: The target frequency at which to sample [kHz].")
        .def("event_generator",
            [](single_cell_model& m, const pyarb::event_generator_shim& event_generator) {
                m.event_generator(arb::event_generator(
                    event_generator.target, event_generator.weight, event_generator.time_sched));},
            "event_generator"_a,
            "Register an event generator.\n"
            " event_generator: An Arbor event generator.")
        .def_property_readonly("cable_cell",
            [](const single_cell_model& m) {
                return m.cable_cell();}, "The cable cell held by this model.")
        .def_property_readonly("spikes",
            [](const single_cell_model& m) {
                return m.spike_times();}, "Holds spike times [ms] after a call to run().")
        .def_property_readonly("traces",
            [](const single_cell_model& m) {
                return m.traces();},
            "Holds sample traces after a call to run().")
        .def_readwrite("properties", &single_cell_model::gprop, "Global properties.")
        .def("__repr__", [](const single_cell_model&){return "<arbor.single_cell_model>";})
        .def("__str__",  [](const single_cell_model&){return "<arbor.single_cell_model>";});
}

} // namespace pyarb

