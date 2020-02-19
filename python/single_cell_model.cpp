#include <string>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <arbor/cable_cell.hpp>
#include <arbor/load_balance.hpp>
#include <arbor/recipe.hpp>
#include <arbor/simulation.hpp>

#include "error.hpp"

namespace pyarb {

//
// Implementation of sampling infrastructure for Python single cell models.
//
// Note that only voltage sampling is currently supported, which simplifies the
// code somewhat.
//

// Stores the location and sampling frequency for a probe in a single cell model.
struct probe_site {
    arb::mlocation site;  // Location of sample on morphology.
    double frequency;     // Sampling frequency [Hz].
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
    trace& trace_;

    trace_callback(trace& t): trace_(t) {}

    void operator()(arb::cell_member_type probe_id, arb::probe_tag tag, std::size_t n, const arb::sample_record* recs) {
        // Push each (time, value) pair from the last epoch into trace_.
        for (std::size_t i=0; i<n; ++i) {
            if (auto p = arb::util::any_cast<const double*>(recs[i].data)) {
                trace_.t.push_back(recs[i].time);
                trace_.v.push_back(*p);
            }
            else {
                throw std::runtime_error("unexpected sample type in simple_sampler");
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

    single_cell_recipe(
            const arb::cable_cell& c,
            const std::vector<probe_site>& probes,
            const arb::cable_cell_global_properties& props):
        cell_(c), probes_(probes), gprop_(props)
    {}

    virtual arb::cell_size_type num_cells() const override {
        return 1;
    }

    virtual arb::util::unique_any get_cell_description(arb::cell_gid_type gid) const override {
        return cell_;
    }

    virtual arb::cell_kind get_cell_kind(arb::cell_gid_type) const override {
        return arb::cell_kind::cable;
    }

    virtual arb::cell_size_type num_sources(arb::cell_gid_type) const override {
        return cell_.detectors().size();
    }

    // synapses, connections and event generators

    virtual arb::cell_size_type num_targets(arb::cell_gid_type) const override {
        return cell_.synapses().size();
    }

    virtual std::vector<arb::cell_connection> connections_on(arb::cell_gid_type) const override {
        return {}; // no connections on a single cell model
    }

    virtual std::vector<arb::event_generator> event_generators(arb::cell_gid_type) const override {
        return {};
    }

    // probes

    virtual arb::cell_size_type num_probes(arb::cell_gid_type)  const override {
        return probes_.size();
    }

    virtual arb::probe_info get_probe(arb::cell_member_type probe_id) const override {
        // Test that a valid probe site is requested.
        if (probe_id.gid || probe_id.index>=probes_.size()) {
            throw arb::bad_probe_id(probe_id);
        }

        // For now only voltage can be selected for measurement.
        auto kind = arb::cell_probe_address::membrane_voltage;
        const auto& loc = probes_[probe_id.index].site;
        return arb::probe_info{probe_id, kind, arb::cell_probe_address{loc, kind}};
    }

    // gap junctions

    virtual arb::cell_size_type num_gap_junction_sites(arb::cell_gid_type gid)  const override {
        return 0; // No gap junctions on a single cell model.
    }

    virtual std::vector<arb::gap_junction_connection> gap_junctions_on(arb::cell_gid_type) const override {
        return {}; // No gap junctions on a single cell model.
    }

    virtual arb::util::any get_global_properties(arb::cell_kind) const override {
        return gprop_;
    }
};

class single_cell_model {
    arb::cable_cell cell_;
    arb::context ctx_;
    bool run_ = false;
    arb::cable_cell_global_properties gprop_;

    std::vector<probe_site> probes_;
    std::unique_ptr<arb::simulation> sim_;
    std::vector<double> spike_times_;
    // Create one trace for each probe.
    std::vector<trace> traces_;

public:
    single_cell_model(arb::cable_cell c):
        cell_(std::move(c)), ctx_(arb::make_context())
    {
        gprop_.default_parameters = arb::neuron_parameter_defaults;
    }

    // example use:
    //      m.probe('voltage', arbor.location(2,0.5))
    //      m.probe('voltage', '(location 2 0.5)')
    //      m.probe('voltage', 'term')

    void probe(const std::string& what, const arb::locset& where, double frequency) {
        if (what != "voltage") {
            throw pyarb_error(
                util::pprintf("{} does not name a valid variable to trace (currently only 'voltage' is supported)", what));
        }
        if (frequency<=0) {
            throw pyarb_error(
                util::pprintf("sampling frequency is not greater than zero", what));
        }
        for (auto& l: cell_.concrete_locset(where)) {
            probes_.push_back({l, frequency});
        }
    }

    void add_ion(const std::string& ion, double valence, double int_con, double ext_con, double rev_pot) {
        gprop_.add_ion(ion, valence, int_con, ext_con, rev_pot);
    }

    void run(double tfinal) {
        single_cell_recipe rec(cell_, probes_, gprop_);

        auto domdec = arb::partition_load_balance(rec, ctx_);

        sim_ = std::make_unique<arb::simulation>(rec, domdec, ctx_);

        // Create one trace for each probe.
        traces_.reserve(probes_.size());

        // Add probes
        for (arb::cell_lid_type i=0; i<probes_.size(); ++i) {
            const auto& p = probes_[i];

            traces_.push_back({"voltage", p.site, {}, {}});

            auto sched = arb::regular_schedule(1000./p.frequency);

            // Now attach the sampler at probe site, with sampling schedule sched, writing to voltage
            sim_->add_sampler(arb::one_probe({0,i}), sched, trace_callback(traces_[i]));
        }

        // Set callback that records spike times.
        sim_->set_global_spike_callback(
            [this](const std::vector<arb::spike>& spikes) {
                for (auto& s: spikes) {
                    spike_times_.push_back(s.time);
                }
            });

        sim_->run(tfinal, 0.025);

        run_ = true;
    }

    const std::vector<double>& spike_times() const {
        return spike_times_;
    }

    const std::vector<trace>& traces() const {
        return traces_;
    }
};

void register_single_cell(pybind11::module& m) {
    using namespace pybind11::literals;

    pybind11::class_<trace> tr(m, "trace", "Values and meta-data for a sample-trace on a single cell model.");
    tr
        .def_readonly("variable", &trace::variable, "Name of the variable being recorded.")
        .def_readonly("location", &trace::loc, "Location on cell morphology.")
        .def_readonly("time",    &trace::t, "Time stamps of samples [ms].")
        .def_readonly("value",   &trace::v, "Sample values.");

    pybind11::class_<single_cell_model> model(m, "single_cell_model",
        "Wrapper for simplified description, and execution, of single cell models.");

    model
        .def(pybind11::init<arb::cable_cell>(),
            "cell"_a, "Initialise a single cell model for a cable cell.")
        .def("run", &single_cell_model::run, "tfinal"_a, "Run model from t=0 to t=tfinal ms.")
        .def("probe",
            [](single_cell_model& m, const char* what, const char* where, double frequency) {
                m.probe(what, where, frequency);},
            "what"_a, "where"_a, "frequency"_a,
            "Sample a variable on the cell.\n"
            " what:      Name of the variable to record (currently only 'voltage').\n"
            " where:     Location on cell morphology at which to sample the variable.\n"
            " frequency: The target frequency at which to sample [Hz].")
        .def("probe",
            [](single_cell_model& m, const char* what, const arb::mlocation& where, double frequency) {
                m.probe(what, where, frequency);},
            "what"_a, "where"_a, "frequency"_a,
            "Sample a variable on the cell.\n"
            " what:      Name of the variable to record (currently only 'voltage').\n"
            " where:     Location on cell morphology at which to sample the variable.\n"
            " frequency: The target frequency at which to sample [Hz].")
        .def("add_ion", &single_cell_model::add_ion,
            "ion"_a, "valence"_a, "int_con"_a, "ext_con"_a, "rev_pot"_a,
            "Add a new ion species to the model.\n"
            " ion: name of the ion species.\n"
            " valence: valence of the ion species.\n"
            " int_con: initial internal concentration [mM].\n"
            " ext_con: initial external concentration [mM].\n"
            " rev_pot: reversal potential [mV].")
        .def_property_readonly("spikes",
            [](const single_cell_model& m) {
                return m.spike_times();}, "Holds spike times [ms] after a call to run().")
        .def_property_readonly("traces",
            [](const single_cell_model& m) {
                return m.traces();}, "Holds sample traces after a call to run().")
        .def("__repr__", [](const single_cell_model&){return "<arbor.single_cell_model>";})
        .def("__str__",  [](const single_cell_model&){return "<arbor.single_cell_model>";});
}

} // namespace pyarb

