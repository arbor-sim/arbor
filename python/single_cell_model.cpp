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

struct probe_site {
    arb::mlocation site;
    double frequency;     // [Hz]
};

struct trace {
    std::string variable;
    arb::mlocation loc;
    std::vector<arb::time_type> t;
    std::vector<double> v;
};

struct trace_callback {
    trace& trace_;

    trace_callback(trace& t): trace_(t) {}

    void operator()(arb::cell_member_type probe_id, arb::probe_tag tag, std::size_t n, const arb::sample_record* recs) {
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

// Used internally by the single cell model.
struct single_cell_recipe: arb::recipe {
    const arb::cable_cell& cell_;

    // todo: make these references
    const std::vector<probe_site>& probes_;
    const std::vector<arb::event_generator>& generators_;

    single_cell_recipe(
            const arb::cable_cell& c,
            const std::vector<probe_site>& probes,
            const std::vector<arb::event_generator>& gens):
        cell_(c), probes_(probes), generators_(gens)
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
        return generators_;
    }

    // probes

    virtual arb::cell_size_type num_probes(arb::cell_gid_type)  const override {
        return probes_.size();
    }

    virtual arb::probe_info get_probe(arb::cell_member_type probe_id) const override {
        if (probe_id.gid || probe_id.index>=probes_.size()) {
            throw arb::bad_probe_id(probe_id);
        }

        // For now only voltage can be measured.
        auto kind = arb::cell_probe_address::membrane_voltage;
        const auto& loc = probes_[probe_id.index].site;
        return arb::probe_info{probe_id, kind, arb::cell_probe_address{loc, kind}};
    }

    // gap junctions

    virtual arb::cell_size_type num_gap_junction_sites(arb::cell_gid_type gid)  const override {
        return 0; // no gap junctions on a single cell model
    }

    virtual std::vector<arb::gap_junction_connection> gap_junctions_on(arb::cell_gid_type) const override {
        return {}; // no gap junctions on a single cell model
    }

    virtual arb::util::any get_global_properties(arb::cell_kind) const override {
        // TODO: make this setable
        arb::cable_cell_global_properties gprop;
        gprop.default_parameters = arb::neuron_parameter_defaults;
        return gprop;
    }
};

class single_cell_model {
    arb::cable_cell cell_;
    arb::context ctx_;
    bool run_ = false;

    std::vector<probe_site> probes_;
    std::vector<arb::event_generator> generators_;
    std::unique_ptr<arb::simulation> sim_;
    std::vector<double> spike_times_;
    // Create one trace for each probe.
    std::vector<trace> traces_;

public:
    single_cell_model(arb::cable_cell c):
        cell_(std::move(c)), ctx_(arb::make_context()) {}

    // example use:
    //      m.probe('voltage', arbor.location(2,0.5))
    void probe(const std::string& what, const arb::mlocation& where, double frequency) {
        if (what != "voltage") {
            throw pyarb_error(
                util::pprintf("{} does not name a valid variable to trace (currently only 'voltage' is supported)", what));
        }
        if (frequency<=0) {
            throw pyarb_error(
                util::pprintf("sampling frequency is not greater than zero", what));
        }
        if (where.branch>=cell_.morphology().num_branches()) {
            throw pyarb_error(
                util::pprintf("invalid location", what));
        }
        probes_.push_back({where, frequency});
    }

    void run(double tfinal) {

        single_cell_recipe rec(cell_, probes_, generators_);

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

    pybind11::class_<trace> tr(m, "trace");
    tr
        .def_readonly("variable", &trace::variable)
        .def_readonly("location", &trace::loc)
        .def_readonly("time",    &trace::t)
        .def_readonly("value",   &trace::v);

    pybind11::class_<single_cell_model> model(m, "single_cell_model",
        "Wrapper for easily building and running single cell models.");

    model
        .def(pybind11::init<arb::cable_cell>(),
            "cell"_a, "Build a single cell model.")
        .def("run", &single_cell_model::run, "tfinal"_a, "run model from t=0 to t=tfinal ms")
        .def("probe", &single_cell_model::probe, "what"_a, "where"_a, "frequency"_a)
        .def_property_readonly("spikes", [](const single_cell_model& m) {return m.spike_times();}, "spike times (ms)")
        .def_property_readonly("traces", [](const single_cell_model& m) {return m.traces();}, "traces from probes")
        .def("__repr__", [](const single_cell_model&){return "<arbor.single_cell_model>";})
        .def("__str__",  [](const single_cell_model&){return "<arbor.single_cell_model>";});
}

} // namespace pyarb

