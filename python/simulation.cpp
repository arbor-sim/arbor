#include <memory>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <arbor/common_types.hpp>
#include <arbor/sampling.hpp>
#include <arbor/simulation.hpp>

#include "context.hpp"
#include "error.hpp"
#include "pyarb.hpp"
#include "recipe.hpp"
#include "schedule.hpp"

namespace py = pybind11;

namespace pyarb {

// Argument type for simulation_shim::record() (see below).

enum class spike_recording {
    off, local, all
};

// Wraps an arb::simulation object and in addition manages a set of
// sampler callbacks for retrieving probe data.

class simulation_shim {
    std::unique_ptr<arb::simulation> sim_;
    std::vector<arb::spike> spike_record_;
    pyarb_global_ptr global_ptr_;

    using sample_recorder_ptr = std::unique_ptr<sample_recorder>;
    using sample_recorder_vec = std::vector<sample_recorder_ptr>;

    // These are only used as the target sampler of a single probe id.
    struct sampler_callback {
        std::shared_ptr<sample_recorder_vec> recorders;

        void operator()(arb::probe_metadata pm, std::size_t n_record, const arb::sample_record* records) {
            recorders->at(pm.index)->record(pm.meta, n_record, records);
        }

        py::list samples() const {
            std::size_t size = recorders->size();
            py::list result(size);

            for (std::size_t i = 0; i<size; ++i) {
                result[i] = py::make_tuple(recorders->at(i)->samples(), recorders->at(i)->meta());
            }
            return result;
        }
    };

    std::unordered_map<arb::sampler_association_handle, sampler_callback> sampler_map_;

public:
    simulation_shim(std::shared_ptr<py_recipe>& rec, const context_shim& ctx, const arb::domain_decomposition& decomp, std::uint64_t seed, pyarb_global_ptr global_ptr):
        global_ptr_(global_ptr)
    {
        try {
            sim_.reset(new arb::simulation(py_recipe_shim(rec), ctx.context, decomp, seed));
        }
        catch (...) {
            py_reset_and_throw();
            throw;
        }
    }

    void update(std::shared_ptr<py_recipe>& rec) {
        try {
            sim_->update(py_recipe_shim(rec));
        }
        catch (...) {
            py_reset_and_throw();
            throw;
        }
    }

    void reset() {
        sim_->reset();
        spike_record_.clear();
        for (auto&& [handle, cb]: sampler_map_) {
            for (auto& rec: *cb.recorders) {
                rec->reset();
            }
        }
    }

    void clear_samplers() {
        spike_record_.clear();
        for (auto&& [handle, cb]: sampler_map_) {
            for (auto& rec: *cb.recorders) {
                rec->reset();
            }
        }
    }

    arb::time_type run(arb::time_type tfinal, arb::time_type dt) {
        return sim_->run(tfinal, dt);
    }

    void set_binning_policy(arb::binning_kind policy, arb::time_type bin_interval) {
        sim_->set_binning_policy(policy, bin_interval);
    }

    void record(spike_recording policy) {
        auto spike_recorder = [this](const std::vector<arb::spike>& spikes) {
            auto old_size = spike_record_.size();
            // Append the new spikes to the end of the spike record.
            spike_record_.insert(spike_record_.end(), spikes.begin(), spikes.end());
            // Sort the newly appended spikes.
            std::sort(spike_record_.begin()+old_size, spike_record_.end(),
                    [](const auto& lhs, const auto& rhs) {
                        return std::tie(lhs.time, lhs.source.gid, lhs.source.index)<std::tie(rhs.time, rhs.source.gid, rhs.source.index);
                    });
        };

        switch (policy) {
        case spike_recording::off:
            sim_->set_global_spike_callback();
            sim_->set_local_spike_callback();
            break;
        case spike_recording::local:
            sim_->set_global_spike_callback();
            sim_->set_local_spike_callback(spike_recorder);
            break;
        case spike_recording::all:
            sim_->set_global_spike_callback(spike_recorder);
            sim_->set_local_spike_callback();
            break;
        }
    }

    py::object spikes() const {
        return py::array_t<arb::spike>(py::ssize_t(spike_record_.size()), spike_record_.data());
    }

    py::list get_probe_metadata(arb::cell_member_type probeset_id) const {
        py::list result;
        for (auto&& pm: sim_->get_probe_metadata(probeset_id)) {
             result.append(global_ptr_->probe_meta_converters.convert(pm.meta));
        }
        return result;
    }

    arb::sampler_association_handle sample(arb::cell_member_type probeset_id, const pyarb::schedule_shim_base& sched, arb::sampling_policy policy) {
        std::shared_ptr<sample_recorder_vec> recorders{new sample_recorder_vec};

        for (const arb::probe_metadata& pm: sim_->get_probe_metadata(probeset_id)) {
            recorders->push_back(global_ptr_->recorder_factories.make_recorder(pm.meta));
        }

        // Constructed callbacks are passed to the underlying simulator object, _and_ a copy
        // is kept in sampler_map_; the two copies share the same recorder data.

        sampler_callback cb{std::move(recorders)};
        auto sah = sim_->add_sampler(arb::one_probe(probeset_id), sched.schedule(), cb, policy);
        sampler_map_.insert({sah, cb});

        return sah;
    }

    void remove_sampler(arb::sampler_association_handle sah) {
        sim_->remove_sampler(sah);
        sampler_map_.erase(sah);
    }

    void remove_all_samplers() {
        sim_->remove_all_samplers();
        sampler_map_.clear();
    }

    py::list samples(arb::sampler_association_handle sah) {
        if (auto iter = sampler_map_.find(sah); iter!=sampler_map_.end()) {
            return iter->second.samples();
        }
        else {
            return py::list{};
        }
    }

    void progress_banner() {
        sim_->set_epoch_callback(arb::epoch_progress_bar());
    }
};

void register_simulation(pybind11::module& m, pyarb_global_ptr global_ptr) {
    using namespace pybind11::literals;

    py::enum_<arb::sampling_policy>(m, "sampling_policy")
       .value("lax", arb::sampling_policy::lax)
       .value("exact", arb::sampling_policy::exact);

    py::enum_<spike_recording>(m, "spike_recording")
       .value("off", spike_recording::off)
       .value("local", spike_recording::local)
       .value("all", spike_recording::all);

    // Simulation
    py::class_<simulation_shim> simulation(m, "simulation",
        "The executable form of a model.\n"
        "A simulation is constructed from a recipe, and then used to update and monitor model state.");
    simulation
        // A custom constructor that wraps a python recipe with arb::py_recipe_shim
        // before forwarding it to the arb::recipe constructor.
        .def(pybind11::init(
                 [global_ptr](std::shared_ptr<py_recipe>& rec,
                              const std::shared_ptr<context_shim>& ctx_,
                              const std::optional<arb::domain_decomposition>& decomp,
                              std::uint64_t seed) {
                try {
                    auto ctx = ctx_ ? ctx_ : std::make_shared<context_shim>(arb::make_context());
                    auto dec = decomp.value_or(arb::partition_load_balance(py_recipe_shim(rec), ctx->context));
                    return new simulation_shim(rec, *ctx, dec, seed, global_ptr);
                }
                catch (...) {
                    py_reset_and_throw();
                    throw;
                }
            }),
            // Release the python gil, so that callbacks into the python recipe don't deadlock.
            pybind11::call_guard<pybind11::gil_scoped_release>(),
            "Initialize the model described by a recipe, with cells and network distributed\n"
            "according to the domain decomposition and computational resources described by a context.",
             "recipe"_a,
             pybind11::arg_v("context", pybind11::none(), "Execution context"),
             pybind11::arg_v("domains", pybind11::none(), "Domain decomposition"),
             pybind11::arg_v("seed", 0u, "Random number generator seed"))
        .def("update", &simulation_shim::update,
             pybind11::call_guard<pybind11::gil_scoped_release>(),
             "Rebuild the connection table from recipe::connections_on and the event"
             "generators based on recipe::event_generators.",
             "recipe"_a)
        .def("reset", &simulation_shim::reset,
            pybind11::call_guard<pybind11::gil_scoped_release>(),
            "Reset the state of the simulation to its initial state.")
        .def("clear_samplers", &simulation_shim::clear_samplers,
             pybind11::call_guard<pybind11::gil_scoped_release>(),
             "Clearing spike and sample information. restoring memory")
        .def("run", &simulation_shim::run,
            pybind11::call_guard<pybind11::gil_scoped_release>(),
            "Run the simulation from current simulation time to tfinal [ms], with maximum time step size dt [ms].",
            "tfinal"_a, "dt"_a=0.025)
        .def("set_binning_policy", &simulation_shim::set_binning_policy,
            "Set the binning policy for event delivery, and the binning time interval if applicable [ms].",
             "policy"_a, "bin_interval"_a)
        .def("record", &simulation_shim::record,
            "Disable or enable local or global spike recording.")
        .def("spikes", &simulation_shim::spikes,
            "Retrieve recorded spikes as numpy array.")
        .def("probe_metadata", &simulation_shim::get_probe_metadata,
            "Retrieve metadata associated with given probe id.",
            "probeset_id"_a)
        .def("sample", &simulation_shim::sample,
            "Record data from probes with given probeset_id according to supplied schedule.\n"
            "Returns handle for retrieving data or removing the sampling.",
            "probeset_id"_a, "schedule"_a, "policy"_a = arb::sampling_policy::lax)
        .def("samples", &simulation_shim::samples,
            "Retrieve sample data as a list, one element per probe associated with the query.",
            "handle"_a)
        .def("remove_sampler", &simulation_shim::remove_sampler,
            "Remove sampling associated with the given handle.",
            "handle"_a)
        .def("remove_all_samplers", &simulation_shim::remove_sampler,
            "Remove all sampling on the simulatr.")
        .def("progress_banner", &simulation_shim::progress_banner,
            "Show a text progress bar during simulation.");

}

} // namespace pyarb
