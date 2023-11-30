#include <memory>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include <arbor/common_types.hpp>
#include <arbor/sampling.hpp>
#include <arbor/simulation.hpp>

#include "context.hpp"
#include "error.hpp"
#include "pyarb.hpp"
#include "recipe.hpp"
#include "schedule.hpp"

#include <arborio/json_serdes.hpp>
#include <arbor/serdes.hpp>

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


    std::string serialize() {
        arborio::json_serdes writer;
        arb::serializer serializer{writer};
        arb::serialize(serializer, "sim", *sim_);
        return writer.get_json().dump();
    }

    void deserialize(const std::string& data) {
        arborio::json_serdes writer;
        writer.set_json(nlohmann::json::parse(data));
        arb::serializer serializer{writer};
        arb::deserialize(serializer, "sim", *sim_);
    }

    void set_remote_spike_filter(const arb::spike_predicate& p) { return sim_->set_remote_spike_filter(p); }

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

    arb::time_type run(const arb::units::quantity& tfinal, const arb::units::quantity& dt) {
        return sim_->run(tfinal, dt);
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

    py::list get_probe_metadata(const arb::cell_address_type& probeset_id) const {
        py::list result;
        for (auto&& pm: sim_->get_probe_metadata(probeset_id)) {
             result.append(global_ptr_->probe_meta_converters.convert(pm.meta));
        }
        return result;
    }

    arb::sampler_association_handle sample(const arb::cell_address_type& probeset_id, const pyarb::schedule_shim_base& sched) {
        std::shared_ptr<sample_recorder_vec> recorders{new sample_recorder_vec};

        for (const arb::probe_metadata& pm: sim_->get_probe_metadata(probeset_id)) {
            recorders->push_back(global_ptr_->recorder_factories.make_recorder(pm.meta));
        }

        // Constructed callbacks are passed to the underlying simulator object, _and_ a copy
        // is kept in sampler_map_; the two copies share the same recorder data.

        sampler_callback cb{std::move(recorders)};
        auto sah = sim_->add_sampler(arb::one_probe(probeset_id), sched.schedule(), cb);
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

void register_simulation(py::module& m, pyarb_global_ptr global_ptr) {
    using namespace py::literals;

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
        .def(py::init(
                 [global_ptr](std::shared_ptr<py_recipe>& rec,
                              std::optional<std::shared_ptr<context_shim>> ctx_,
                              const std::optional<arb::domain_decomposition>& decomp,
                              std::uint64_t seed) {
                     try {
                         auto ctx = ctx_ ? ctx_.value() : std::make_shared<context_shim>(make_context_shim());
                         auto dec = decomp.value_or(arb::partition_load_balance(py_recipe_shim(rec), ctx->context));
                         return new simulation_shim(rec, *ctx, dec, seed, global_ptr);
                     }
                     catch (...) {
                         py_reset_and_throw();
                         throw;
                     }
                 }),
             // Release the python gil, so that callbacks into the python recipe don't deadlock.
             py::call_guard<py::gil_scoped_release>(),
             "recipe"_a,
             "context"_a=py::none(),
             "domains"_a=py::none(),
             "seed"_a=0u,
             "Initialize the model described by a recipe, with cells and network distributed\n"
             "according to the domain decomposition and computational resources described by a\n"
             "context. Initialize PRNG using seed")
        .def("set_remote_spike_filter",
             &simulation_shim::set_remote_spike_filter,
             "pred"_a,
             "Add a callback to filter spikes going out over external connections. `pred` is"
             "a callable on the `spike` type. **Caution**: This will be extremely slow; use C++ "
             "if you want to make use of this.")
        .def("update", &simulation_shim::update,
             py::call_guard<py::gil_scoped_release>(),
             "Rebuild the connection table from recipe::connections_on and the event"
             "generators based on recipe::event_generators.",
             "recipe"_a)
        .def("deserialize", &simulation_shim::deserialize,
             py::call_guard<py::gil_scoped_release>(),
             "Deserialize the simulation object from a JSON string."
             "json"_a)
        .def("serialize", &simulation_shim::serialize,
             py::call_guard<py::gil_scoped_release>(),
             "Serialize the simulation object to a JSON string.")
        .def("reset", &simulation_shim::reset,
            py::call_guard<py::gil_scoped_release>(),
            "Reset the state of the simulation to its initial state.")
        .def("clear_samplers", &simulation_shim::clear_samplers,
             py::call_guard<py::gil_scoped_release>(),
             "Clearing spike and sample information. restoring memory")
        .def("run", &simulation_shim::run,
            py::call_guard<py::gil_scoped_release>(),
            "Run the simulation from current simulation time to tfinal [ms], with maximum time step size dt [ms].",
            "tfinal"_a, py::arg_v("dt", 0.025*arb::units::ms, "0.025*arbor.units.ms"))
        .def("record", &simulation_shim::record,
            "Disable or enable local or global spike recording.")
        .def("spikes", &simulation_shim::spikes,
            "Retrieve recorded spikes as numpy array.")
        .def("probe_metadata", &simulation_shim::get_probe_metadata,
            "Retrieve metadata associated with given probe id.",
            "probeset_id"_a)
        .def("probe_metadata",
             [](const simulation_shim& sim, const std::tuple<arb::cell_gid_type, arb::cell_tag_type>& addr) {
                 return sim.get_probe_metadata({std::get<0>(addr), std::get<1>(addr)});
             },
            "Retrieve metadata associated with given probe id.",
            "addr"_a)
        .def("probe_metadata",
             [](const simulation_shim& sim,
                arb::cell_gid_type gid, const arb::cell_tag_type& tag) {
                 return sim.get_probe_metadata({gid, tag});
             },
            "Retrieve metadata associated with given probe id.",
            "gid"_a, "tag"_a)
        .def("sample", &simulation_shim::sample,
            "Record data from probes with given probeset_id according to supplied schedule.\n"
            "Returns handle for retrieving data or removing the sampling.",
            "probeset_id"_a, "schedule"_a)
        .def("sample",
             [](simulation_shim& sim, arb::cell_gid_type gid, const arb::cell_tag_type& tag, const schedule_shim_base& schedule) {
                 return sim.sample({gid, tag}, schedule);
             },
            "Record data from probes with given probeset_id=(gid, tag) according to supplied schedule.\n"
            "Returns handle for retrieving data or removing the sampling.",
            "gid"_a, "tag"_a, "schedule"_a)
        .def("sample",
             [](simulation_shim& sim, const std::tuple<arb::cell_gid_type, const arb::cell_tag_type>& addr, const schedule_shim_base& schedule) {
                 return sim.sample({std::get<0>(addr), std::get<1>(addr)}, schedule);
             },
            "Record data from probes with given probeset_id=(gid, tag) according to supplied schedule.\n"
            "Returns handle for retrieving data or removing the sampling.",
            "probeset_id"_a, "schedule"_a)
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
