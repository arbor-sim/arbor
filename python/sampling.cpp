#include <mutex>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <arbor/common_types.hpp>
#include <arbor/sampling.hpp>
#include <arbor/simulation.hpp>

#include "error.hpp"
#include "strprintf.hpp"

namespace pyarb {

// TODO: trace entry of different types/container (e.g. vector of doubles to get all samples of a cell)

struct trace_entry {
    arb::time_type t;
    double v;
};

// A helper struct (state) ensuring that only one thread can write to the probe_buffers holding the trace entries (mapped by probe id)
struct sampler_state {
    std::mutex mutex;
    std::unordered_map<arb::cell_member_type, std::vector<trace_entry>> probe_buffers;

    std::vector<trace_entry>& probe_buffer(arb::cell_member_type pid) {
        // lock the mutex, s.t. other threads cannot write
        std::lock_guard<std::mutex> lock(mutex);
        // return or create entry
        return probe_buffers[pid];
    }

    // helper function to search probe id in probe_buffers
    bool has_pid(arb::cell_member_type pid) {
        return probe_buffers.count(pid);
    }

    // helper function to push back to locked vector
    void push_back(arb::cell_member_type pid, trace_entry value) {
        auto& v = probe_buffer(pid);
        v.push_back(std::move(value));
    }

    // Access the probe buffers
    const std::unordered_map<arb::cell_member_type, std::vector<trace_entry>>& samples() const {
        return probe_buffers;
    }
};

// A functor that models arb::sampler_function.
// Holds a shared pointer to the trace_entry used to store the samples, so that if
// the trace_entry in sampler is garbage collected in Python, stores will
// not seg fault.

struct sample_callback {
    std::shared_ptr<sampler_state> sample_store;

    sample_callback(const std::shared_ptr<sampler_state>& state):
        sample_store(state)
    {}

    void operator() (arb::cell_member_type probe_id, arb::probe_tag tag, std::size_t n, const arb::sample_record* recs) {
        auto& v = sample_store->probe_buffer(probe_id);
        for (std::size_t i = 0; i<n; ++i) {
            if (auto p = arb::util::any_cast<const double*>(recs[i].data)) {
                v.push_back({recs[i].time, *p});
            }
            else {
                throw std::runtime_error("unexpected sample type");
            }
        }
    };
};

// Helper type for recording samples from a simulation.
// This type is wrapped in Python, to expose sampler::sample_store.
struct sampler {
    std::shared_ptr<sampler_state> sample_store;

    sample_callback callback() {
        // initialize the sample_store
        sample_store = std::make_shared<sampler_state>();

        // The callback holds a copy of sample_store, i.e. the shared
        // pointer is held by both the sampler and the callback, so if
        // the sampler is destructed in the calling Python code, attempts
        // to write to sample_store inside the callback will not seg fault.
        return sample_callback(sample_store);
    }

    const std::vector<trace_entry>& samples(arb::cell_member_type pid) const {
        if (!sample_store->has_pid(pid)) {
            throw std::runtime_error(util::pprintf("probe id {} does not exist", pid));
        }
        return sample_store->probe_buffer(pid);
    }

    void clear() {
        for (auto b: sample_store->probe_buffers) {
            b.second.clear();
        }
    }
};

// Adds sampler to one probe with pid
std::shared_ptr<sampler> attach_sampler(arb::simulation& sim, arb::time_type interval, arb::cell_member_type pid) {
    auto r = std::make_shared<sampler>();
    sim.add_sampler(arb::one_probe(pid), arb::regular_schedule(interval), r->callback());
    return r;
}

// Adds sampler to all probes
std::shared_ptr<sampler> attach_sampler(arb::simulation& sim, arb::time_type interval) {
    auto r = std::make_shared<sampler>();
    sim.add_sampler(arb::all_probes, arb::regular_schedule(interval), r->callback());
    return r;
}

std::string sample_str(const trace_entry& s) {
        return util::pprintf("<arbor.sample: time {} ms, \tvalue {}>", s.t, s.v);
}

void register_sampling(pybind11::module& m) {
    using namespace pybind11::literals;

    // Sample
    pybind11::class_<trace_entry> sample(m, "sample");
    sample
        .def_readonly("time", &trace_entry::t, "The sample time [ms] at a specific probe.")
        .def_readonly("value", &trace_entry::v, "The sample record at a specific probe.")
        .def("__str__",  &sample_str)
        .def("__repr__", &sample_str);

    // Sampler
    pybind11::class_<sampler, std::shared_ptr<sampler>> samplerec(m, "sampler");
    samplerec
        .def(pybind11::init<>())
        .def("samples", &sampler::samples,
            "A list of the recorded samples of a probe with probe id.",
            "probe_id"_a)
        .def("clear", &sampler::clear, "Clear all recorded samples.");

    m.def("attach_sampler",
        (std::shared_ptr<sampler> (*)(arb::simulation&, arb::time_type)) &attach_sampler,
        "Attach a sample recorder to an arbor simulation.\n"
        "The recorder will record all samples from a regular sampling interval [ms] matching all probe ids.",
        "sim"_a, "dt"_a);

    m.def("attach_sampler",
        (std::shared_ptr<sampler> (*)(arb::simulation&, arb::time_type, arb::cell_member_type)) &attach_sampler,
        "Attach a sample recorder to an arbor simulation.\n"
        "The recorder will record all samples from a regular sampling interval [ms] matching one probe id.",
        "sim"_a, "dt"_a, "probe_id"_a);
}

} // namespace pyarb
