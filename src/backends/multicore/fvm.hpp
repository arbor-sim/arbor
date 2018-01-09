#pragma once

#include <map>
#include <string>
#include <utility>
#include <vector>

#include <backends/event.hpp>
#include <backends/fvm_types.hpp>
#include <common_types.hpp>
#include <constants.hpp>
#include <event_queue.hpp>
#include <ion.hpp>
#include <math.hpp>
#include <mechanism.hpp>
#include <memory/memory.hpp>
#include <memory/wrappers.hpp>
#include <util/maputil.hpp>
#include <util/meta.hpp>
#include <util/rangeutil.hpp>
#include <util/span.hpp>
#include <util/xtuple.hpp>

#include <util/debug.hpp>

#include "matrix_state.hpp"
#include "multi_event_stream.hpp"
#include "threshold_watcher.hpp"

namespace arb {
namespace multicore {

struct backend {
    static bool is_supported() { return true; }
    static std::string name() { return "cpu"; }

    using value_type = fvm_value_type;
    using size_type  = fvm_size_type;

    using array  = memory::host_vector<value_type>;
    using iarray = memory::host_vector<size_type>;

    using view       = typename array::view_type;
    using const_view = typename array::const_view_type;

    using iview       = typename iarray::view_type;
    using const_iview = typename iarray::const_view_type;

    using host_array = array;
    using host_view = array::view_type;

    using matrix_state = arb::multicore::matrix_state<value_type, size_type>;
    using threshold_watcher = arb::multicore::threshold_watcher<value_type, size_type>;

    // backend-specific multi event streams.
    using deliverable_event_stream = arb::multicore::multi_event_stream<deliverable_event>;
    using sample_event_stream = arb::multicore::multi_event_stream<sample_event>;

    // Shared cell(s) state for mechanisms and integration:
    struct shared_state {
        size_type n_cell = 0;   // Number of distinct cells (integration domains).
        size_type n_cv = 0;     // Total number of CVs.

        iarray cv_to_cell;      // Maps CV index to cell index.
        array  time;            // Maps cell index to integration start time [ms].
        array  time_to;         // Maps cell index to integration stop time [ms].
        array  dt;              // Maps cell index to (stop time) - (start time) [ms].
        array  dt_comp;         // Maps CV index to dt [ms].
        array  voltage;         // Maps CV index to membrane voltage [mV].
        array  current_density; // Maps CV index to current density [A/m²].

        std::map<ionKind, ion<backend>> ion_data;

        deliverable_event_stream deliverable_events;

        // debug interface only
        friend std::ostream& operator<<(std::ostream& o, const shared_state& s) {
            s.emit(o);
            return o;
        }

        // debug interface only
        void emit(std::ostream& out) const {
            using util::sepval;

            out << "n_cell " << n_cell << "\n----\n";
            out << "n_cv " << n_cell << "\n----\n";
            out << "cv_to_cell:\n" << sepval(cv_to_cell, ',') << "\n";
            out << "time:\n" << sepval(time, ',') << "\n";
            out << "time_to:\n" << sepval(time_to, ',') << "\n";
            out << "dt:\n" << sepval(dt, ',') << "\n";
            out << "dt_comp:\n" << sepval(dt_comp, ',') << "\n";
            out << "voltage:\n" << sepval(voltage, ',') << "\n";
            out << "current_density:\n" << sepval(current_density, ',') << "\n";
            for (auto& ki: ion_data) {
                auto kn = to_string(ki.first);
                auto& i = const_cast<ion<backend>&>(ki.second);
                out << kn << ".current_density:\n" << sepval(i.current_density(), ',') << "\n";
                out << kn << ".reversal_potential:\n" << sepval(i.reversal_potential(), ',') << "\n";
                out << kn << ".internal_concentration:\n" << sepval(i.internal_concentration(), ',') << "\n";
                out << kn << ".external_concentration:\n" << sepval(i.external_concentration(), ',') << "\n";
            }
        }

        shared_state() = default;

        shared_state(
            size_type n_cell,
            const std::vector<size_type>& cv_to_cell
        ):
            n_cell(n_cell),
            n_cv(cv_to_cell.size()),
            cv_to_cell(memory::make_const_view(cv_to_cell)),
            time(n_cell),
            time_to(n_cell),
            dt(n_cell),
            dt_comp(n_cv),
            voltage(n_cv),
            current_density(n_cv),
            deliverable_events(n_cell)
        {}

        void add_ion(ionKind kind, ion<backend> ion) {
            ion_data.emplace(kind, std::move(ion));
        }
    };

    // perform min/max reductions on 'array' type
    template <typename V>
    static std::pair<V, V> minmax_value(const memory::host_vector<V>& v) {
        return util::minmax_value(v);
    }

    static void update_time_to(array& time_to, const_view time, value_type dt, value_type tmax) {
        size_type ncell = util::size(time);
        for (size_type i = 0; i<ncell; ++i) {
            time_to[i] = std::min(time[i]+dt, tmax);
        }
    }

    // set the per-cell and per-compartment dt_ from time_to_ - time_.
    static void set_dt(array& dt_cell, array& dt_comp, const_view time_to, const_view time, const_iview cv_to_cell) {
        size_type ncell = util::size(dt_cell);
        size_type ncomp = util::size(dt_comp);

        for (size_type j = 0; j<ncell; ++j) {
            dt_cell[j] = time_to[j]-time[j];
        }

        for (size_type i = 0; i<ncomp; ++i) {
            dt_comp[i] = dt_cell[cv_to_cell[i]];
        }
    }

    // perform sampling as described by marked events in a sample_event_stream
    static void take_samples(
        const sample_event_stream::state& s,
        const_view time,
        array& sample_time,
        array& sample_value)
    {
        for (size_type i = 0; i<s.n_streams(); ++i) {
            auto begin = s.begin_marked(i);
            auto end = s.end_marked(i);

            for (auto p = begin; p<end; ++p) {
                sample_time[p->offset] = time[i];
                sample_value[p->offset] = *p->handle;
            }
        }
    }

    // Calculate the reversal potential eX (mV) using Nernst equation
    // eX = RT/zF * ln(Xo/Xi)
    //      R: universal gas constant 8.3144598 J.K-1.mol-1
    //      T: temperature in Kelvin
    //      z: valency of species (K, Na: +1) (Ca: +2)
    //      F: Faraday's constant 96485.33289 C.mol-1
    //      Xo/Xi: ratio of out/in concentrations
    static void nernst(int valency, value_type temperature, const_view Xo, const_view Xi, view eX) {
        // factor 1e3 to scale from V -> mV
        constexpr value_type RF = 1e3*constant::gas_constant/constant::faraday;
        value_type factor = RF*temperature/valency;
        for (std::size_t i=0; i<Xi.size(); ++i) {
            eX[i] = factor*std::log(Xo[i]/Xi[i]);
        }
    }

    static void init_concentration(
            view Xi, view Xo,
            const_view weight_Xi, const_view weight_Xo,
            value_type c_int, value_type c_ext)
    {
        for (std::size_t i=0u; i<Xi.size(); ++i) {
            Xi[i] = c_int*weight_Xi[i];
            Xo[i] = c_ext*weight_Xo[i];
        }
    }
};

// Base class for all generated mechanisms for multicore back-end.

class mechanism: public arb::concrete_mechanism<arb::multicore::backend> {
public:
    using value_type = fvm_value_type;
    using size_type = fvm_size_type;

protected:
    using backend = arb::multicore::backend;
    using deliverable_event_stream = backend::deliverable_event_stream;

    using array  = backend::array;
    using iarray = backend::iarray;

    using view       = backend::view;
    using const_view = backend::const_view;

    using iview       = backend::iview;
    using const_iview = backend::const_iview;

    struct ion_state {
        view current_density;
        view reversal_potential;
        view internal_concentration;
        view external_concentration;
    };

private:
    std::size_t width_ = 0; // Instance width (number of CVs/sites)
    std::size_t n_ion_ = 0;

public:
    std::size_t size() const override {
        return width_;
    }

    std::size_t memory() const override {
        std::size_t s = object_sizeof();

        s += sizeof(value_type) * (data_.size() + weight_.size());
        s += sizeof(size_type) * width_ * (n_ion_ + 1); // node and ion indices.
        return s;
    }

    void instantiate(fvm_size_type id, backend::shared_state& shared, const layout& w) override {
        using memory::make_view;
        using memory::make_const_view;

        mechanism_id_ = id;
        width_ = w.cv.size();

        // Assign non-owning views onto shared state:

        vec_ci_   = make_const_view(shared.cv_to_cell);
        vec_t_    = make_const_view(shared.time);
        vec_t_to_ = make_const_view(shared.time_to);
        vec_dt_   = make_const_view(shared.dt_comp);
        vec_v_    = make_view(shared.voltage);
        vec_i_    = make_view(shared.current_density);

        auto ion_state_tbl = ion_state_table();
        n_ion_ = ion_state_tbl.size();
        for (auto i: ion_state_tbl) {
            util::optional<ion<backend>&> oion = util::value_by_key(shared.ion_data, i.first);
            if (!oion) {
                throw std::logic_error("mechanism holds ion with no corresponding shared state");
            }

            ion_state& state = *i.second;
            state.current_density = oion->current_density();
            state.reversal_potential = oion->reversal_potential();
            state.internal_concentration = oion->internal_concentration();
            state.external_concentration = oion->external_concentration();
        }

        event_stream_ptr_ = &shared.deliverable_events;

        // Allocate and copy local state: weight, node indices, ion indices.

        node_index_ = iarray(make_const_view(w.cv));
        weight_     = array(make_const_view(w.weight));

        for (auto i: ion_index_table()) {
            std::vector<size_type> mech_ion_index;

            util::optional<ion<backend>&> oion = util::value_by_key(shared.ion_data, i.first);
            if (!oion) {
                throw std::logic_error("mechanism holds ion with no corresponding shared state");
            }

            iarray& x = *i.second;
            util::assign(mech_ion_index, algorithms::index_into(w.cv, oion->node_index()));
            x = iarray(make_view(mech_ion_index));
        }

        // Allocate and initialize state and parameter vectors.

        constexpr std::size_t align = data_.alignment();
        static_assert(align%sizeof(value_type)==0 || sizeof(value_type)%align==0, "alignment incompatible with value type");

        auto stride = math::round_up(width_*sizeof(value_type), align)/sizeof(value_type);

        auto fields = field_table();
        std::size_t n_field = fields.size();

        data_ = array(n_field*stride, NAN);
        for (std::size_t i = 0; i<n_field; ++i) {
            auto& field_view = *std::get<1>(fields[i]);

            field_view = data_(i*stride, i*stride+width_);
            memory::fill(field_view, std::get<2>(fields[i]));
        }
    }

    void deliver_events() override {
        // Delegate to derived class, passing in event queue state.
        deliver_events(event_stream_ptr_->marked_events());
    }

    void set_parameter(const std::string& key, const std::vector<fvm_value_type>& values) override {
        if (auto opt_ptr = util::value_by_key(field_table(), key)) {
            view& field = *opt_ptr.value();
            if (field.size() != values.size()) {
                throw std::logic_error("internal error: mechanism parameter size mismatch");
            }
            memory::copy(memory::make_const_view(values), field);
        }
    }

    void set_global(const std::string& key, fvm_value_type value) override {
        if (auto opt_ptr = util::value_by_key(global_table(), key)) {
            value_type& global = *opt_ptr.value();
            global = value;
        }
    }

protected:
    // Non-owning views onto shared cell state, excepting ion state.

    const_iview vec_ci_;
    const_view vec_t_;
    const_view vec_t_to_;
    const_view vec_dt_;
    view vec_v_;
    view vec_i_;
    deliverable_event_stream* event_stream_ptr_;

    // Per-mechanism index and weight data, excepting ion indices.

    iarray node_index_;
    array weight_;

    // Bulk storage for state and parameter variables.

    array data_;

    // Generated mechanism field, global and ion table lookup types.
    // First component is name, second is pointer to corresponing member.
    // Field table entries have a third component for the field default value.

    using global_table_entry = std::pair<const char*, value_type*>;
    using mechanism_global_table = std::vector<global_table_entry>;

    using field_table_entry = util::xtuple<const char*, view*, value_type>;
    using mechanism_field_table = std::vector<field_table_entry>;

    using ion_state_entry = std::pair<ionKind, ion_state*>;
    using mechanism_ion_state_table = std::vector<ion_state_entry>;

    using ion_index_entry = std::pair<ionKind, iarray*>;
    using mechanism_ion_index_table = std::vector<ion_index_entry>;

    // Generated mechanisms must implement the following methods, together with
    // clone(), kind(), nrn_init(), nrn_state(), nrn_current() and deliver_events() (if
    // required) from arb::mechanism.

    // Member tables: introspection into derived mechanism fields, views etc.
    // Default implementations correspond to no corresponding fields/globals/ions.

    virtual mechanism_field_table field_table() { return {}; }
    virtual mechanism_global_table global_table() { return {}; }
    virtual mechanism_ion_state_table ion_state_table() { return {}; }
    virtual mechanism_ion_index_table ion_index_table() { return {}; }

    // Report raw size in bytes of mechanism object.

    virtual std::size_t object_sizeof() const = 0;

    // Event delivery, given event queue state:

    virtual void deliver_events(deliverable_event_stream::state) {};
};

} // namespace multicore
} // namespace arb
