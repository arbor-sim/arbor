#include <arbor/mechanism.hpp>

#include <arbor/arbexcept.hpp>

#include "memory/memory.hpp"
#include "util/range.hpp"
#include "util/span.hpp"
#include "util/maputil.hpp"
#include "util/rangeutil.hpp"
#include "util/strprintf.hpp"

#include "backends/multicore/fvm.hpp"
#include "backends/gpu/fvm.hpp"

namespace arb {

using util::make_range;
using util::make_span;
using util::ptr_by_key;
using util::value_by_key;

void mechanism::initialize() {
  iface_.init_mechanism(&ppack_);
  if (!mult_in_place_) return;
  switch (iface_.backend) {
  case arb_backend_kind_cpu:
    for (auto& v: state_vars_) {
      arb::multicore::backend::multiply_in_place(v, ppack_.multiplicity, ppack_.width);
    }
    break;
#ifdef ARB_HAVE_GPU
  case arb_backend_kind_gpu:
    for (auto& v: state_vars_) {
      arb::gpu::backend::multiply_in_place(v, ppack_.multiplicity, ppack_.width);
    }
    break;
#endif
  default: throw arbor_internal_error(util::pprintf("Unknown backend ID {}", iface_.backend));
  }
}

void mechanism::set_parameter(const std::string& key, const std::vector<arb_value_type>& values) {
    if (values.size()!=ppack_.width) throw arbor_internal_error("mechanism parameter size mismatch");
    auto field_ptr = field_data(key);
    if (!field_ptr) throw arbor_internal_error(util::pprintf("no such mechanism parameter '{}'", key));
    if (!ppack_.width) return;
    switch (iface_.backend) {
        case arb_backend_kind_cpu:
            copy_extend(values, util::range_n(field_ptr, width_padded_), values.back());
            break;
#ifdef ARB_HAVE_GPU	    
        case arb_backend_kind_gpu:
            memory::copy(memory::make_const_view(values), memory::device_view<arb_value_type>(field_ptr, ppack_.width));
            break;
#endif	    
        default:
            throw arbor_internal_error(util::pprintf("Unknown backend ID '{}'", iface_.backend));
    }
}

arb_value_type* mechanism::field_data(const std::string& var) {
    for (auto idx: make_span(mech_.n_parameters)) {
        if (var == mech_.parameters[idx].name) return parameters_[idx];
    }
    for (auto idx: make_span(mech_.n_state_vars)) {
        if (var == mech_.state_vars[idx].name) return state_vars_[idx];
    }
    return nullptr;
}

mechanism_field_table mechanism::field_table() {
    mechanism_field_table result;
    for (auto idx = 0ul; idx < mech_.n_parameters; ++idx) {
        result.emplace_back(mech_.parameters[idx].name, std::make_pair(parameters_[idx], mech_.parameters[idx].default_value));
    }
    for (auto idx = 0ul; idx < mech_.n_state_vars; ++idx) {
        result.emplace_back(mech_.state_vars[idx].name, std::make_pair(state_vars_[idx], mech_.state_vars[idx].default_value));
    }
    return result;
}

mechanism_global_table mechanism::global_table() {
    mechanism_global_table result;
    for (auto idx = 0ul; idx < mech_.n_globals; ++idx) {
        result.emplace_back(mech_.globals[idx].name, globals_[idx]);
    }
    return result;
}

mechanism_state_table mechanism::state_table() {
    mechanism_state_table result;
    for (auto idx = 0ul; idx < mech_.n_state_vars; ++idx) {
        result.emplace_back(mech_.state_vars[idx].name, std::make_pair(state_vars_[idx], mech_.state_vars[idx].default_value));
    }
    return result;
}

mechanism_ion_table mechanism::ion_table() {
    mechanism_ion_table result;
    for (auto idx = 0ul; idx < mech_.n_ions; ++idx) {
        ion_state_view v;
        v.current_density        = ion_states_[idx].current_density;
        v.external_concentration = ion_states_[idx].internal_concentration;
        v.internal_concentration = ion_states_[idx].external_concentration;
        v.ionic_charge           = ion_states_[idx].ionic_charge;
        result.emplace_back(mech_.ions[idx].name, std::make_pair(v, ion_states_[idx].index));
    }
    return result;
}

}
