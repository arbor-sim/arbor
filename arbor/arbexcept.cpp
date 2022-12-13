#include <string>
#include <sstream>

#include <arbor/arbexcept.hpp>
#include <arbor/common_types.hpp>

#include "util/unwind.hpp"
#include "util/strprintf.hpp"

namespace arb {

using arb::util::pprintf;

arbor_exception::arbor_exception(const std::string& what):
    std::runtime_error{what} {
    // Backtrace w/o this c'tor and that of backtrace.
    where = util::backtrace{}.pop(2).to_string();
}

arbor_internal_error::arbor_internal_error(const std::string& what):
        std::logic_error(what) {
    // Backtrace w/o this c'tor and that of backtrace.
    where = util::backtrace{}.pop(2).to_string();
}

domain_error::domain_error(const std::string& w): arbor_exception(w) {}

bad_cell_probe::bad_cell_probe(cell_kind kind, cell_gid_type gid):
    arbor_exception(pprintf("recipe::get_grobe() is not supported for cell with gid {} of kind {})", gid, kind)),
    gid(gid),
    kind(kind)
{}

bad_cell_description::bad_cell_description(cell_kind kind, cell_gid_type gid):
    arbor_exception(pprintf("recipe::get_cell_kind(gid={}) -> {} does not match the cell type provided by recipe::get_cell_description(gid={})", gid, kind, gid)),
    gid(gid),
    kind(kind)
{}

invalid_mechanism_kind::invalid_mechanism_kind(arb_mechanism_kind kind):
    arbor_exception(pprintf("Invalid mechanism kind: {})", kind)),
    kind(kind)
{}

bad_connection_source_gid::bad_connection_source_gid(cell_gid_type gid, cell_gid_type src_gid, cell_size_type num_cells):
    arbor_exception(pprintf("Model building error on cell {}: connection source gid {} is out of range: there are only {} cells in the model, in the range [{}:{}].", gid, src_gid, num_cells, 0, num_cells-1)),
    gid(gid), src_gid(src_gid), num_cells(num_cells)
{}

bad_connection_label::bad_connection_label(cell_gid_type gid, const cell_tag_type& label, const std::string& msg):
    arbor_exception(pprintf("Model building error on cell {}: connection endpoint label \"{}\": {}.", gid, label, msg)),
    gid(gid), label(label)
{}

bad_global_property::bad_global_property(cell_kind kind):
    arbor_exception(pprintf("bad global property for cell kind {}", kind)),
    kind(kind)
{}

zero_thread_requested_error::zero_thread_requested_error(unsigned nbt):
    arbor_exception(pprintf("threads must be a positive integer")),
    nbt(nbt)
{}

bad_probeset_id::bad_probeset_id(cell_member_type probeset_id):
    arbor_exception(pprintf("bad probe id {}", probeset_id)),
    probeset_id(probeset_id)
{}

gj_unsupported_lid_selection_policy::gj_unsupported_lid_selection_policy(cell_gid_type gid, cell_tag_type label):
    arbor_exception(pprintf("Model building error on cell {}: gap junction site label \"{}\" must be univalent.", gid, label)),
    gid(gid),
    label(label)
{}

gj_kind_mismatch::gj_kind_mismatch(cell_gid_type gid_0, cell_gid_type gid_1):
    arbor_exception(pprintf("Cells on gid {} and {} connected via gap junction have different cell kinds", gid_0, gid_1)),
    gid_0(gid_0),
    gid_1(gid_1)
{}

bad_event_time::bad_event_time(time_type event_time, time_type sim_time):
    arbor_exception(pprintf("event time {} precedes current simulation time {}", event_time, sim_time)),
    event_time(event_time),
    sim_time(sim_time)
{}

no_such_mechanism::no_such_mechanism(const std::string& mech_name):
    arbor_exception(pprintf("no mechanism {} in catalogue", mech_name)),
    mech_name(mech_name)
{}

duplicate_mechanism::duplicate_mechanism(const std::string& mech_name):
    arbor_exception(pprintf("mechanism {} already exists", mech_name)),
    mech_name(mech_name)
{}

fingerprint_mismatch::fingerprint_mismatch(const std::string& mech_name):
    arbor_exception(pprintf("mechanism {} has different fingerprint in schema", mech_name)),
    mech_name(mech_name)
{}

no_such_parameter::no_such_parameter(const std::string& mech_name, const std::string& param_name):
    arbor_exception(pprintf("mechanism {} has no parameter {}", mech_name, param_name)),
    mech_name(mech_name),
    param_name(param_name)
{}

illegal_diffusive_mechanism::illegal_diffusive_mechanism(const std::string& m, const std::string& i):
        arbor_exception(pprintf("mechanism '{}' accesses diffusive value of ion '{}', but diffusivity is disabled for it.", m, i)),
        mech{m},
        ion{i}
{}

invalid_parameter_value::invalid_parameter_value(const std::string& mech_name, const std::string& param_name, double value):
    arbor_exception(pprintf("invalid parameter value for mechanism {} parameter {}: {}", mech_name, param_name, value)),
    mech_name(mech_name),
    param_name(param_name),
    value_str(),
    value(value)
{}

invalid_parameter_value::invalid_parameter_value(const std::string& mech_name, const std::string& param_name, const std::string& value_str):
    arbor_exception(pprintf("invalid parameter value for mechanism {} parameter {}: {}", mech_name, param_name, value_str)),
    mech_name(mech_name),
    param_name(param_name),
    value_str(value_str),
    value(0)
{}

invalid_ion_remap::invalid_ion_remap(const std::string& mech_name):
    arbor_exception(pprintf("invalid ion parameter remapping for mechanism {}", mech_name))
{}

invalid_ion_remap::invalid_ion_remap(const std::string& mech_name, const std::string& from_ion = "", const std::string& to_ion = ""):
    arbor_exception(pprintf("invalid ion parameter remapping for mechanism {}: {} -> {}", mech_name, from_ion, to_ion)),
    from_ion(from_ion),
    to_ion(to_ion)
{}

no_such_implementation::no_such_implementation(const std::string& mech_name):
    arbor_exception(pprintf("missing implementation for mechanism {} in catalogue", mech_name)),
    mech_name(mech_name)
{}

range_check_failure::range_check_failure(const std::string& whatstr, double value):
    arbor_exception(pprintf("range check failure: {} with value {}", whatstr, value)),
    value(value)
{}

file_not_found_error::file_not_found_error(const std::string &fn)
    : arbor_exception(pprintf("Could not find readable file at '{}'", fn)),
      filename{fn}
{}

bad_catalogue_error::bad_catalogue_error(const std::string& msg)
    : arbor_exception(pprintf("Error while opening catalogue '{}'", msg))
{}

bad_catalogue_error::bad_catalogue_error(const std::string& msg, const std::any& pe)
    : arbor_exception(pprintf("Error while opening catalogue '{}'", msg)), platform_error(pe)
{}

unsupported_abi_error::unsupported_abi_error(size_t v):
    arbor_exception(pprintf("ABI version is not supported by this version of arbor '{}'", v)),
    version{v} {}

bad_alignment::bad_alignment(size_t a):
    arbor_exception(pprintf("Mechanism reported unsupported alignment '{}'", a)),
    alignment{a} {}

} // namespace arb

