#include <string>
#include <sstream>

#include <arbor/arbexcept.hpp>
#include <arbor/common_types.hpp>

#include "util/strprintf.hpp"

namespace arb {

using arb::util::pprintf;

bad_cell_description::bad_cell_description(cell_kind kind, cell_gid_type gid):
    arbor_exception(pprintf("recipe::get_cell_kind(gid={}) -> {} does not match the cell type provided by recipe::get_cell_description(gid={})", gid, kind, gid)),
    gid(gid),
    kind(kind)
{}

bad_target_description::bad_target_description(cell_gid_type gid, cell_size_type rec_val, cell_size_type cell_val):
    arbor_exception(pprintf("recipe::num_targets(gid={}) -> {} does not match the number of synapses on the cell -> {})", gid, rec_val, cell_val)),
    gid(gid), rec_val(rec_val), cell_val(cell_val)
{}

bad_source_description::bad_source_description(cell_gid_type gid, cell_size_type rec_val, cell_size_type cell_val):
    arbor_exception(pprintf("recipe::num_sources(gid={}) -> {} does not match the number of threshold detectors on the cell -> {})", gid, rec_val, cell_val)),
    gid(gid), rec_val(rec_val), cell_val(cell_val)
{}

bad_connection_source::bad_connection_source(cell_gid_type gid, cell_member_type source):
    arbor_exception(pprintf("recipe::connections_on(gid={}): source ({}, {}) does not exist)", gid, source.gid, source.index)),
    gid(gid), source(source)
{}

bad_connection_target::bad_connection_target(cell_gid_type gid, cell_member_type target):
    arbor_exception(pprintf("recipe::connections_on(gid={}): target ({}, {}) does not exist)", gid, target.gid, target.index)),
    gid(gid), target(target)
{}

connection_target_mismatch::connection_target_mismatch(cell_gid_type gid, cell_member_type target):
arbor_exception(pprintf("recipe::connections_on(gid={}): target ({}, {}) must be on the cell with gid = {})", gid, target.gid, target.index, gid)),
gid(gid), target(target)
{}

bad_global_property::bad_global_property(cell_kind kind):
    arbor_exception(pprintf("bad global property for cell kind {}", kind)),
    kind(kind)
{}

bad_probe_id::bad_probe_id(cell_member_type probe_id):
    arbor_exception(pprintf("bad probe id {}", probe_id)),
    probe_id(probe_id)
{}

gj_unsupported_domain_decomposition::gj_unsupported_domain_decomposition(cell_gid_type gid_0, cell_gid_type gid_1):
    arbor_exception(pprintf("No support for gap junctions across domain decomposition groups for gid {} and {}", gid_0, gid_1)),
    gid_0(gid_0),
    gid_1(gid_1)
{}

gj_connection_mismatch::gj_connection_mismatch(cell_gid_type gid, cell_member_type site_0, cell_member_type site_1):
        arbor_exception(pprintf("recipe::gap_junctions_on(gid={}) -> ({}, {}) <-> ({}, {}): one of the sites must be on the cell with gid = {})", gid, site_0, site_1, gid)),
        gid(gid),
        site_0(site_0),
        site_1(site_1)
{}

bad_gj_connection::bad_gj_connection(cell_gid_type gid, cell_member_type site_0, cell_member_type site_1):
    arbor_exception(pprintf("recipe::gap_junctions_on(gid={}) -> ({}, {}) <-> ({}, {}): one of the sites does not exist)", gid, site_0.gid, site_0.index, site_1.gid, site_1.index)),
    gid(gid),
    site_0(site_0),
    site_1(site_1)
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

} // namespace arb

