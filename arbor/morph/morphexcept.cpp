#include <string>
#include <sstream>

#include <arbor/morph/primitives.hpp>
#include <arbor/morph/morphexcept.hpp>

#include "util/strprintf.hpp"

namespace arb {

using arb::util::pprintf;

static std::string msize_string(msize_t x) {
    return x==mnpos? "mnpos": pprintf("{}", x);
}

invalid_mlocation::invalid_mlocation(mlocation loc):
    morphology_error(pprintf("invalid mlocation {}", loc)),
    loc(loc)
{}

no_such_branch::no_such_branch(msize_t bid):
    morphology_error(pprintf("no such branch id {}", msize_string(bid))),
    bid(bid)
{}

no_such_segment::no_such_segment(msize_t id):
    arbor_exception(pprintf("no such segment {}", id)),
    sid(id)
{}

invalid_mcable::invalid_mcable(mcable cable):
    morphology_error(pprintf("invalid mcable {}", cable)),
    cable(cable)
{}

invalid_mcable_list::invalid_mcable_list():
    morphology_error("bad mcable_list")
{}

invalid_segment_parent::invalid_segment_parent(msize_t parent, msize_t tree_size):
    morphology_error(pprintf("invalid segment parent {} for a segment tree of size {}", msize_string(parent), tree_size)),
    parent(parent),
    tree_size(tree_size)
{}

duplicate_stitch_id::duplicate_stitch_id(const std::string& id):
    morphology_error(pprintf("duplicate stitch id {}", id)),
    id(id)
{}

no_such_stitch::no_such_stitch(const std::string& id):
    morphology_error(pprintf("no such stitch id {}", id)),
    id(id)
{}

missing_stitch_start::missing_stitch_start(const std::string& id):
    morphology_error(pprintf("require proximal point for stitch id {}", id)),
    id(id)
{}

invalid_stitch_position::invalid_stitch_position(const std::string& id, double along):
    morphology_error(pprintf("invalid stitch position {} on stitch {}", along, id)),
    id(id),
    along(along)
{}

label_type_mismatch::label_type_mismatch(const std::string& label):
    morphology_error(pprintf("label \"{}\" is already bound to a different type of object", label)),
    label(label)
{}

incomplete_branch::incomplete_branch(msize_t bid):
    morphology_error(pprintf("insufficent samples to define branch id {}", msize_string(bid))),
    bid(bid)
{}

unbound_name::unbound_name(const std::string& name):
    morphology_error(pprintf("no definition for '{}'", name)),
    name(name)
{}

circular_definition::circular_definition(const std::string& name):
    morphology_error(pprintf("definition of '{}' requires a definition for '{}'", name, name)),
    name(name)
{}


} // namespace arb

