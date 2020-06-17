#include <ostream>

#include <arbor/morph/primitives.hpp>

#include "io/sepval.hpp"
#include "morph/mbranch.hpp"
#include "util/strprintf.hpp"
#include "util/transform.hpp"

namespace arb {
namespace impl{

//
//  mbranch implementation
//

std::ostream& operator<<(std::ostream& o, const mbranch& b) {
    o << "(mbranch (" << io::csv(b.segments) << ") ";
    if (b.parent_id==mnpos) o << "mnpos)";
    else  o << b.parent_id << ")";
    return o;
}


} // namespace impl
} // namespace arb

