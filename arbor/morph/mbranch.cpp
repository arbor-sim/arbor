#include <ostream>

#include <arbor/morph/primitives.hpp>

#include "io/sepval.hpp"
#include "morph/mbranch.hpp"

namespace arb {
namespace impl{

//
//  mbranch implementation
//

bool operator==(const mbranch& l, const mbranch& r) {
    return l.parent_id==r.parent_id && l.index==r.index;
}

std::ostream& operator<<(std::ostream& o, const mbranch& b) {
    o <<"mbranch([" << io::csv(b.index) << "], ";
    if (b.parent_id==mnpos) o << "none)";
    else  o << b.parent_id << ")";
    return o;
}


} // namespace impl
} // namespace arb

