#pragma once

/*
 * Common definitions for index types etc. across prototype simulator
 * library. (Expect analogues in future versions to be template parameters?)
 */

#include <iosfwd>
#include <type_traits>

#include <util/lexcmp_def.hpp>

namespace nest {
namespace mc {

// for identifying cells globally
using cell_gid_type = std::uint32_t;

// for sizes of collections of cells
using cell_size_type = typename std::make_unsigned<cell_gid_type>::type;

// for indexes into cell-local data
using cell_lid_type = std::uint32_t;

// for counts of cell-local data
using cell_local_size_type = typename std::make_unsigned<cell_lid_type>::type;

struct cell_member_type {
    cell_gid_type gid;
    cell_lid_type index;
};

DEFINE_LEXICOGRAPHIC_ORDERING(cell_member_type,(a.gid,a.index),(b.gid,b.index))

} // namespace mc
} // namespace nest

std::ostream &operator<<(std::ostream& O, nest::mc::cell_member_type m);
