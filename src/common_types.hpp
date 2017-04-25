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

// For identifying cells globally.

using cell_gid_type = std::uint32_t;

// For sizes of collections of cells.

using cell_size_type = typename std::make_unsigned<cell_gid_type>::type;

// For indexes into cell-local data.
//
// Local indices for items within a particular cell-local collection should be
// zero-based and numbered contiguously.

using cell_lid_type = std::uint32_t;

// For counts of cell-local data.

using cell_local_size_type = typename std::make_unsigned<cell_lid_type>::type;

// For global identification of an item of cell local data.
//
// Items of cell_member_type must:
//
//  * be associated with a unique cell, identified by the member `gid`
//    (see: cell_gid_type);
//
//  * identify an item within a cell-local collection by the member `index`
//    (see: cell_lid_type).

struct cell_member_type {
    cell_gid_type gid;
    cell_lid_type index;
};

DEFINE_LEXICOGRAPHIC_ORDERING(cell_member_type,(a.gid,a.index),(b.gid,b.index))

// For storing time values [ms]

using time_type = float;

/* Enumeration used to indentify the cell type/kind, used by the model to
* group equal kinds in the same cell group.
*
*
*/
enum cell_kind {
    multicompartment,
    poisson,
    from_file,
    //IAF,
    //from_music,
    //NestML
};


} // namespace mc
} // namespace nest

std::ostream& operator<<(std::ostream& O, nest::mc::cell_member_type m);
