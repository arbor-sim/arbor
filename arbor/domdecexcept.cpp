#include <string>

#include <arbor/domdecexcept.hpp>

#include "util/strprintf.hpp"

namespace arb {

using arb::util::pprintf;

invalid_gj_cell_group::invalid_gj_cell_group(cell_gid_type gid_0, cell_gid_type gid_1):
    dom_dec_exception(pprintf("cell {} needs to be in the same group as cell {} because they are connected via gap-junction.",
                              gid_0, gid_1)),
    gid_0(gid_0),
    gid_1(gid_1)
{}

invalid_sum_local_cells::invalid_sum_local_cells(unsigned gc_wrong, unsigned gc_right):
    dom_dec_exception(pprintf("sum of local cells on the individual ranks ({}) is not equal to the total number of cells in the recipe ({}).",
                              gc_wrong, gc_right)),
    gc_wrong(gc_wrong),
    gc_right(gc_right)
{}

duplicate_gid::duplicate_gid(cell_gid_type gid):
    dom_dec_exception(pprintf("gid {} is present in multiple cell-groups or multiple times in the same cell group.",
                              gid)),
    gid(gid)
{}

skipped_gid::skipped_gid(cell_gid_type gid, cell_gid_type nxt):
    dom_dec_exception(pprintf("gid list must be contiguous, found [..., {}, {}, ...]",
                              gid, nxt)),
    gid(gid), nxt(nxt)
{}

out_of_bounds::out_of_bounds(cell_gid_type gid, unsigned num_cells):
    dom_dec_exception(pprintf("cell {} is out-of-bounds of the allowed gids in the simulation which has {} total cells.",
                              gid, num_cells)),
    gid(gid),
    num_cells(num_cells)
{}

invalid_backend::invalid_backend(int rank, backend_kind be):
    dom_dec_exception(pprintf("rank {} contains a group requested to run on the {} backend, "
                              "but no such backend detected in the context.",
                              rank, backend_kind_str(be))),
    rank(rank),
    backend(be)
{}

incompatible_backend::incompatible_backend(int rank, cell_kind kind, backend_kind be):
    dom_dec_exception(pprintf("rank {} contains a group with cells of kind {} "
                              "requested to run on the {} backend, which is not supported.",
                              rank, cell_kind_str(kind), backend_kind_str(be))),
    rank(rank),
    kind(kind),
    backend(be)
{}

} // namespace arb
