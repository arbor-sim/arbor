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

invalid_num_domains::invalid_num_domains(int doms_wrong, int doms_right):
    dom_dec_exception(pprintf("provided num_domains ({}) is not equal to the detected number of domains ({}).",
                              domains_wrong, domains_right)),
    domains_wrong(doms_wrong),
    domains_right(doms_right)
{}

invalid_domain_id::invalid_domain_id(int id_wrong, int id_right):
    dom_dec_exception(pprintf("provided domain_id ({}) is not equal to the detected domain id ({}).",
                              id_wrong, id_right)),
    id_wrong(id_wrong),
    id_right(id_right)
{}

invalid_num_local_cells::invalid_num_local_cells(int rank, unsigned lc_wrong, unsigned lc_right):
    dom_dec_exception(pprintf("provided num_local_cells ({}) on rank {} is not equal to the detected number of local cells ({}).",
                              lc_wrong, rank, lc_right)),
    rank(rank),
    lc_wrong(lc_wrong),
    lc_right(lc_right)
{}

invalid_num_global_cells::invalid_num_global_cells(unsigned gc_wrong, unsigned gc_right):
    dom_dec_exception(pprintf("provided num_global_cells ({}) is not equal to the total number of cells in the recipe ({}).",
                              gc_wrong, gc_right)),
    gc_wrong(gc_wrong),
    gc_right(gc_right)
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

non_existent_rank::non_existent_rank(cell_gid_type gid, int rank):
    dom_dec_exception(pprintf("gid {} is assigned to a non-existent rank {}.", gid, rank)),
    gid(gid),
    rank(rank)
{}

out_of_bounds::out_of_bounds(cell_gid_type gid, unsigned num_cells):
    dom_dec_exception(pprintf("cell {} is out-of-bounds of the allowed gids in the simulation which has {} total cells.",
                              gid, num_cells)),
    gid(gid),
    num_cells(num_cells)
{}

invalid_backend::invalid_backend(int rank):
    dom_dec_exception(pprintf("rank {} contains a group meant to run on GPU, but no GPU backend was detected in the context.",
                              rank)),
    rank(rank)
{}

incompatible_backend::incompatible_backend(int rank, cell_kind kind):
    dom_dec_exception(pprintf("rank {} contains a group with cells of kind {} meant to run on the GPU backend, but no GPU backend support exists for {}",
                              rank, kind, kind)),
    rank(rank),
    kind(kind)
{}

} // namespace arb

