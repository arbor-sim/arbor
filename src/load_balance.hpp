#include <communication/global_policy.hpp>
#include <domain_decomposition.hpp>
#include <hardware/node_info.hpp>
#include <recipe.hpp>

namespace arb {

domain_decomposition partition_load_balance(const recipe& rec, hw::node_info nd);

} // namespace arb
