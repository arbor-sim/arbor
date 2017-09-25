#include <backends/fvm_types.hpp>
#include "detail.hpp"
#include "interleave.hpp"

namespace nest {
namespace mc {
namespace gpu {

// host side wrapper for the flat to interleaved operation
void flat_to_interleaved(
    const fvm_value_type* in,
    fvm_value_type* out,
    const fvm_size_type* sizes,
    const fvm_size_type* starts,
    unsigned padded_size,
    unsigned num_vec)
{
    constexpr unsigned BlockWidth = impl::block_dim();
    constexpr unsigned LoadWidth  = impl::load_width();

    flat_to_interleaved
        <fvm_value_type, fvm_size_type, BlockWidth, LoadWidth>
        (in, out, sizes, starts, padded_size, num_vec);
}

// host side wrapper for the interleave to flat operation
void interleaved_to_flat(
    const fvm_value_type* in,
    fvm_value_type* out,
    const fvm_size_type* sizes,
    const fvm_size_type* starts,
    unsigned padded_size,
    unsigned num_vec)
{
    constexpr unsigned BlockWidth = impl::block_dim();
    constexpr unsigned LoadWidth  = impl::load_width();

    interleaved_to_flat
        <fvm_value_type, fvm_size_type, BlockWidth, LoadWidth>
        (in, out, sizes, starts, padded_size, num_vec);
}

} // namespace gpu
} // namespace mc
} // namespace nest
