#include <cstdint>

#include <constants.hpp>

#include "../ions.hpp"
#include "detail.hpp"

namespace arb {
namespace gpu {

namespace kernels {
    template <typename T>
    __global__
    void nernst(std::size_t n, int valency, T temperature, const T* Xo, const T* Xi, T* eX) {
        auto i = threadIdx.x+blockIdx.x*blockDim.x;

        // factor 1e3 to scale from V -> mV
        constexpr T RF = 1e3*constant::gas_constant/constant::faraday;
        T factor = RF*temperature/valency;
        if (i<n) {
            eX[i] = factor*std::log(Xo[i]/Xi[i]);
        }
    }

    template <typename T>
    __global__
    void init_concentration(std::size_t n, T* Xi, T* Xo, const T* weight_Xi, const T* weight_Xo, T c_int, T c_ext) {
        auto i = threadIdx.x+blockIdx.x*blockDim.x;

        if (i<n) {
            Xi[i] = c_int*weight_Xi[i];
            Xo[i] = c_ext*weight_Xo[i];
        }
    }
} // namespace kernels

void nernst(std::size_t n,
            int valency,
            fvm_value_type temperature,
            const fvm_value_type* Xo,
            const fvm_value_type* Xi,
            fvm_value_type* eX)
{
    constexpr int block_dim = 128;
    const int grid_dim = impl::block_count(n, block_dim);
    kernels::nernst<<<grid_dim, block_dim>>>
        (n, valency, temperature, Xo, Xi, eX);
}

void init_concentration(
            std::size_t n,
            fvm_value_type* Xi, fvm_value_type* Xo,
            const fvm_value_type* weight_Xi, const fvm_value_type* weight_Xo,
            fvm_value_type c_int, fvm_value_type c_ext)
{
    constexpr int block_dim = 128;
    const int grid_dim = impl::block_count(n, block_dim);
    kernels::init_concentration<<<grid_dim, block_dim>>>
        (n, Xi, Xo, weight_Xi, weight_Xo, c_int, c_ext);
}

} // namespace gpu
} // namespace arb
