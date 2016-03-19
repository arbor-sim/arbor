#pragma once

#include <cmath>

namespace nestmc {

    template <typename T>
    T constexpr pi() {
        return 3.141592653589793238462643383279502;
    }

    template <typename T>
    T area_frustrum(T L, T r1, T r2)
    {
        auto dr = r1 - r2;
        return pi<double>() * (r1+r2) * std::sqrt(L*L + dr*dr);
    }

    template <typename T>
    T volume_frustrum(T L, T r1, T r2)
    {
        auto meanr = (r1+r2) / 2.;
        return pi<double>() * meanr * meanr * L;
    }

} // namespace nestmc
