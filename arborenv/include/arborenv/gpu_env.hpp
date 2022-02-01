#pragma once

#include <arborenv/export.hpp>

namespace arbenv {

template <typename Comm>
ARB_ARBORENV_API int find_private_gpu(Comm comm);

} // namespace arbenv

