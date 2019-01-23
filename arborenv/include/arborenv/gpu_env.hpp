#pragma once

namespace arbenv {

int default_gpu();

template <typename Comm>
int find_private_gpu(Comm comm);

} // namespace arbenv

