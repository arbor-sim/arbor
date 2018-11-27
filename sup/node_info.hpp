#pragma once

namespace arb {
namespace hw {

// Number of GPUs detected on the node.
unsigned node_gpus();

// Number of visible logical processors on the node.
// 0 => unable to determine.
unsigned node_processors();

} // namespace hw
} // namespace arb
