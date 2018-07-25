#include <ostream>

namespace arb {
namespace gpu {

struct level {
    level() = default;

    level(unsigned branches);
    level(level&& other);
    level(const level& other) = delete;

    ~level();

    unsigned num_branches = 0; // Number of branches in the level
    unsigned max_length = 0;   // Length of the longest branch
    unsigned data_index = 0;   // Index into data values

    unsigned* lengths = nullptr; // stored in managed memory

    // Parent index of each branch: in range [0:n),
    // where n is the number of branches in the parent level.
    unsigned* parents = nullptr; // stored in managed memory
};

std::ostream& operator<<(std::ostream& o, const level& l);

} // namespace gpu
} // namespace arb
