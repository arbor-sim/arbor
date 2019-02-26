#pragma once

namespace arb {
namespace memory {
// tag for final element in a range
struct end_type {};

namespace{
    // attach the unused attribute so that -Wall won't generate warnings when
    // translation units that include this file don't use end
    end_type end [[gnu::unused]];
}

} // namespace memory
} // namespace arb
