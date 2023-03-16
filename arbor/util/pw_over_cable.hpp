#include <arbor/morph/mcable_map.hpp>

#include "util/piecewise.hpp"

namespace arb {
namespace util {

namespace impl {
struct get_value {
    template <typename X>
    double operator()(const mcable&, const X& x) const { return x.value; }
    double operator()(const mcable&, double x) const { return x; }
};
} // namespace impl

// Convert mcable_map values to a piecewise function over an mcable.
// The projection gives the map from the values in the mcable_map to the values in the piecewise function.
template <typename T, typename U, typename Proj = impl::get_value>
util::pw_elements<U> pw_over_cable(const mcable_map<T>& mm, mcable cable, U dflt_value, Proj projection = Proj{}) {
    using value_type = typename mcable_map<T>::value_type;
    msize_t bid = cable.branch;

    struct as_branch {
        msize_t value;
        as_branch(const value_type& x): value(x.first.branch) {}
        as_branch(const msize_t& x): value(x) {}
    };

    auto map_on_branch = util::make_range(
        std::equal_range(mm.begin(), mm.end(), bid, [](as_branch a, as_branch b) { return a.value<b.value; }));

    if (map_on_branch.empty()) {
        return util::pw_elements<U>({cable.prox_pos, cable.dist_pos}, {dflt_value});
    }

    util::pw_elements<U> pw;
    pw.reserve(2*map_on_branch.size());
    double pw_right = 0.0;
    for (const auto& [elf, els]: map_on_branch) {
        if (elf.prox_pos > pw_right) {
            pw.push_back(pw_right, elf.prox_pos, dflt_value);
        }
        pw.push_back(elf.prox_pos, elf.dist_pos, projection(elf, els));
        pw_right = pw.upper_bound();
    }

    if (pw_right < 1.0) {
        pw.push_back(pw_right, 1.0, dflt_value);
    }

    if (cable.prox_pos!=0 || cable.dist_pos!=1) {
        pw = util::pw_zip_with(pw, util::pw_elements<>({cable.prox_pos, cable.dist_pos}));
    }
    return pw;
}
} // namespace util
} // namespace arb
