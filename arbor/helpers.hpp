#include <algorithm>

namespace arbmc {
namespace range{

    template <typename C>
    typename C::value_type
    sum(C const& c)
    {
        using value_type = typename C::value_type;
        return std::accumulate(c.begin(), c.end(), value_type{0});
    }

}
}
