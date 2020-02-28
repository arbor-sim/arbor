#include <sstream>
#include <unordered_map>
#include <vector>

#include <arbor/cable_cell.hpp>
#include <arbor/morph/label_dict.hpp>
#include <arbor/morph/morphology.hpp>
#include <arbor/morph/mprovider.hpp>
#include <arbor/util/pp_util.hpp>

#include "util/piecewise.hpp"
#include "util/rangeutil.hpp"
#include "util/span.hpp"
#include "util/strprintf.hpp"

namespace arb {

using region_map = std::unordered_map<std::string, mcable_list>;
using locset_map = std::unordered_map<std::string, mlocation_list>;

using value_type = cable_cell::value_type;
using index_type = cable_cell::index_type;
using size_type = cable_cell::size_type;

template <typename T> struct constant_type {
    template <typename> using type = T;
};

struct cable_cell_impl {
    using value_type = cable_cell::value_type;
    using index_type = cable_cell::index_type;
    using size_type  = cable_cell::size_type;

    cable_cell_impl(const arb::morphology& m, const label_dict& dictionary):
        provider(m, dictionary)
    {}

    cable_cell_impl(): cable_cell_impl({},{}) {}

    cable_cell_impl(const cable_cell_impl& other):
        provider(other.provider),
        region_map(other.region_map),
        location_map(other.location_map)
    {}

    cable_cell_impl(cable_cell_impl&& other) = default;

    // Embedded morphology and labelled region/locset lookup.
    mprovider provider;

    // Regional assignments.
    cable_cell_region_map region_map;

    // Point assignments.
    cable_cell_location_map location_map;

    // Track number of point assignments by type for lid/target numbers.
    dynamic_typed_map<constant_type<cell_lid_type>::type> placed_count;

    template <typename T>
    mlocation_map<T>& get_location_map(const T&) {
        return location_map.get<T>();
    }

    mlocation_map<mechanism_desc>& get_location_map(const mechanism_desc& desc) {
        return location_map.get<mechanism_desc>()[desc.name()];
    }

    template <typename Item>
    lid_range place(const locset& ls, const Item& item) {
        auto& mm = get_location_map(item);
        cell_lid_type& lid = placed_count.get<Item>();
        cell_lid_type first = lid;

        for (auto l: thingify(ls, provider)) {
            placed<Item> p{l, lid++, item};
            mm.push_back(p);
        }
        return lid_range(first, lid);
    }

    template <typename T>
    mcable_map<T>& get_region_map(const T&) {
        return region_map.get<T>();
    }

    mcable_map<mechanism_desc>& get_region_map(const mechanism_desc& desc) {
        return region_map.get<mechanism_desc>()[desc.name()];
    }

    mcable_map<initial_ion_data>& get_region_map(const initial_ion_data& init) {
        return region_map.get<initial_ion_data>()[init.ion];
    }

    template <typename Property>
    void paint(const region& reg, const Property& prop) {
        mextent cables = thingify(reg, provider);
        auto& mm = get_region_map(prop);

        for (auto c: cables) {
            // Skip zero-length cables in extent:
            if (c.prox_pos==c.dist_pos) continue;

            if (!mm.insert(c, prop)) {
                throw cable_cell_error(util::pprintf("cable {} overpaints", c));
            }
        }
    }

    mlocation_list concrete_locset(const locset& l) const {
        return thingify(l, provider);
    }

    mextent concrete_region(const region& r) const {
        return thingify(r, provider);
    }
};

using impl_ptr = std::unique_ptr<cable_cell_impl, void (*)(cable_cell_impl*)>;
impl_ptr make_impl(cable_cell_impl* c) {
    return impl_ptr(c, [](cable_cell_impl* p){delete p;});
}

cable_cell::cable_cell(const arb::morphology& m, const label_dict& dictionary):
    impl_(make_impl(new cable_cell_impl(m, dictionary)))
{}

cable_cell::cable_cell(): impl_(make_impl(new cable_cell_impl())) {}

cable_cell::cable_cell(const cable_cell& other):
    default_parameters(other.default_parameters),
    impl_(make_impl(new cable_cell_impl(*other.impl_)))
{}

const concrete_embedding& cable_cell::embedding() const {
    return impl_->provider.embedding();
}

const arb::morphology& cable_cell::morphology() const {
    return impl_->provider.morphology();
}

const mprovider& cable_cell::provider() const {
    return impl_->provider;
}

mlocation_list cable_cell::concrete_locset(const locset& l) const {
    return impl_->concrete_locset(l);
}

mextent cable_cell::concrete_region(const region& r) const {
    return impl_->concrete_region(r);
}

const cable_cell_location_map& cable_cell::location_assignments() const {
    return impl_->location_map;
}

const cable_cell_region_map& cable_cell::region_assignments() const {
    return impl_->region_map;
}

// Forward paint methods to implementation class.

#define FWD_PAINT(proptype)\
void cable_cell::paint(const region& target, proptype prop) {\
    impl_->paint(target, prop);\
}
ARB_PP_FOREACH(FWD_PAINT,\
    mechanism_desc, init_membrane_potential, axial_resistivity,\
    temperature_K, membrane_capacitance, initial_ion_data)

// Forward place methods to implementation class.

#define FWD_PLACE(proptype)\
lid_range cable_cell::place(const locset& target, proptype prop) {\
    return impl_->place(target, prop);\
}
ARB_PP_FOREACH(FWD_PLACE,\
    mechanism_desc, i_clamp, gap_junction_site, threshold_detector)

} // namespace arb
