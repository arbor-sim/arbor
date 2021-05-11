#include <sstream>
#include <unordered_map>
#include <variant>
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

    // Embedded morphology and labelled region/locset lookup.
    mprovider provider;

    // Regional assignments.
    cable_cell_region_map region_map;

    // Point assignments.
    cable_cell_location_map location_map;

    // Track number of point assignments by type for lid/target numbers.
    dynamic_typed_map<constant_type<cell_lid_type>::type> placed_count;

    // The label dictionary.
    const label_dict dictionary;

    // The decorations on the cell.
    decor decorations;

    // The placeable label to lid_range map
    dynamic_typed_map<constant_type<std::unordered_multimap<cell_tag_type, lid_range>>::type> labeled_lid_ranges;

    cable_cell_impl(const arb::morphology& m, const label_dict& labels, const decor& decorations):
        provider(m, labels),
        dictionary(labels),
        decorations(decorations)
    {
        init(decorations);
    }

    cable_cell_impl(): cable_cell_impl({},{},{}) {}

    cable_cell_impl(const cable_cell_impl& other) = default;

    cable_cell_impl(cable_cell_impl&& other) = default;

    void init(const decor&);

    template <typename T>
    mlocation_map<T>& get_location_map(const T&) {
        return location_map.get<T>();
    }

    mlocation_map<mechanism_desc>& get_location_map(const mechanism_desc& desc) {
        return location_map.get<mechanism_desc>()[desc.name()];
    }

    template <typename Item>
    void place(const locset& ls, const Item& item, const cell_tag_type& label) {
        auto& mm = get_location_map(item);
        cell_lid_type& lid = placed_count.get<Item>();
        cell_lid_type first = lid;

        for (auto l: thingify(ls, provider)) {
            placed<Item> p{l, lid++, item};
            mm.push_back(p);
        }
        auto range = lid_range(first, lid);
        auto& lid_ranges = labeled_lid_ranges.get<Item>();
        lid_ranges.insert(std::make_pair(label, range));
    }

    template <typename T>
    mcable_map<T>& get_region_map(const T&) {
        return region_map.get<T>();
    }

    mcable_map<mechanism_desc>& get_region_map(const mechanism_desc& desc) {
        return region_map.get<mechanism_desc>()[desc.name()];
    }

    mcable_map<init_int_concentration>& get_region_map(const init_int_concentration& init) {
        return region_map.get<init_int_concentration>()[init.ion];
    }

    mcable_map<init_ext_concentration>& get_region_map(const init_ext_concentration& init) {
        return region_map.get<init_ext_concentration>()[init.ion];
    }

    mcable_map<init_reversal_potential>& get_region_map(const init_reversal_potential& init) {
        return region_map.get<init_reversal_potential>()[init.ion];
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

void cable_cell_impl::init(const decor& d) {
    for (const auto& p: d.paintings()) {
        auto& where = p.first;
        std::visit([this, &where] (auto&& what) {this->paint(where, what);}, p.second);
    }
    for (const auto& p: d.placements()) {
        auto& where = std::get<0>(p);
        auto& label = std::get<2>(p);
        std::visit([this, &where, &label] (auto&& what) {return this->place(where, what, label);}, std::get<1>(p));
    }
}

cable_cell::cable_cell(const arb::morphology& m, const label_dict& dictionary, const decor& decorations):
    impl_(make_impl(new cable_cell_impl(m, dictionary, decorations)))
{}

cable_cell::cable_cell(): impl_(make_impl(new cable_cell_impl())) {}

cable_cell::cable_cell(const cable_cell& other):
    impl_(make_impl(new cable_cell_impl(*other.impl_)))
{}

const label_dict& cable_cell::labels() const {
    return impl_->dictionary;
}

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

const decor& cable_cell::decorations() const {
    return impl_->decorations;
}

const cable_cell_parameter_set& cable_cell::default_parameters() const {
    return impl_->decorations.defaults();
}

const std::unordered_multimap<cell_tag_type, lid_range>& cable_cell::detector_ranges() const {
    return impl_->labeled_lid_ranges.get<threshold_detector>();
}

const std::unordered_multimap<cell_tag_type, lid_range>& cable_cell::synapse_ranges() const {
    return impl_->labeled_lid_ranges.get<mechanism_desc>();
}

const std::unordered_multimap<cell_tag_type, lid_range>& cable_cell::gap_junction_ranges() const {
    return impl_->labeled_lid_ranges.get<gap_junction_site>();
}

} // namespace arb
