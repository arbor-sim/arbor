#pragma once

#include <cstdlib>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <arbor/morph/error.hpp>
#include <arbor/morph/primitives.hpp>
#include <arbor/morph/morphology.hpp>

namespace arb {

// Dorward declarations of locset, region and their respective dictionaries,
// which are required to define the interface for region.
class locset;
class region;
using locset_dictionary = std::unordered_map<std::string, locset>;
using region_dictionary = std::unordered_map<std::string, region>;

// Forward declare the backend em_morphology type, required for defining the
// interface for concretising locsets.
class em_morphology;

class region {
public:
    region();

    template <typename Impl,
              typename X=std::enable_if_t<!std::is_same<std::decay_t<Impl>, region>::value>>
    explicit region(Impl&& impl):
        impl_(new wrap<Impl>(std::forward<Impl>(impl))) {}

    template <typename Impl>
    explicit region(const Impl& impl):
        impl_(new wrap<Impl>(impl)) {}

    region(region&& other) = default;

    region(const region& other):
        impl_(other.impl_->clone()) {}

    region& operator=(const region& other) {
        impl_ = other.impl_->clone();
        return *this;
    }

    template <typename Impl,
              typename X=std::enable_if_t<!std::is_same<std::decay_t<Impl>, region>::value>>
    region& operator=(Impl&& other) {
        impl_ = new wrap<Impl>(std::forward<Impl>(other));
        return *this;
    }

    template <typename Impl>
    region& operator=(const Impl& other) {
        impl_ = new wrap<Impl>(other);
        return *this;
    }

    friend mcable_list concretise(const region& r, const em_morphology& m) {
        return r.impl_->concretise(m);
    }

    friend std::ostream& operator<<(std::ostream& o, const region& p) {
        return p.impl_->print(o);
    }

    // The union of two regions.
    friend region or_(region, region);

    // The intersection of two regions.
    friend region and_(region, region);

private:
    struct interface {
        virtual ~interface() {}
        virtual std::unique_ptr<interface> clone() = 0;
        virtual std::ostream& print(std::ostream&) = 0;
        virtual mcable_list concretise(const em_morphology&) = 0;
    };

    std::unique_ptr<interface> impl_;

    template <typename Impl>
    struct wrap: interface {
        explicit wrap(const Impl& impl): wrapped(impl) {}
        explicit wrap(Impl&& impl): wrapped(std::move(impl)) {}

        virtual std::unique_ptr<interface> clone() override {
            return std::unique_ptr<interface>(new wrap<Impl>(wrapped));
        }

        virtual mcable_list concretise(const em_morphology& m) override {
            return concretise_(wrapped, m);
        }

        virtual std::ostream& print(std::ostream& o) override {
            return o << wrapped;
        }

        Impl wrapped;
    };
};

namespace reg {


// An explicit cable section.
region cable(mcable);

// An explicit list of cable sections.
region cable_list(mcable_list);

// An explicit branch.
region branch(msize_t);

// Region with all segments with segment tag id.
region tagged(int id);

// Region with all segments in a cell.
region all();

} // namespace reg

} // namespace arb
