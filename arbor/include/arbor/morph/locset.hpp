#pragma once

#include <cstdlib>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <arbor/morph/primitives.hpp>
#include <arbor/morph/morphology.hpp>

namespace arb {

// Forward declarations of locset, region and their respective dictionaries,
// which are required to define the interface for locset.
class locset;
class region;
using locset_dictionary = std::unordered_map<std::string, locset>;
using region_dictionary = std::unordered_map<std::string, region>;

// Forward declare the backend em_morphology type, required for defining the
// interface for concretising locsets.
class em_morphology;

class locset {
public:
    locset();

    template <typename Impl,
              typename X=std::enable_if_t<!std::is_same<std::decay_t<Impl>, locset>::value>>
    explicit locset(Impl&& impl):
        impl_(new wrap<Impl>(std::forward<Impl>(impl))) {}

    template <typename Impl>
    explicit locset(const Impl& impl):
        impl_(new wrap<Impl>(impl)) {}

    locset(locset&& other) = default;

    locset(const locset& other):
        impl_(other.impl_->clone()) {}

    locset& operator=(const locset& other) {
        impl_ = other.impl_->clone();
        return *this;
    }

    template <typename Impl,
              typename X=std::enable_if_t<!std::is_same<std::decay_t<Impl>, locset>::value>>
    locset& operator=(Impl&& other) {
        impl_ = new wrap<Impl>(std::forward<Impl>(other));
        return *this;
    }
    template <typename Impl>
    locset& operator=(const Impl& other) {
        impl_ = new wrap<Impl>(other);
        return *this;
    }

    friend mlocation_list concretise(const locset& p, const em_morphology& m) {
        return p.impl_->concretise(m);
    }

    friend std::set<std::string> named_dependencies(const locset& p) {
      return p.impl_->named_dependencies();
    }

    friend locset replace_named_dependencies(const locset& p, const region_dictionary& reg_dict, const locset_dictionary& ps_dict) {
      return p.impl_->replace_named_dependencies(reg_dict, ps_dict);
    }

    friend std::ostream& operator<<(std::ostream& o, const locset& p) {
        return p.impl_->print(o);
    }

private:
    struct interface {
        virtual ~interface() {}
        virtual std::unique_ptr<interface> clone() = 0;
        virtual std::ostream& print(std::ostream&) = 0;
        virtual mlocation_list concretise(const em_morphology&) = 0;
        virtual std::set<std::string> named_dependencies() = 0;
        virtual locset replace_named_dependencies(const region_dictionary&, const locset_dictionary&) = 0;
    };

    std::unique_ptr<interface> impl_;

    template <typename Impl>
    struct wrap: interface {
        explicit wrap(const Impl& impl): wrapped(impl) {}
        explicit wrap(Impl&& impl): wrapped(std::move(impl)) {}

        virtual std::unique_ptr<interface> clone() override {
            return std::unique_ptr<interface>(new wrap<Impl>(wrapped));
        }

        virtual mlocation_list concretise(const em_morphology& m) override {
            return do_concretise(wrapped, m);
        }

        virtual std::set<std::string> named_dependencies() override {
            return do_named_dependencies(wrapped);
        }

        virtual locset replace_named_dependencies(const region_dictionary& r, const locset_dictionary& p) override {
            return do_replace_named_dependencies(wrapped, r, p);
        }

        virtual std::ostream& print(std::ostream& o) override {
            return o << wrapped;
        }

        Impl wrapped;
    };
};

namespace ls {

// Location of a sample.
locset location(mlocation);

// Location of a sample.
locset sample(msize_t);

// Set of terminal nodes on a morphology.
locset terminal();

// The root node of a morphology.
locset root();

// A named pointset.
locset named(std::string);

// The union of two location sets.
locset lor(locset, locset);

// The intersection of two location sets.
locset land(locset, locset);

} // namespace ps

} // namespace arb
