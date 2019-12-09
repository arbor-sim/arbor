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

struct mprovider;

class locset;

class locset {
public:
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

    // The default constructor creates an empty "nil" set.
    locset();

    // Construct an explicit location set with a single location.
    locset(mlocation other);

    // Implicitly convert string to named locset expression.
    locset(std::string label);
    locset(const char* label);

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

    friend mlocation_list thingify(const locset& p, const mprovider& m) {
        return p.impl_->thingify(m);
    }

    friend std::ostream& operator<<(std::ostream& o, const locset& p) {
        return p.impl_->print(o);
    }

    // The sum of two location sets.
    friend locset sum(locset, locset);

    template <typename ...Args>
    friend locset sum(locset l, locset r, Args... args) {
        return sum(sum(std::move(l), std::move(r)), std::move(args)...);
    }

    // The union of two location sets.
    friend locset join(locset, locset);

    template <typename ...Args>
    friend locset join(locset l, locset r, Args... args) {
        return join(join(std::move(l), std::move(r)), std::move(args)...);
    }

private:
    struct interface {
        virtual ~interface() {}
        virtual std::unique_ptr<interface> clone() = 0;
        virtual std::ostream& print(std::ostream&) = 0;
        virtual mlocation_list thingify(const mprovider&) = 0;
    };

    std::unique_ptr<interface> impl_;

    template <typename Impl>
    struct wrap: interface {
        explicit wrap(const Impl& impl): wrapped(impl) {}
        explicit wrap(Impl&& impl): wrapped(std::move(impl)) {}

        virtual std::unique_ptr<interface> clone() override {
            return std::unique_ptr<interface>(new wrap<Impl>(wrapped));
        }

        virtual mlocation_list thingify(const mprovider& m) override {
            return thingify_(wrapped, m);
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

// Named locset.
locset named(std::string);

// The null (empty) set.
locset nil();

} // namespace ls

} // namespace arb
