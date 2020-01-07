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

class region {
public:
    template <typename Impl,
              typename X=std::enable_if_t<!std::is_same<std::decay_t<Impl>, region>::value>>
    explicit region(Impl&& impl):
        impl_(new wrap<Impl>(std::forward<Impl>(impl))) {}

    template <typename Impl>
    explicit region(const Impl& impl):
        impl_(new wrap<Impl>(impl)) {}

    region(region&& other) = default;

    // The default constructor creates an empty "nil" region.
    region();

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

    // Implicitly convert string to named region expression.
    region(std::string label);
    region(const char* label);

    friend mcable_list thingify(const region& r, const mprovider& m) {
        return r.impl_->thingify(m);
    }

    friend std::ostream& operator<<(std::ostream& o, const region& p) {
        return p.impl_->print(o);
    }

    // The union of regions.
    friend region join(region, region);

    template <typename ...Args>
    friend region join(region l, region r, Args... args) {
        return join(join(std::move(l), std::move(r)), std::move(args)...);
    }

    // The intersection of regions.
    friend region intersect(region, region);

    template <typename ...Args>
    friend region intersect(region l, region r, Args... args) {
        return intersect(intersect(std::move(l), std::move(r)), std::move(args)...);
    }

private:
    struct interface {
        virtual ~interface() {}
        virtual std::unique_ptr<interface> clone() = 0;
        virtual std::ostream& print(std::ostream&) = 0;
        virtual mcable_list thingify(const mprovider&) = 0;
    };

    std::unique_ptr<interface> impl_;

    template <typename Impl>
    struct wrap: interface {
        explicit wrap(const Impl& impl): wrapped(impl) {}
        explicit wrap(Impl&& impl): wrapped(std::move(impl)) {}

        virtual std::unique_ptr<interface> clone() override {
            return std::unique_ptr<interface>(new wrap<Impl>(wrapped));
        }

        virtual mcable_list thingify(const mprovider& m) override {
            return thingify_(wrapped, m);
        }

        virtual std::ostream& print(std::ostream& o) override {
            return o << wrapped;
        }

        Impl wrapped;
    };
};

namespace reg {

// An empty region.
region nil();

// An explicit cable section.
region cable(mcable);

region interval(mlocation, mlocation);

// An explicit branch.
region branch(msize_t);

// Region with all segments with segment tag id.
region tagged(int id);

// Region with all segments in a cell.
region all();

// Region associated with a name.
region named(std::string);

} // namespace reg

} // namespace arb
