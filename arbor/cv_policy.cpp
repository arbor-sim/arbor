#include <utility>
#include <ostream>
#include <vector>

#include <arbor/cable_cell.hpp>
#include <arbor/cv_policy.hpp>
#include <arbor/morph/locset.hpp>
#include <arbor/morph/region.hpp>

#include "util/rangeutil.hpp"

namespace arb {

static std::string print_flag(cv_policy_flag flag) {
    switch (flag) {
        case arb::cv_policy_flag::none: return "(flag-none)";
        case arb::cv_policy_flag::interior_forks: return "(flag-interior-forks)";
    }
    throw std::runtime_error("UNREACHABLE");
}

static bool has_flag(cv_policy_flag lhs, cv_policy_flag rhs) {
    return static_cast<bool>(static_cast<unsigned>(lhs) & static_cast<unsigned>(rhs));
}


static auto unique_sum = [](auto&&... lss) {
    return ls::support(sum(std::forward<decltype(lss)>(lss)...));
};

struct cvp_cv_policy_plus {
    locset cv_boundary_points(const cable_cell& c) const {
        return unique_sum(lhs_.cv_boundary_points(c), rhs_.cv_boundary_points(c));
    }

    region domain() const { return join(lhs_.domain(), rhs_.domain()); }

    std::ostream& format(std::ostream& os) const {
        os << "(join " << lhs_ << ' ' << rhs_ << ')';
        return os;
    }

    // TODO This is needed seemingly only for older compilers
    cvp_cv_policy_plus(cv_policy lhs, cv_policy rhs): lhs_{std::move(lhs)}, rhs_(std::move(rhs)) {}

    cv_policy lhs_, rhs_;
};

ARB_ARBOR_API cv_policy operator+(const cv_policy& lhs, const cv_policy& rhs) {
    return cv_policy{cvp_cv_policy_plus(lhs, rhs)};
}

struct cvp_cv_policy_bar {
    locset cv_boundary_points(const cable_cell& c) const {
        return unique_sum(ls::restrict_to(lhs_.cv_boundary_points(c),
                                          complement(rhs_.domain())),
                          rhs_.cv_boundary_points(c));
    }

    region domain() const { return join(lhs_.domain(), rhs_.domain()); }

    std::ostream& format(std::ostream& os) const {
        os << "(replace " << lhs_ << ' ' << rhs_ << ')';
        return os;
    }

    // TODO This is needed seemingly only for older compilers
    cvp_cv_policy_bar(cv_policy lhs, cv_policy rhs): lhs_{std::move(lhs)}, rhs_(std::move(rhs)) {}

    cv_policy lhs_, rhs_;
};

ARB_ARBOR_API cv_policy operator|(const cv_policy& lhs, const cv_policy& rhs) {
    return cv_policy{cvp_cv_policy_bar(lhs, rhs)};
}

struct cvp_cv_policy_max_extent {
    double max_extent_;
    region domain_;
    cv_policy_flag flags_;

    std::ostream& format(std::ostream& os) const {
        os << "(max-extent " << max_extent_ << ' ' << domain_ << ' ' << print_flag(flags_) << ')';
        return os;
    }

    locset cv_boundary_points(const cable_cell& cell) const {
        const unsigned nbranch = cell.morphology().num_branches();
        const auto& embed = cell.embedding();
        if (!nbranch || max_extent_<=0) return ls::nil();

        std::vector<mlocation> points;
        double oomax_extent = 1./max_extent_;
        auto comps = components(cell.morphology(), thingify(domain_, cell.provider()));

        for (auto& comp: comps) {
            for (mcable c: comp) {
                double cable_length = embed.integrate_length(c);
                unsigned ncv = std::ceil(cable_length*oomax_extent);
                double scale = (c.dist_pos-c.prox_pos)/ncv;

                if (has_flag(flags_, cv_policy_flag::interior_forks)) {
                    for (unsigned i = 0; i<ncv; ++i) {
                        points.push_back({c.branch, c.prox_pos+(1+2*i)*scale/2});
                    }
                }
                else {
                    for (unsigned i = 0; i<ncv; ++i) {
                        points.push_back({c.branch, c.prox_pos+i*scale});
                    }
                    points.push_back({c.branch, c.dist_pos});
                }
            }
        }

        util::sort(points);
        return unique_sum(locset(std::move(points)), ls::cboundary(domain_));
    }

    region domain() const { return domain_; }
};

ARB_ARBOR_API cv_policy cv_policy_max_extent(double ext, region reg, cv_policy_flag flag) {
    return cv_policy{cvp_cv_policy_max_extent{ext, std::move(reg), flag}};
}

ARB_ARBOR_API cv_policy cv_policy_max_extent(double ext, cv_policy_flag flag) {
    return cv_policy{cvp_cv_policy_max_extent{ext, reg::all(), flag}};
}

struct cvp_cv_policy_explicit {
    locset locs_;
    region domain_;

    std::ostream& format(std::ostream& os) const {
        os << "(explicit " << locs_ << ' ' << domain_ << ')';
        return os;
    }

    locset cv_boundary_points(const cable_cell& cell) const {
        return ls::support(
            util::foldl(
                [this](locset l, const auto& comp) {
                    return sum(std::move(l), ls::restrict_to(locs_, comp));
                },
                ls::boundary(domain_),
                components(cell.morphology(), thingify(domain_, cell.provider()))));
    }

    region domain() const { return domain_; }
};

ARB_ARBOR_API cv_policy cv_policy_explicit(locset ls, region reg) {
    return cv_policy{cvp_cv_policy_explicit{std::move(ls), std::move(reg)}};
}

struct cvp_cv_policy_single {
    region domain() const { return domain_; }

    std::ostream& format(std::ostream& os) const {
        os << "(single " << domain_ << ')';
        return os;
    }

    locset cv_boundary_points(const cable_cell&) const {
        return ls::cboundary(domain_);
    }

    region domain_;
};

ARB_ARBOR_API cv_policy cv_policy_single(region reg) {
    return cv_policy{cvp_cv_policy_single{std::move(reg)}};
}
    
struct cvp_cv_policy_fixed_per_branch {
    region domain() const { return domain_; }

    std::ostream& format(std::ostream& os) const {
        os << "(fixed-per-branch " << cv_per_branch_ << ' ' << domain_ << ' ' << print_flag(flags_) << ')';
        return os;
    }

    locset cv_boundary_points(const cable_cell& cell) const {
        const unsigned nbranch = cell.morphology().num_branches();
        if (!nbranch) return ls::nil();

        std::vector<mlocation> points;
        double ooncv = 1./cv_per_branch_;
        auto comps = components(cell.morphology(), thingify(domain_, cell.provider()));

        for (auto& comp: comps) {
            for (mcable c: comp) {
                double scale = (c.dist_pos-c.prox_pos)*ooncv;

                if (has_flag(flags_, cv_policy_flag::interior_forks)) {
                    for (unsigned i = 0; i<cv_per_branch_; ++i) {
                        points.push_back({c.branch, c.prox_pos+(1+2*i)*scale/2});
                    }
                }
                else {
                    for (unsigned i = 0; i<cv_per_branch_; ++i) {
                        points.push_back({c.branch, c.prox_pos+i*scale});
                    }
                    points.push_back({c.branch, c.dist_pos});
                }
            }
        }

        util::sort(points);
        return unique_sum(locset(std::move(points)), ls::cboundary(domain_));
    }

    unsigned cv_per_branch_;
    region domain_;
    cv_policy_flag flags_;
};

ARB_ARBOR_API cv_policy cv_policy_fixed_per_branch(unsigned n, region reg, cv_policy_flag flag) {
    return cv_policy{cvp_cv_policy_fixed_per_branch{n, std::move(reg), flag}};
}

ARB_ARBOR_API cv_policy cv_policy_fixed_per_branch(unsigned n, cv_policy_flag flag) {
    return cv_policy{cvp_cv_policy_fixed_per_branch{n, reg::all(), flag}};
}

struct cvp_cv_policy_every_segment {
    region domain() const { return domain_; }

    std::ostream& format(std::ostream& os) const {
        os << "(every-segment " << domain_ << ')';
        return os;
    }

    locset cv_boundary_points(const cable_cell& cell) const {
        const unsigned nbranch = cell.morphology().num_branches();
        if (!nbranch) return ls::nil();

        return unique_sum(
            ls::cboundary(domain_),
            ls::restrict_to(ls::segment_boundaries(), domain_));
    }

    region domain_;
};

ARB_ARBOR_API cv_policy cv_policy_every_segment(region reg) {
    return cv_policy{cvp_cv_policy_every_segment{std::move(reg)}};
}

} // namespace arb
