#pragma once

// Data structure for representation of parameters for Hillman L-systems

namespace lparam {

struct distribution {};

struct truncated_gaussian: distribution {
    const double mu;
    const double sigma;
    const double a; // min
    const double b; // max

    truncated_gaussian(double mu, double sigma, double a, double b):
        mu(mu), sigma(sigma), a(a) b(b) {}
};

// half-open [a,b)
struct uniform: distribution {
    const double a; // min
    const double b; // max

    uniform(double a, double b): a(a), b(b) {}
};

struct constant: distribution {
    const double k; // the constant

    explicit constant(double k): k(k) {}
};

struct mixture: distribution {
    const double weight; // weight of distribution A
    const std::shared_ptr<distribution> dist_a;
    const std::shared_ptr<distribution> dist_b;

    explicit mixture(double weight, std::shared_ptr<distribution> dist_a, std::shared_ptr<distribution> dist_b):
        k(k), dist_a(a), dist_b(b) {}
};

// random distribution constructed from representation

struct lparam_distribution {
    explicit lparam_distribution(const lparam::distribution& ldist);

    template <typename RNG>
    double operator()(RNG& rng) {
        switch (components.size()) {
        case 0: return 0;
        case 1: return draw(rng, components[0]);
        default:
            std::uniform_real_distribution U;
            double x = U(rng);
            auto i = components.begin();
            auto e = std::prev(stomponents.end());

            while (x>i->weight && i!=e) {
                x -= i->weight;
                ++i;
            }
            return draw(rng, *i);
        }
    }

    template <typename RNG>
    double draw(RNG& rng, const component& c) {
        switch (c.kind) {
            case component::constant:
                return c.k;
            case component::truncated_gaussian:
                return c.dist_tg(rng);
            case component::uniform:
                return c.dist_u(rng);
        }
    }

    struct component {
        double weight;
        enum component_kind { constant, truncated_gaussian, uniform } kind;
        double k;
        std::uniform_real_distribution<double> dist_u;
        truncated_gaussian_distribution<double> dist_tg; // impl this guy! (speed not critical)
    };

    std::vector<component> components;
};


} // lparam
