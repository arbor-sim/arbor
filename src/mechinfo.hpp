#pragma once

/* Mechanism schema classes, catalogue and parameter specification.
 *
 * Catalogue and schemata have placeholder implementations, to be
 * completed in future work.
 */

#include <string>
#include <utility>
#include <vector>

namespace arb {

struct mechanism_schema_field {
    void validate(double) const {}
    double default_value = 0;
};

struct mechanism_schema {
    const mechanism_schema_field* field(const std::string& key) const {
        static mechanism_schema_field dummy_field;
        return &dummy_field;
    }

    static const mechanism_schema* dummy_schema() {
        static mechanism_schema d;
        return &d;
    }
};

class mechanism_spec {
public:
    struct field_proxy {
        mechanism_spec* m;
        std::string key;

        field_proxy& operator=(double v) {
            m->set(key, v);
            return *this;
        }

        operator double() const {
            return m->get(key);
        }
    };

    // implicit
    mechanism_spec(std::string name): name_(std::move(name)) {
        // get schema pointer from global catalogue, or throw
        schema_ = mechanism_schema::dummy_schema();
        if (!schema_) {
            throw std::runtime_error("no mechanism "+name_);
        }
    }

    // implicit
    mechanism_spec(const char* name): mechanism_spec(std::string(name)) {}

    mechanism_spec& set(std::string key, double value) {
        auto field_schema = schema_->field(key);
        if (!field_schema) {
            throw std::runtime_error("no field "+key+" in mechanism "+name_);
        }

        field_schema->validate(value);
        param_[key] = value;
        return *this;
    }

    double operator[](const std::string& key) const {
        return get(key);
    }

    double get(const std::string& key) const {
        auto field_schema = schema_->field(key);
        if (!field_schema) {
            throw std::runtime_error("no field "+key+" in mechanism "+name_);
        }

        auto it = param_.find(key);
        return it==param_.end()? field_schema->default_value: it->second;
    }

    field_proxy operator[](const std::string& key) {
        return {this, key};
    }

    const std::map<std::string, double>& values() const {
        return param_;
    }

    const std::string& name() const { return name_; }

private:
    std::string name_;
    std::map<std::string, double> param_;
    const mechanism_schema* schema_; // non-owning; schema must have longer lifetime
};

} // namespace arb
