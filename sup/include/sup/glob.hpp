#pragma once

// Glob implementation via glob (3) wrapper or fallback implementation.

#include <functional>
#include <string>
#include <vector>

#include <sup/path.hpp>

namespace sup {

// Wrapper (provided by either glob_posix.cpp or
// glob_basic_wrapper.cpp based on configuration.)

std::vector<path> glob(const std::string& pattern);

// Basic globber.
//
// Uses `glob_fs_provider` to provide required filesystem
// operations, defaults to implementation based around
// sup-provided directory iterators.

struct glob_fs_provider {
    using action_type = std::function<void (const sup::path&)>;

    template <typename Impl>
    glob_fs_provider(Impl impl): inner_(new wrap<Impl>(std::move(impl))) {}

    glob_fs_provider(const glob_fs_provider& x): inner_(x.inner_->clone()) {}

    bool is_directory(const sup::path& p) const {
        return inner_->is_directory(p);
    }

    bool exists(const sup::path& p) const {
        return inner_->exists(p);
    }

    void for_each_directory(const sup::path& p, action_type action) const {
        inner_->for_each_directory(p, action);
    }

    void for_each_entry(const sup::path& p, action_type action) const {
        inner_->for_each_entry(p, action);
    }

private:
    struct base {
        virtual bool is_directory(const sup::path&) const = 0;
        virtual bool exists(const sup::path&) const = 0;
        virtual void for_each_directory(const sup::path&, action_type action) const = 0;
        virtual void for_each_entry(const sup::path&, action_type action) const = 0;
        virtual base* clone() const = 0;
        virtual ~base() {}
    };

    template <typename Impl>
    struct wrap: base {
        wrap(Impl impl): impl_(std::move(impl)) {}

        bool is_directory(const sup::path& p) const override {
            return impl_.is_directory(p);
        }

        bool exists(const sup::path& p) const override {
            return impl_.exists(p);
        }

        void for_each_directory(const sup::path& p, action_type action) const override {
            impl_.for_each_directory(p, action);
        }

        void for_each_entry(const sup::path& p, action_type action) const override {
            impl_.for_each_entry(p, action);
        }

        base* clone() const override {
            return new wrap(impl_);
        }

        Impl impl_;
    };

    std::unique_ptr<base> inner_;
};

extern glob_fs_provider glob_native_provider;

std::vector<path> glob_basic(const std::string& pattern, const glob_fs_provider& = glob_native_provider);

// Expose glob filename expression matcher for unit testing.
//
// Follows glob(7) description except for:
// * No character class support, e.g. [:alpha:].
// * Ignores LC_COLLATE for character ranges, and does not accommodate multibyte encodings.

bool glob_basic_match(const char* p, const char* t);

} // namespace sup

