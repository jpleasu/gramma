#include <iostream>
#include <string>
#include <utility>
#include <variant>

#include "gramma/gramma.hpp"
#include "gramma/sample.hpp"

class variety_sampler;

// implementation declaration
using char_t = char;

using string_t = std::basic_string<char_t>;
// using char_t = wchar_t;

struct denotation_t : public std::variant<int, double, string_t> {
    using base_type = std::variant<int, double, string_t>;
    using base_type::variant;
};

template <class T>
string_t str(T &&arg) {
    return std::to_string(std::forward<T>(arg));
    // return std::to_wstring(std::forward<T>(arg));
};

template <>
string_t str<denotation_t &>(denotation_t &d) {
    switch (d.index()) {
    case 0:
        return str(std::get<int>(d));
    case 1:
        return str(std::get<double>(d));
    case 2:
        return std::get<string_t>(d);
    }
    return {};
}

using sample_t = gramma::basic_sample<denotation_t, char>;

class variety_sampler_impl : public gramma::SamplerBase<variety_sampler, sample_t> {
  public:
    using base_type = gramma::SamplerBase<variety_sampler, sample_t>;
    using trace_type = bool;

    // sampler API
    void icat(sample_t &a, const sample_t &b) {
        a += b;
    }
    sample_t denote(const sample_t &a, const denotation_t &b) {
        return sample_t(a, b);
    }

    // gfuncs

    // for testing
    sample_t show_den_lazy(sample_factory_type m) {
        auto a = m();
        return sample_t(a + "<" + str(a.d) + ">", a.d);
    }

    // this non-lazy form of the previous gfunc relies on a
    // callable-converting constructor of sample_t
    sample_t show_den(sample_t a) {
        return sample_t(a + "<" + str(a.d) + ">", a.d);
    }

    sample_t x2(sample_factory_type m) {
        auto a = m();
        icat(a, a);
        return a;
    }
    sample_type ff();
    sample_type f();

    // gdfuncs

    // sample_t implicity passed via string_t variant constructor
    denotation_t get_den(sample_t s) {
        return s.d;
    }

    int rule_depth = 1000;
    bool a = false;
    int a_func() {
        return a ? 2 : 1;
    }
};
#include "variety_sampler_decl.inc"

// implementation definition
variety_sampler_impl::sample_type variety_sampler_impl::ff() {
    return "ff";
}
variety_sampler_impl::sample_type variety_sampler_impl::f() {
    return "f";
}
#include "variety_sampler_def.inc"

// entry point
int main() {
    variety_sampler sampler = variety_sampler();

    for (;;)
        std::cout << sampler.start();
    return 0;
}