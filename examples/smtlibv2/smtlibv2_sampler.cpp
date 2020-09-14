#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include "gramma/gramma.hpp"
#include "gramma/sample.hpp"

class smtlibv2_sampler;

using char_t = char;

using string_t = std::basic_string<char_t>;

struct denotation_t;
using sample_t = gramma::basic_sample<denotation_t, char>;

struct denotation_t {

    std::unique_ptr<sample_t> domain, range;

    enum { INTEGER, BOOLEAN, ARRAY, UNKNOWN } type;

    denotation_t() {
        type = UNKNOWN;
    }
    denotation_t(const denotation_t &den) {
        *this = den;
    }
    denotation_t &operator=(const denotation_t &den);
    denotation_t(denotation_t &&) = default;
    denotation_t(const sample_t &domain, const sample_t &range);
    denotation_t(const char *a) {
        if (a[0] == 'i') {
            type = INTEGER;
        } else if (a[0] == 'b') {
            type = BOOLEAN;
        } else {
            throw std::runtime_error(std::string("unknown sort: ") + a);
        }
    }
};

denotation_t &denotation_t::operator=(const denotation_t &den) {
    type = den.type;
    if (den.domain)
        domain = std::make_unique<sample_t>(*den.domain);
    if (den.range)
        range = std::make_unique<sample_t>(*den.range);
    return *this;
}
denotation_t::denotation_t(const sample_t &domain, const sample_t &range) {
    type = ARRAY;
    this->domain = std::make_unique<sample_t>(domain);
    this->range = std::make_unique<sample_t>(range);
}

class smtlibv2_sampler_impl : public gramma::sampler_base<smtlibv2_sampler, sample_t, std::mt19937_64> {
  public:
    using base_type = gramma::sampler_base<sampler_type, sample_type, random_engine_type>;
    using trace_type = bool;

    static constexpr double array_sexpr_rec = .001;
    static constexpr double sort_rec = .1;

    void icat(sample_t &a, const sample_t &b) {
        a += b;
    }
    sample_t denote(const sample_t &a, const denotation_t &b) {
        return sample_t(a, b);
    }

    void enter_rule(auto ruleid) {
    }

    void exit_rule() {
    }

    sample_type domain(sample_factory_type arg0);
    sample_type range(sample_factory_type arg0);
    sample_type switch_sort(sample_factory_type sort, sample_factory_type ifa, sample_factory_type bf,
                            sample_factory_type af);

    denotation_type mk_array_sort(const sample_t &domain, const sample_t &range);
};
#include "smtlibv2_sampler_decl.inc"

// implementation definition
smtlibv2_sampler_impl::sample_type smtlibv2_sampler_impl::domain(sample_factory_type af) {
    return *af().d.domain;
}
smtlibv2_sampler_impl::sample_type smtlibv2_sampler_impl::range(sample_factory_type af) {
    return *af().d.range;
}
smtlibv2_sampler_impl::sample_type smtlibv2_sampler_impl::switch_sort(sample_factory_type sortf,
                                                                      sample_factory_type ifa, sample_factory_type bf,
                                                                      sample_factory_type af) {
    switch (sortf().d.type) {
    case denotation_type::INTEGER:
        return ifa();
    case denotation_type::BOOLEAN:
        return bf();
    case denotation_type::ARRAY:
        return af();
    default:
        throw std::runtime_error("switch on incomplete sort!");
    }
}
smtlibv2_sampler_impl::denotation_type smtlibv2_sampler_impl::mk_array_sort(const sample_t &domain,
                                                                            const sample_t &range) {
    return { domain, range };
}

#include "smtlibv2_sampler_def.inc"

int main() {
    smtlibv2_sampler sampler = smtlibv2_sampler();

    for (;;)
        std::cout << sampler.start();
    return 0;
}
