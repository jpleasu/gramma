#ifndef GRAMMA_SAMPLE_HPP
#define GRAMMA_SAMPLE_HPP
#pragma once

#include <string>

namespace gramma {
    template <class DenotationT, class CharT = char>
    struct basic_sample : public std::basic_string<CharT> {
        using char_type = CharT;
        using denotation_type = DenotationT;
        using string_type = std::basic_string<char_type>;

        denotation_type d;

        // constructors
        basic_sample() = default;
        basic_sample(basic_sample &&) = default;
        basic_sample(const basic_sample &) = default;
        basic_sample &operator=(const basic_sample &) = default;

        basic_sample(string_type &&s, denotation_type d = {}) : string_type(std::move(s)), d(d) {
        }

        basic_sample(const string_type &s, denotation_type d = {}) : string_type(s), d(d) {
        }

        basic_sample(const CharT *s, denotation_type d = {}) : string_type(s), d(d) {
        }

        basic_sample(int count, CharT c, denotation_type d = {}) : string_type(count, c), d(d) {
        }

        // allows non-lazy (basic_sample) gfunc arguments by converting from lazy (func_type)
        template <class T>
        basic_sample(T m) : basic_sample(m()) {
            static_assert(std::is_base_of<basic_sample<denotation_type, CharT>, decltype(m())>::value,
                          "callable-converting constructor argument must return sample");
        }
    };
} // namespace gramma
#endif // GRAMMA_SAMPLE_HPP
