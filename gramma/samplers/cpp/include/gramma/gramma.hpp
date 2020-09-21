#ifndef GRAMMA_HPP
#define GRAMMA_HPP
#pragma once

#include <iostream>
#include <iterator>
#include <sstream>
#include <string>

#include <stdexcept>

#include <functional>
#include <type_traits>

#include <deque>
#include <map>

#include <algorithm>
#include <numeric>

#include <random>

namespace gramma {
    template <typename EnumT>
    constexpr auto to_underlying(EnumT e) noexcept {
        return static_cast<std::underlying_type_t<EnumT>>(e);
    }

    template <typename RandomEngineT = std::mt19937_64>
    class RandomAPI {
      public:
        using random_engine_type = RandomEngineT;
        random_engine_type rand;

        RandomAPI() : rand{} {
        }

        template <size_t N>
        void normalize(double (&weights)[N]) {
            double sum = std::accumulate(weights, weights + N, 0.0);
            for (double &x : weights)
                x /= sum;
        }

        void set_seed(typename random_engine_type::result_type seed) {
            rand.seed(seed);
        }

        std::string get_state() {
            std::ostringstream oss;
            oss << rand;
            return oss.str();
        }
        void set_state(const std::string &state) {
            std::istringstream iss(state);
            iss >> rand;
        }

        template <typename T, size_t N>
        T choice(const T (&choices)[N]) {
            return choices[std::uniform_int_distribution<int>(0, N - 1)(rand)];
        }
        template <typename T, size_t N>
        T choice(const T (&choices)[N], const double (&weights)[N]) {
            return choices[std::discrete_distribution<int>(weights, weights + N)(rand)];
        }

        template <size_t N>
        int weighted_select(const double (&weights)[N]) {
            return std::discrete_distribution<int>(weights, weights + N)(rand);
        }

        int weighted_select(const double *weights, size_t n) {
            return std::discrete_distribution<int>(weights, weights + n)(rand);
        }

        std::uniform_real_distribution<double> u01{ 0.0, 1.0 };
        double uniform() {
            return u01(rand);
        }

        // inclusive, [lo,hi]
        int uniform(int lo, int hi) {
            return std::uniform_int_distribution<int>(lo, hi)(rand);
        }

        double uniform(double lo, double hi) {
            return std::uniform_real_distribution<double>(lo, hi)(rand);
        }

        int geometric(double p) {
            return std::geometric_distribution<int>(p)(rand);
        }

        double normal(double mean, double stddev) {
            return std::normal_distribution<double>(mean, stddev)(rand);
        }

        int binomial(int n, double p) {
            return std::binomial_distribution<int>(n, p)(rand);
        }
    };

    template <typename SamplerT, typename SampleT, typename RandomEngineT = std::mt19937_64>
    class sampler_base {
      public:
        using sample_factory_type = std::function<SampleT()>;
        using sampler_type = SamplerT;
        using sample_type = SampleT;
        using random_engine_type = RandomEngineT;
        using denotation_type = typename sample_type::denotation_type;

        std::deque<std::map<int, sample_type>> vars;

        // variable stack
        void push_vars() {
            vars.push_front({});
        }
        void pop_vars() {
            vars.pop_front();
        }
        struct var_ctx_t {
            sampler_base &sampler;

            var_ctx_t(sampler_base &sampler) : sampler(sampler) {
                sampler.push_vars();
            }
            ~var_ctx_t() {
                sampler.pop_vars();
            }
        };

        var_ctx_t var_ctx() {
            return { *this };
        }

        template <typename RuleIdT>
        struct rule_ctx_t {
            sampler_base &sampler;

            rule_ctx_t(sampler_base &sampler, RuleIdT ruleid) : sampler(sampler) {
                sampler._impl().enter_rule(ruleid);
            }
            ~rule_ctx_t() {
                sampler._impl().exit_rule();
            }
        };

        template <typename RuleIdT>
        rule_ctx_t<RuleIdT> rule_ctx(RuleIdT ruleid) {
            return { *this, ruleid };
        }

        template <typename T>
        using is_sample_t =
            std::enable_if_t<std::is_base_of<sample_type, std::remove_cv_t<std::remove_reference_t<T>>>::value, int>;

        template <typename T>
        using is_convertible_to_sample_t = std::enable_if_t<
            !std::is_base_of<sample_type, std::remove_cv_t<std::remove_reference_t<T>>>::value &&
                std::is_constructible<sample_type, std::remove_cv_t<std::remove_reference_t<T>>>::value,
            int>;

        template <typename EnumT, typename T, is_sample_t<T> = 0>
        void set_var(EnumT varid, T &&value) {
            vars.front()[to_underlying(varid)] = std::forward<T>(value);
        }

        template <typename EnumT, typename T, is_convertible_to_sample_t<T> = 0>
        void set_var(EnumT varid, T &&value) {
            vars.front()[to_underlying(varid)] = sample_type(std::forward<T>(value));
        }

        template <typename EnumT>
        sample_type get_var(EnumT varid) {
            for (auto &m : vars) {
                auto it = m.find(to_underlying(varid));
                if (it != m.end())
                    return it->second;
            }
            throw std::runtime_error("no var defined with id " + std::to_string(to_underlying(varid)));
        }

        // access to the generated class in the implementation
        SamplerT &_impl() {
            return *static_cast<SamplerT *>(this);
        }

        RandomAPI<RandomEngineT> random;
    };
} // namespace gramma
#endif // GRAMMA_HPP
