#ifndef GRAMMA_HPP
#define GRAMMA_HPP
#pragma once

#include <iostream>
#include <iterator>
#include <sstream>
#include <string>

#include <functional>
#include <type_traits>

#include <deque>
#include <map>

#include <algorithm>
#include <numeric>

#include <random>

namespace gramma {
    class RandomAPI {
      public:
        using rand_t = std::mt19937_64;
        rand_t rand;

        template <size_t N>
        void normalize(double (&weights)[N]) {
            double sum = std::accumulate(weights, weights + N, 0.0);
            for (double &x : weights)
                x /= sum;
        }

        void set_seed(rand_t::result_type seed) {
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

    template <class SamplerT, class SampleT>
    class SamplerBase {
      public:
        using func_type = std::function<SampleT()>;
        using sample_type = SampleT;
        using denotation_type = typename sample_type::denotation_type;

        std::deque<std::map<int, sample_type>> vars;

        // variable stack
        void push_vars() {
            vars.push_front({});
        }
        void pop_vars() {
            vars.pop_front();
        }

        template <class T>
        void set_var(int varid, T value) {
            static_assert(std::is_base_of<sample_type, T>::value, "set_var values must be samples");
            vars.back()[varid] = std::forward<T>(value);
        }

        sample_type get_var(int varid) {
            for (auto &m : vars) {
                auto it = m.find(varid);
                if (it != m.end())
                    return it->second;
            }
            return {}; // raise bad grammar exception?
        }

        // access to the generated class in the implementation
        SamplerT &_generated() {
            return *static_cast<SamplerT *>(this);
        }

        RandomAPI random;
    };
} // namespace gramma
#endif // GRAMMA_HPP
