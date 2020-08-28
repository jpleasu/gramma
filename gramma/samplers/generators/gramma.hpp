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
    class SamplerBase {
      public:
        using string_t = std::string;
        using method_t = std::function<string_t()>;

        using rand_t = std::mt19937_64;
        rand_t rand;

        std::deque<std::map<string_t, string_t>> vars;

        // arrays
        template <size_t N> void normalize(double (&arr)[N]) {
            double sum = std::accumulate(arr, arr + N, 0.0);
            for (double &x : arr)
                x /= sum;
        }

        // random
        void set_seed(rand_t::result_type seed) {
            rand.seed(seed);
        }
        void set_state(const std::string &state) {
            std::istringstream iss(state);
            iss >> rand;
        }
        std::string get_state() {
            std::ostringstream oss;
            oss << rand;
            return oss.str();
        }

        std::uniform_real_distribution<double> u01{ 0.0, 1.0 };

        template <size_t N> int weighted_select(const double (&arr)[N]) {
            return std::discrete_distribution<int>(arr, arr + N)(rand);
        }

        template <typename T, size_t N> T uniform_selection(const T (&arr)[N]) {
            return arr[std::uniform_int_distribution<int>(0, N - 1)(rand)];
        }

        // variable stack
        void push_vars() {
            vars.push_front({});
        }
        void pop_vars() {
            vars.pop_front();
        }
        void set_var(const string_t &name, const string_t &value) {
            vars.back()[name] = value;
        }
        string_t get_var(const string_t &name) {
            for (auto &m : vars) {
                auto it = m.find(name);
                if (it != m.end())
                    return it->second;
            }
            return {}; // raise bad grammar exception?
        }
    };
} // namespace gramma
