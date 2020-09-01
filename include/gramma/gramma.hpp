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
        void normalize(double (&arr)[N]) {
            double sum = std::accumulate(arr, arr + N, 0.0);
            for (double &x : arr)
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

        std::uniform_real_distribution<double> u01{ 0.0, 1.0 };

        template <size_t N>
        int weighted_select(const double (&arr)[N]) {
            return std::discrete_distribution<int>(arr, arr + N)(rand);
        }

        template <typename T, size_t N>
        T uniform_selection(const T (&arr)[N]) {
            return arr[std::uniform_int_distribution<int>(0, N - 1)(rand)];
        }
    };

    template <class ImplT, class SampleT, class DenotationT>
    class SamplerBase {
      public:
        using method_t = std::function<SampleT()>;

        std::deque<std::map<int, SampleT>> vars;

        // variable stack
        void push_vars() {
            vars.push_front({});
        }
        void pop_vars() {
            vars.pop_front();
        }
        void set_var(int name_id, const SampleT &value) {
            vars.back()[name_id] = value;
        }
        SampleT get_var(int name_id) {
            for (auto &m : vars) {
                auto it = m.find(name_id);
                if (it != m.end())
                    return it->second;
            }
            return {}; // raise bad grammar exception?
        }

        ImplT &impl() {
            return *static_cast<ImplT *>(this);
        }

        SampleT cat(const SampleT &a, const SampleT &b) {
          return impl().cat(a,b);
        }

        SampleT denote(const SampleT &a, const DenotationT &b) {
          return impl().denote(a,b);
        }

        RandomAPI random;
    };
} // namespace gramma
