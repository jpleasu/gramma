#include <iterator>
#include <string>
#include <iostream>
#include <sstream>

#include <functional>
#include <type_traits>

#include <deque>
#include <map>

#include <numeric>
#include <algorithm>

#include <random>

namespace gramma {
    class SamplerBase {
    public:
        using string = std::string;
        using method_type = std::function<string ()>;

        std::mt19937_64 rand;

        std::deque<std::map<string,string>> vars;

        // arrays
        template<size_t N>
        void normalize(double (&arr)[N]) {
            double sum = std::accumulate(arr, arr+N,0.0);
            for(double &x:arr)
                x/=sum;
        }

        // random
        void set_seed(auto seed) {
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

        std::uniform_real_distribution<double> u01 {0.0,1.0};

        // XXX make sure I'm using this correctly
        template<size_t N>
        int weighted_select(const double (&arr)[N]) {
            return std::discrete_distribution<int>(arr,arr+N)(rand);
        }

        template<size_t N>
        int weighted_select0(const double (&arr)[N]) {
            double p=u01(rand);
            int i=0;

            for(const double &d:arr) {
                if((p-=d)<=0)
                    return i;
                ++i;
            }
            return N-1;
        }

        template<typename T, size_t N>
        T uniform_selection(const T (&arr)[N]) {
            return arr[std::uniform_int_distribution<int>(0,N-1)(rand)];
        }




        // variable stack
        void push_vars() {
            vars.push_front({});
        }
        void pop_vars() {
            vars.pop_front();
        }
        void set_var(const string &name, const string &value) {
            vars.back()[name]=value;
        }
        string get_var(const string &name) {
            for(auto &m:vars) {
                auto it = m.find(name);
                if ( it != m.end() )
                    return it->second;
            }
            return ""; // raise bad grammar exception?
        }
    };
}
