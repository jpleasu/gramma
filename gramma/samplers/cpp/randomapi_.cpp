#include "gramma/gramma.hpp"
namespace {
    using namespace gramma;
    using random_engine_type = RandomAPI<>;
    random_engine_type *new_random() {
        return new random_engine_type{};
    }
    void delete_random(random_engine_type *r) {
        if (r != nullptr) {
            delete r;
        }
    }
    void seed(random_engine_type *r, uint64_t seed) {
        r->set_seed(seed);
    }
    int integers(random_engine_type *r, int lo, int hi) {
        return r->uniform(lo, hi - 1);
    }
    int geometric(random_engine_type *r, double p) {
        return r->geometric(p);
    }
    double normal(random_engine_type *r, double mean, double stddev) {
        return r->normal(mean, stddev);
    }
    int binomial(random_engine_type *r, int n, double p) {
        return r->binomial(n, p);
    }
    int weighted_select(random_engine_type *r, const double *weights, size_t n) {
        return r->weighted_select(weights, n);
    }

    struct API {
        random_engine_type *(*new_random)();
        void (*delete_random)(random_engine_type *);
        void (*seed)(random_engine_type *r, uint64_t seed);
        int (*integers)(random_engine_type *r, int lo, int hi);
        int (*geometric)(random_engine_type *r, double p);
        double (*normal)(random_engine_type *r, double mean, double stddev);
        int (*binomial)(random_engine_type *r, int n, double p);
        int (*weighted_select)(random_engine_type *r, const double *weights, size_t n);
    } api = { .new_random = new_random,
              .delete_random = delete_random,
              .seed = seed,
              .integers = integers,
              .geometric = geometric,
              .normal = normal,
              .binomial = binomial,
              .weighted_select = weighted_select };
} // namespace
extern "C" {

API *get_api() {
    return &api;
}
}
