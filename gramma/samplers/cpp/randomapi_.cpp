#include "gramma/gramma.hpp"
namespace {
    using namespace gramma;
    RandomAPI *new_random() {
        return new RandomAPI{};
    }
    void delete_random(RandomAPI *r) {
        if (r != nullptr) {
            delete r;
        }
    }
    void seed(RandomAPI *r, uint64_t seed) {
        r->set_seed(seed);
    }
    int integers(RandomAPI *r, int lo, int hi) {
        return r->uniform(lo, hi - 1);
    }
    int geometric(RandomAPI *r, double p) {
        return r->geometric(p);
    }
    double normal(RandomAPI *r, double mean, double stddev) {
        return r->normal(mean, stddev);
    }
    int binomial(RandomAPI *r, int n, double p) {
        return r->binomial(n, p);
    }
    int weighted_select(RandomAPI *r, const double *weights, size_t n) {
        return r->weighted_select(weights, n);
    }

    struct API {
        RandomAPI *(*new_random)();
        void (*delete_random)(RandomAPI *);
        void (*seed)(RandomAPI *r, uint64_t seed);
        int (*integers)(RandomAPI *r, int lo, int hi);
        int (*geometric)(RandomAPI *r, double p);
        double (*normal)(RandomAPI *r, double mean, double stddev);
        int (*binomial)(RandomAPI *r, int n, double p);
        int (*weighted_select)(RandomAPI *r, const double *weights, size_t n);
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
