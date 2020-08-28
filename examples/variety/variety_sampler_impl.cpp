/* === declare state variables === */

// for g42glf generated grammars
int maxrep = 3;

// rule depth in trace tree
int rule_depth;

int a = 2;

// set state prior to each sample
void reset_state() {
    rule_depth = 0;
}
#if defined(USE_SIDEEFFECT_API)
// XXX choose the type associated with each rule
using sideeffect_t = bool;

// XXX return the value associated with a rule just prior to its execution
sideeffect_t push(rule_t rule) {
    ++rule_depth;
    return true;
}

// XXX handle the result immediately after rule execution
void pop(const sideeffect_t &assoc, const string_t &subsample) {
    if (assoc) {
        --rule_depth;
    }
}
#endif
// XXX define gfuncs

// run method and return empty string.. mnemonic "execute"
template <typename T> string_t e(T m) {
    m();
    return "";
}
string_t f() {
    return "f";
}
string_t ff() {
    return "ff";
}

auto g_func() {
    return 7;
}
