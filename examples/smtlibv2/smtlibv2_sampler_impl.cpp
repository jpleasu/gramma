/* === declare state variables === */

// for g4toglf generated grammars
int maxrep=3;

// rule depth in trace tree
int rule_depth;

// set state prior to each sample
void reset_state() {
    rule_depth=0;
}
#if defined(USE_SIDEEFFECT_API)
// XXX choose the type associated with each rule 
using sideeffect_t=bool;

// XXX return the value associated with a rule just prior to its execution
sideeffect_t push(rule_t rule) {
    ++rule_depth;
    return true;
}

// XXX handle the result immediately after rule execution
void pop(const sideeffect_t &assoc, const string_t &subsample) {
    if(assoc) {
        --rule_depth;
    }
}
#endif
// XXX define gfuncs

// run method and return empty string.. mnemonic "execute"
template<typename T>
string_t e(T m) {
    m();
    return "";
}
string_t domain(method_t arg0){
    return "?";
}
string_t mk_array_sort(method_t arg0,method_t arg1){
    return "?";
}
string_t range(method_t arg0){
    return "?";
}
string_t switch_sort(method_t arg0,method_t arg1,method_t arg2,method_t arg3){
    return "?";
}
