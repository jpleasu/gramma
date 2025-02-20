start := '(assert '
        .equals(sort)
        .')\n'
        .'(check-sat)\n'
    ;

equals(s) := '(= '.sexpr(s).' '.sexpr(s).' )';

#### sexprs ####
sexpr(s) := switch_sort(s, int_sexpr, bool_sexpr, array_sexpr(domain(s), range(s)));

bool_sexpr := 'true' | 'false'
            | array_wrap(bool_sort)
            ;
int_sexpr := ['1'..'9'].['0'..'9']{geom(4)}
            | array_wrap(int_sort)
            ;
array_sexpr(domain_sort, range_sort) :=
              "((as const (Array ".domain_sort." ".range_sort.")) ".sexpr(range_sort).")"
            | `array_sexpr_rec` "(store ".array_sexpr(domain_sort, range_sort)." ".sexpr(domain_sort)." ".sexpr(range_sort).")"
            ;

# generate an sexpr of the given sort by invoking an array with random domain and the given range
array_wrap(r) := choose d~array_domain_sort in
            "( select " . array_sexpr(d, r)." ".sexpr(d).")";

#### sorts ####
int_sort := 'Int'/'i';
bool_sort := 'Bool'/'b';
array_domain_sort := int_sort | bool_sort;
array_sort(d, r) := '( Array '.d.' '.r.' )' / mk_array_sort(d,r);

# random sort
sort := int_sort | bool_sort | `sort_rec` array_sort(array_domain_sort,sort);

