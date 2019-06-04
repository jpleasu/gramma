start := 
    def . (def{0,1.} . use{1,3.}) {1,3.}
    . ineq{1,3.}
;

def :=    "newthing " . new('thing',id) . ";\n" ;
use :=    "oldthing " . old('thing') . ";\n" ;

ineq := rand_val() . " < " . bigger_val() . ";\n" ;

id := ['a'..'z'] . ['a'..'z']{5.};

