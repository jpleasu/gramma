int := ['1'..'9'].['0'..'9']{0,5};
#int := "0" | "1" | "1207963648" | "4294967295";
#int := "1" | "1207963648" | "4294967295";

id := ['a'..'z']{3,10};
   

size :=
     "int"
     | "uint"
     | "short"
     | "ushort"
     | "byte"
     | "ubyte"
;
     
new_num := "new num " . new('num',id) . " " . int . "\n";

new_arr := "new arr " . new('arr',id) . " " . int . "\n";

new_view := "new view " . new('view',id) . " " . old('arr') . " " . size . "\n";
 
get_num := "get " . old('num') . "\n";

get_arr := "get " . old('arr') . " " . int . "\n";

get_view := "get " . old('view') . " " . int . "\n";

set_arr := "set " . old('arr') . " " . int . " " . int . "\n";

set_num := "set " . old('num') . " " . int . "\n";

set_view := "set " . old('view') . " byteSize " . old('num') . "\n";

del := "del " . (old('num') | old('arr') | old('view')) . "\n";


oneline :=
  nonempty(
      new_num
    | new_arr
    | ifdef('arr', new_view)
    | ifdef('num', set_num)
    | ifdef('view', ifdef('num', set_view|get_view))
  )
;

segfault :=
"""\
new num a 0
new arr b 1
new view c b byte
set c byteSize a
set a 4294967295
get c 1207963648
""";


segfault1 :=
      new_num
    . new_arr
    . new_view
    . set_view
    . set_num
    . get_view
;

#start := segfault;
#start := segfault1;
start := oneline{20,100};


# instance bound ids 
testing := "def ". push(id) . (";" . testing{0,1}) . (" use " . peek()){1,3} . pop();
#start := (testing . "\n"){1,3};
