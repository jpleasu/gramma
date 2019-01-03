int := ['1'..'9'].['0'..'9']{0,9};
#int := "0" | "1" | "1207963648" | "4294967295";

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
	 
get_num	:= "get " . old('num') . "\n";
	
get_arr	:= "get " . old('arr') . " " . int . "\n";

set_arr := "set " . old('arr') . " " . int . " " . int . "\n";

set_num	:= "set " . old('num') . " " . int . "\n";
	
set_view := "set " . old('view') . " byteSize " . old('num') . "\n";

del := "del " . (old('num') | old('arr') | old('view')) . "\n";

nesting :=  rlim(
      new_num . nesting{0,2} . get_num
    | new_arr . nesting{0,2} . new_view{0,1} . nesting{0,2} . get_arr. nesting{0,1}
    | ifdef('arr', "del ". old('arr')."\n")
    | ifdef('view', ifdef('num', set_view))
  , 5, "")
;

start := nesting;




# instance bound ids 
testing := "def ". push(id) . (";" . testing{0,1}) . (" use " . peek()){1,3} . pop();
#start := (testing . "\n"){1,3};
