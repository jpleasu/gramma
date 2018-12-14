int := ['1'..'9'].['0'..'9']{0,5};

id :=
   "a"
   | "b"
   | "c"
   | "d"
;
   
op :=
   "set"
   | "get"
   | "del"
;

na := "num" | "arr";

size :=
     "int"
     | "uint"
     | "short"
     | "ushort"
     | "byte"
     | "ubyte"
;
     
new_na :=
	"new ". na . " " . id . " " . int;

new_view :=
	"new view " . id . " " . id . " " . size;
	 
get_num	:= 
	"get " . id;
	
get_av	:= 
	"get " . id . " " . int;

set_arr :=
	"set " . id . " " . int . " " . int;

set_num	:= 
	"set " . id . " " . int;
	
set_view :=
	"set " . id . " byteSize " . id;

del :=
    "del " . id;

gv_l :=
   new_na
   | new_view
   | get_num
   | get_av
   | del
   | set_arr
   | set_num
   | set_view
;


start := (gv_l . ";\n"){3,50};

