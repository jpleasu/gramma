# these can actualy be specified via cml syntax
left := '<';
right := '>';
sep := ' ';
assign := '=';
close := '/';
escape := '&';

name := ['a'..'z']{1,2};
data := ['a'..'z']{3,10};

init := '\x00' . left . right . sep . assign . close . escape;
load := '\x01' . size(data);
get_attr := '\x02' . data;
set_attr := '\x03' . data;
list_attr := '\x04' . data;
get_tag := '\x05' . data;
get_ns := '\x06' . data;
query := '\x07' . data;
ver_check := '\x08' . data;

attribute := (' ' . name . '=' . '"' . data . '"'){0,2};
content := data
	 | (element . data){1,3};
element := rlim(
	(el_builder(left, name, attribute, content, right) |
	 left . name . attribute . '/' . right),
	3, "");

start := element;
#start := init . size(data{0,5});
