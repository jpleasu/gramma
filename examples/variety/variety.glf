start := recurs."\n";

yyy:='a'{,`g_func()`};

xxx:='a'{1,`g_func()`};

recurs := 10 ".".recurs | vars . " ". words . " " . ['1'..'9'] . digit{1,15,geom(5)};

digit := ['0' .. '9'];

weirddigits := ['0','2'..'5','7','9'];

words := (`1000*(rule_depth>20)` "*" | " ").
         ( .75 "dog" | .25 "cat" ).
         (" f=".f()." a=".(`a`?"1":"0")." ff=".ff()){1,4};

#vars := choose x~('a'|'b'), y~('c'|'d') in x.y.x.y.x.y;
vars := x ~ 'a'|'b', y~ 'c'|'d' in x.y.x.y.x.y;
