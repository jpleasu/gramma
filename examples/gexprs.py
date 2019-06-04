#!/usr/bin/env python
'''

    demo some of the gexpr api

'''
from __future__ import absolute_import, division, print_function

import random

from gramma import *

g=GrammaGrammar('start:="";')

# parsing
print(g.parse(r'''
    ['a'..'z']
'''))
print(g.parse(r'''
    '\''."insinglequotes".'\''
'''))
print(g.parse('''
    "a"| `15` "b"
'''))
print(g.parse('''
    "a"{1,2}
'''))
print(g.parse('''
    "a"{,2}
'''))
print(g.parse('''
    "a"{2}
'''))






# vim: ts=4 sw=4
