#!/usr/bin/env python

from __future__ import absolute_import, division, print_function
import sys
if sys.version_info < (3,0):
    from builtins import (bytes, str, open, super, range,zip, round, input, int, pow)

    import __builtin__

    def func_name(f):
        return f.func_name

    def ast_argname(a):
        return a.id

else:
    import builtins as __builtin__

    def func_name(f):
        return f.__name__

    xrange=range

    def ast_argname(a):
        return a.arg

from six import string_types, with_metaclass

import inspect,ast

import textwrap

import numbers


try:
    import astpretty
except ImportError:
    pass



class NamePath(object):
    __slots__='path','s'
    def __init__(self,n):
        if isinstance(n, ast.AST):
            p=[n]
            while isinstance(p[0],(ast.Attribute,ast.Subscript)):
                p.insert(0,p[0].value)
            self.path=p
        else:
            self.path=n
        self.s=self._compute_s()

    @staticmethod
    def _tostr(x):
        if isinstance(x,ast.Name):
            return x.id
        elif isinstance(x,ast.Attribute):
            return x.attr
        elif isinstance(x,ast.Subscript):
            return '[]'
    def _compute_s(self):
        s=''
        for x in self.path:
            if not isinstance(x,ast.Subscript) or s=='':
                s+='.'
            s+=self._tostr(x)
        return s[1:]

    def __getitem__(self,slc):
        if isinstance(slc,int):
            return NamePath([self.path[slc]])
        elif isinstance(slc,slice):
            return NamePath(self.path[slc])
        return None

    def __len__(self):
        return len(self.path)

    def __hash__(self):
        return hash(self.s)
    
    def __eq__(self,other):
        return self.s==other.s

    def __repr__(self):
        return self.s

def detup(x):
    return x.elts if isinstance(x,ast.Tuple) else [x]

class VariableAccesses(ast.NodeVisitor):
    '''
        an ast node visitor that extracts variables and their acccesses from a
        function (method).
    '''

    def run(self,funcdef):
        self.stack=[]

        al=f.args.args
        self.args=[ast_argname(a) for a in al]

        self.f=funcdef

        for item in self.f.body:
            self.visit(item)

    def visit(self,n):
        self.stack.append(n)
        super().visit(n)
        self.stack.pop()

    def visit_FunctionDef(self,funcdef):
        self.defs(funcdef.name, funcdef)

    def visit_Assign(self,ass):
        if len(ass.targets)!=1:
            astpretty.pprint(ass)
            raise ValueError('unexpected number of targets in assign')
        tn=[NamePath(x) for x in detup(ass.targets[0])]
        tv=detup(ass.value)
        for n,v in zip(tn,tv):
            self.defs(n,v)
        self.visit(ass.value)

    def visit_AugAssign(self,aug):
        self.mods(NamePath(aug.target), aug)
        self.visit(aug.value)

    def visit_Name(self,name):
        i=next(i for i in reversed(range(len(self.stack)-1)) if not isinstance(self.stack[i], (ast.Attribute, ast.Subscript)))
        n=NamePath(self.stack[i+1])
        p=self.stack[i]
        self.uses(n)

    def visit_Lambda(self,lam):
        self.lambdas(lam)
        pass#skip

    def visit_ClassDef(self,classdef):
        pass#skip

    def visit_Call(self,call):
        if isinstance(call.func, (ast.Attribute, ast.Name)):
            self.calls(NamePath(call.func), call)
        else:
            self.visit(call.func)
        for a in call.args:
            self.visit(a)
        for a in call.keywords:
            self.visit(a)

def get_methods(Class):
    s=inspect.getsource(Class)
    s=textwrap.dedent(s)

    classdef=ast.parse(s).body[0]
    return [f for f in classdef.body if isinstance(f,ast.FunctionDef)]


class TestClass(object):
    def __init__(self):
        self.x=5

    def f1(self):
        print(self.x)

    def subdefs(self,a,b):
        def subfunc(x,y):
            return x+y
        def subgen(x,y):
            yield x+y
        l=lambda x,y:x+y
        return subfunc

    def assigns(self,x,y):
        self.a.b.c=x
        self.a,self.b=x,y

    def augassign(self,x):
        self.a+=3

    def calls(self,f):
        self.method()
        self.member.method()
        (lambda:sneak())()
        x=f(12)

    def subscripts(self):
        self.a[15]=self.b[13]
        self.c[15].member.method()
        self.d[18].e[27]+=12

if __name__=='__main__':
    class V(VariableAccesses):
        def __init__(self, f):
            VariableAccesses.__init__(self)
            self._defs={}
            self._uses={}
            self._mods={}
            self._calls={}
            self._lambdas=[]
            self.run(f)
    
        def defs(self, n, v):
            self._defs.setdefault(n,[]).append(v)
        def uses(self, n):
            self._uses.setdefault(n,[]).append(None)
        def mods(self, n, v):
            self._mods.setdefault(n,[]).append(v)
        def calls(self, n, v):
            self._calls.setdefault(n,[]).append(v)
        def lambdas(self, l):
            self._lambdas.append(l)

    for f in get_methods(TestClass):

        print('=======')
        va=V(f)
        print('------')
        print('args: %s' % va.args)
        print('defs: %s' % va._defs.keys())
        print('uses: %s' % va._uses.keys())
        print('mods: %s' % va._mods.keys())
        print('calls: %s' % va._calls.keys())
        print('lambdas: %d' % len(va._lambdas))

# vim: ts=4 sw=4
