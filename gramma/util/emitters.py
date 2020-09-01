import textwrap

from typing import IO, Dict, Callable, List, Optional


class IndentationContext(object):
    def __init__(self, emitter, pre, post, flushleft, tag, level):
        self.emitter = emitter
        self.pre = pre
        self.post = post
        self.flushleft = flushleft
        self.tag = tag
        self.level = level

    def __enter__(self):
        if self.pre is not None:
            self.emitter.emit(self.pre, flushleft=self.flushleft)
        self.emitter.indent += self.level

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.emitter.indent -= self.level
        if self.post is not None:
            self.emitter.emit(self.post, flushleft=self.flushleft, tag=self.tag)


class Emitter(object):
    """
    Utility for emitting formatted code.
    """
    indent: int
    out: IO[str]
    afters: Dict[str, List[Callable[[], None]]]
    echo: Optional[IO[str]]

    def __init__(self, out, echo=None):
        self.indent = 0
        self.out = out
        self.echo = echo
        self.afters = {}

    def indentation(self, pre=None, post=None, flushleft=False, tag=None, level=1):
        return IndentationContext(self, pre, post, flushleft, tag, level)

    def emit(self, s, pre='', post='\n', trim=True, flushleft=False, after=None, tag=None):
        if after is not None:
            def closure():
                self.emit(s, pre=pre, post=post, trim=trim, flushleft=flushleft, tag=tag)

            self.afters.setdefault(after, []).append(closure)
            return

        s = textwrap.dedent(s)
        if trim:
            s = s.strip('\n')
        if not flushleft:
            s = textwrap.indent(s, '    ' * self.indent)
        self.out.write(pre + s + post)
        self.out.flush()
        if self.echo is not None:
            self.echo.write(pre + s + post)
            self.echo.flush()
        if tag is not None:
            for closure in self.afters.pop(tag, []):
                closure()
