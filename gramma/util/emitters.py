import textwrap

from typing import IO, Dict, Callable, List, Optional, Union


class EmitterError(BaseException):
    pass


class IndentationContext:
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


class WriteContext:
    def __init__(self, emitter):
        self.emitter = emitter

    def __enter__(self):
        return self.emitter

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.emitter.close()


class Emitter:
    """
    Utility for emitting formatted code.
    """
    indent: int
    out: Optional[IO[str]]
    afters: Dict[str, List[Callable[[], None]]]
    echo: Optional[IO[str]]

    def __init__(self, out: Union[None, str, IO[str]] = None, echo: Optional[IO[str]] = None):
        self.indent = 0
        self.out = None
        if out is not None:
            self.write_to(out)
        self.echo = echo
        self.afters = {}

    def indentation(self, pre=None, post=None, flushleft=False, tag=None, level=1):
        return IndentationContext(self, pre, post, flushleft, tag, level)

    def close(self) -> None:
        if self.out is not None:
            self.out.close()
            self.out = None

    def write_to(self, out: Union[str, IO[str]], mode: str = 'w') -> WriteContext:
        self.close()
        if isinstance(out, str):
            self.out = open(out, mode)
        else:
            self.out = out
        return WriteContext(self)

    def emit(self, s, pre='', post='\n', trim=True, flushleft=False, after=None, tag=None):
        if self.out is None:
            raise EmitterError('no associated file')

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
