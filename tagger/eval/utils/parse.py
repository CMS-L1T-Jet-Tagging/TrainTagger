"""Variable expressions: the Variable and Call types, and the parser.

A variable is a named, documented function that computes one result from
a fixed set of inputs. It is called by an expression, e.g.
``auc(b, light)`` or ``effAtMistag(taus, jets, 0.1)``; arguments are
class names, class groups, numbers or (for metavariables) other
expressions. The VariableManager (tagger.eval.variable_manager.vm)
parses expressions with :func:`_parse`.
"""

import re
import typing
from dataclasses import dataclass, field

KINDS = ("scalar", "curve", "inherit")


class Expr(str):
    """Annotation for an argument that is itself a variable expression
    (metavariables)."""


def choices(ptype):
    """The allowed names of a Literal["a", "b"] argument annotation,
    else None."""
    if typing.get_origin(ptype) is typing.Literal and all(
        isinstance(a, str) for a in typing.get_args(ptype)
    ):
        return typing.get_args(ptype)
    return None


@dataclass(frozen=True)
class Variable:
    name: str
    func: object = field(repr=False)
    inputs: frozenset
    kind: str
    doc: str
    module: str
    params: tuple  # ((name, type), ...)

    def signature(self):
        if not self.params:
            return self.name
        return f"{self.name}({', '.join(p for p, _ in self.params)})"


# --------------------------------------------------------------------
# Parsing

_TOKEN = re.compile(
    r"\s*(?:"
    r"([A-Za-z_][A-Za-z0-9_.:+-]*)"  # a name
    r"|([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)"  # a number
    r"|(.))"  # any other character
)


@dataclass(frozen=True)
class Call:
    """A parsed and type-checked expression."""

    variable: Variable
    args: tuple

    @property
    def name(self):
        """Canonical spelling, used as the result key and cache key."""
        if not self.variable.params:
            return self.variable.name
        return f"{self.variable.name}({', '.join(_fmt(a) for a in self.args)})"

    def __str__(self):
        return self.name


def _fmt(value):
    if isinstance(value, Call):
        return value.name
    if isinstance(value, float):
        return repr(value)
    return str(value)


def _tokenize(text):
    pos, tokens = 0, []
    while pos < len(text):
        m = _TOKEN.match(text, pos)
        if m is None or m.end() == pos:
            break
        ident, number, other = m.groups()
        pos = m.end()
        if ident is not None:
            tokens.append(("ident", ident))
        elif number is not None:
            tokens.append(("number", number))
        elif other is not None and other.strip():
            tokens.append(("op", other))
    return tokens


def _parse_tokens(tokens, i, text):
    """Parse one term starting at tokens[i]. Returns (raw, i) where raw
    is str/number-string or (name, [args])."""
    if i >= len(tokens):
        raise SyntaxError(f"unexpected end of expression: {text!r}")
    kind, value = tokens[i]
    if kind == "number":
        return ("number", value), i + 1
    if kind != "ident":
        raise SyntaxError(f"unexpected {value!r} in {text!r}")
    i += 1
    if i < len(tokens) and tokens[i] == ("op", "("):
        args, i = [], i + 1
        if i < len(tokens) and tokens[i] == ("op", ")"):
            return ("call", value, args), i + 1
        while True:
            arg, i = _parse_tokens(tokens, i, text)
            args.append(arg)
            if i < len(tokens) and tokens[i] == ("op", ","):
                i += 1
                continue
            if i < len(tokens) and tokens[i] == ("op", ")"):
                return ("call", value, args), i + 1
            raise SyntaxError(f"expected ',' or ')' in {text!r}")
    return ("call", value, None), i


def _raw_to_text(raw):
    if raw[0] == "number":
        return raw[1]
    _, name, args = raw
    if args is None:
        return name
    return f"{name}({', '.join(_raw_to_text(a) for a in args)})"


def _bind(raw, text, lookup):
    if raw[0] != "call":
        raise SyntaxError(
            f"{text!r}: a variable name is expected, got {raw[1]!r}"
        )
    _, name, args = raw
    var = lookup(name)
    args = args or []
    if len(args) != len(var.params):
        raise TypeError(
            f"{var.signature()} takes {len(var.params)} argument(s), got"
            f" {len(args)} in {text!r}"
        )
    bound = []
    for (pname, ptype), arg in zip(var.params, args):
        if ptype is Expr:
            bound.append(_bind(arg, text, lookup))
        elif ptype in (float, int):
            if arg[0] != "number":
                raise TypeError(
                    f"{var.name}: argument {pname} must be a number, got"
                    f" {_raw_to_text(arg)!r}"
                )
            bound.append(
                ptype(float(arg[1])) if ptype is int else float(arg[1])
            )
        else:
            if arg[0] == "call" and arg[2] is not None:
                raise TypeError(
                    f"{var.name}: argument {pname} must be a name, got"
                    f" {_raw_to_text(arg)!r}"
                )
            value = arg[1]
            allowed = choices(ptype)
            if allowed and value not in allowed:
                raise ValueError(
                    f"{var.name}: argument {pname} must be one of"
                    f" {', '.join(allowed)}, got {value!r}"
                )
            bound.append(value)
    return Call(var, tuple(bound))


def _parse(text, lookup):
    """Parse an expression like ``effAtMistag(b, light, 0.1)`` into a
    ``Call``. ``lookup(name)`` returns the Variable of a name
    (VariableManager.get_variable)."""
    tokens = _tokenize(text)
    raw, i = _parse_tokens(tokens, 0, text)
    if i != len(tokens):
        raise SyntaxError(f"trailing input in {text!r}")
    return _bind(raw, text, lookup)
