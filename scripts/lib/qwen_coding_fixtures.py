"""Deterministic, stdlib-only direct-model coding fixtures.

Public API: load_cases(pack_path) -> list[dict] and
validate_response(case, text, finish_reason) -> dict. Hidden inputs and trusted
oracles below are never added to model messages. self_check() exercises trusted
solutions, known wrong answers, and the evaluator boundary.

Generated Python is NEVER compiled, eval'ed, or exec'ed. A small AST interpreter
supports the pure-function subset documented in the prompt. Attribute operations
are available only through explicit type-checked list/dict methods. Imports,
introspection, arbitrary calls, recursion, and host APIs are unavailable. Evaluation
also runs in a fresh -I Python subprocess with an empty temporary working directory,
a minimal environment, fuel limits, a wall timeout, and OS resource limits. This is
not an OS sandbox and makes no claim to safely execute arbitrary Python.

The tests establish these bounded fixtures, not general coding capability. Exact
output types, lack of input mutation, and completion without truncation matter.
For lower_bound, interpreted loop iterations also have a logarithmic bound; this
is an algorithm sanity check, not a proof of asymptotic complexity.
"""

import ast
from collections import ChainMap
import copy
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time


VALIDATOR_VERSION = "qwen-coding-ast-v2"
MAX_SOURCE = 32768
MAX_ITEMS = 8192
MAX_FUEL = 100000
CASE_SIGNATURES = {"merge_intervals": 1, "lower_bound": 2, "select_jobs": 2}
BUILTINS = {
    "abs", "all", "any", "bool", "dict", "enumerate", "int", "len", "list",
    "max", "min", "range", "reversed", "set", "sorted", "str", "sum", "tuple", "zip",
}
METHODS = {"append", "extend", "copy", "sort", "pop", "get", "items", "keys", "values"}
DICT_VIEWS = (type({}.items()), type({}.keys()), type({}.values()))
ALLOWED_NODES = {
    ast.Module, ast.FunctionDef, ast.arguments, ast.arg, ast.Return, ast.Assign,
    ast.AugAssign, ast.If, ast.For, ast.While, ast.Break, ast.Continue, ast.Pass,
    ast.Expr, ast.Name, ast.Constant, ast.List, ast.Tuple, ast.Set, ast.Dict,
    ast.Subscript, ast.Slice, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare,
    ast.IfExp, ast.Call, ast.keyword, ast.Lambda, ast.Attribute,
    ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp, ast.comprehension,
    ast.Load, ast.Store, ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Div, ast.Mod,
    ast.USub, ast.UAdd, ast.Not, ast.And, ast.Or, ast.Eq, ast.NotEq, ast.Lt,
    ast.LtE, ast.Gt, ast.GtE, ast.In, ast.NotIn, ast.Is, ast.IsNot,
}


def load_cases(pack_path):
    """Load prompts; the long case includes its existing module in one fence."""
    pack = json.loads(Path(pack_path).read_text(encoding="utf-8"))
    if pack.get("schema_version") != 1 or pack.get("id") != "qwen4b-coding-v1":
        raise ValueError("unsupported coding fixture pack")
    cases = []
    for raw in pack["cases"]:
        case = dict(raw)
        if CASE_SIGNATURES.get(case["id"]) != case.get("arg_count"):
            raise ValueError("unknown coding fixture signature")
        if case.get("function_name") != case["id"]:
            raise ValueError("fixture function must match its stable ID")
        prompt = case["prompt"]
        if case.get("context"):
            prompt = "Existing dispatch_policy.py:\n\n```python\n" + case["context"] + "```\n\n" + prompt
        case["prompt"] = prompt
        case["system_prompt"] = pack["system_prompt"]
        case["messages"] = [
            {"role": "system", "content": case["system_prompt"]},
            {"role": "user", "content": prompt},
        ]
        cases.append(case)
    if len(cases) != len(CASE_SIGNATURES) or len({case["id"] for case in cases}) != len(cases):
        raise ValueError("fixture pack must contain each coding case exactly once")
    return cases


class Rejected(ValueError):
    pass


class _Return(Exception):
    def __init__(self, value):
        self.value = value


class _Break(Exception):
    pass


class _Continue(Exception):
    pass


def _extract(text):
    if not isinstance(text, str) or len(text) > MAX_SOURCE:
        raise Rejected("response is not bounded text")
    text = text.strip()
    # Some Qwen responses contain a completed thinking block despite no-thinking
    # prompting. It is not code and is never evaluated; incomplete blocks fail.
    if text.startswith("<think>"):
        end = text.find("</think>")
        if end < 0:
            raise Rejected("unclosed thinking block")
        text = text[end + len("</think>"):].strip()
    if text.startswith("```"):
        match = re.fullmatch(r"```(?:python|py)?[ \t]*\n(.*?)\n?```", text, re.DOTALL)
        if not match:
            raise Rejected("expected exactly one complete Python code fence")
        text = match.group(1).strip()
    if not text:
        raise Rejected("response contains no code")
    return text


def _parse(source, case_id):
    tree = ast.parse(source, mode="exec")
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise Rejected("return exactly one function definition")
    function = tree.body[0]
    if function.name != case_id or function.decorator_list or function.returns:
        raise Rejected("function name, decorators, or annotations violate the contract")
    args = function.args
    if (len(args.args) != CASE_SIGNATURES[case_id] or args.posonlyargs or args.kwonlyargs
            or args.vararg or args.kwarg or args.defaults or args.kw_defaults):
        raise Rejected("function must have the requested positional signature")
    if len({arg.arg for arg in args.args}) != len(args.args):
        raise Rejected("duplicate function argument names")
    nodes = list(ast.walk(tree))
    if len(nodes) > 2000:
        raise Rejected("function AST is too large")
    for node in nodes:
        if type(node) not in ALLOWED_NODES:
            raise Rejected("unsupported Python construct: " + type(node).__name__)
        if isinstance(node, ast.FunctionDef) and node is not function:
            raise Rejected("nested functions are outside the fixture contract")
        if isinstance(node, ast.arg) and (node.annotation or node.arg.startswith("_")):
            raise Rejected("annotated or private argument names are unavailable")
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise Rejected("private interpreter names are unavailable")
        if isinstance(node, ast.Constant):
            if type(node.value) not in (str, int, float, bool, type(None)):
                raise Rejected("unsupported constant type")
            if isinstance(node.value, str) and len(node.value) > 4096:
                raise Rejected("string constant too large")
            if type(node.value) in (int, float) and abs(node.value) > 10**12:
                raise Rejected("numeric constant too large")
        if isinstance(node, ast.Attribute) and node.attr not in METHODS:
            raise Rejected("attribute is not an approved container method")
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id not in BUILTINS:
                    raise Rejected("only approved builtin calls are available")
            elif not isinstance(node.func, ast.Attribute):
                raise Rejected("indirect calls are unavailable")
            if any(keyword.arg is None for keyword in node.keywords):
                raise Rejected("expanded keyword arguments are unavailable")
        if isinstance(node, ast.comprehension) and node.is_async:
            raise Rejected("asynchronous comprehensions are unavailable")

    def check_loop_scope(statements, loop_depth=0):
        # ast.parse deliberately omits a few compiler checks. Apply the relevant
        # control-flow check here without compiling any generated Python.
        for statement in statements:
            if isinstance(statement, (ast.Break, ast.Continue)) and not loop_depth:
                raise Rejected("break or continue outside a loop")
            if isinstance(statement, (ast.For, ast.While)):
                check_loop_scope(statement.body, loop_depth + 1)
                check_loop_scope(statement.orelse, loop_depth)
            elif isinstance(statement, ast.If):
                check_loop_scope(statement.body, loop_depth)
                check_loop_scope(statement.orelse, loop_depth)

    check_loop_scope(function.body)
    return function


def _bounded(value):
    if type(value) in (list, tuple, dict, set, str, range) + DICT_VIEWS and len(value) > MAX_ITEMS:
        raise Rejected("container size limit exceeded")
    if type(value) in (int, float) and (not math.isfinite(value) or abs(value) > 10**12):
        raise Rejected("numeric size limit exceeded")
    return value


class _Lambda:
    def __init__(self, evaluator, node, environment):
        args = node.args
        if args.vararg or args.kwarg or args.defaults or args.kwonlyargs or args.posonlyargs:
            raise Rejected("only simple lambda sort keys are available")
        if len({arg.arg for arg in args.args}) != len(args.args):
            raise Rejected("duplicate lambda argument names")
        self.evaluator = evaluator
        self.node = node
        # Python closures resolve names when called, after later local rebindings.
        self.environment = environment

    def __call__(self, *args):
        if len(args) != len(self.node.args.args):
            raise Rejected("lambda argument count mismatch")
        env = ChainMap(dict(zip((arg.arg for arg in self.node.args.args), args)), self.environment)
        return self.evaluator.expression(self.node.body, env)


class _Iterator:
    """Only evaluator-created iterators; preserve Python's live/lazy semantics."""

    def __init__(self, values, evaluator):
        self.values = iter(values)
        self.evaluator = evaluator
        self.count = 0

    def __iter__(self):
        return self

    def __next__(self):
        self.evaluator.tick()
        value = next(self.values)
        self.count += 1
        if self.count > MAX_ITEMS:
            raise Rejected("iterator size limit exceeded")
        return value


class _Evaluator:
    def __init__(self, function):
        self.function = function
        self.fuel = MAX_FUEL
        self.loop_iterations = 0

    def tick(self):
        self.fuel -= 1
        if self.fuel < 0:
            raise Rejected("evaluation fuel exhausted")

    def run(self, arguments):
        env = dict(zip((arg.arg for arg in self.function.args.args), arguments))
        try:
            self.block(self.function.body, env)
        except _Return as result:
            return result.value
        return None

    def assign(self, target, value, env):
        self.tick()
        if isinstance(target, ast.Name):
            if target.id in BUILTINS:
                raise Rejected("builtin names cannot be reassigned")
            env[target.id] = _bounded(value)
        elif isinstance(target, (ast.Tuple, ast.List)):
            values = list(self.iterable(value))
            if len(values) != len(target.elts):
                raise Rejected("unpacking length mismatch")
            for item, assigned in zip(target.elts, values):
                self.assign(item, assigned, env)
        elif isinstance(target, ast.Subscript):
            container = self.expression(target.value, env)
            key = self.expression(target.slice, env)
            if type(container) not in (list, dict):
                raise Rejected("only list and dict elements can be assigned")
            container[key] = value
            _bounded(container)
        else:
            raise Rejected("unsupported assignment target")

    def block(self, statements, env):
        for statement in statements:
            self.statement(statement, env)

    def statement(self, node, env):
        self.tick()
        if isinstance(node, ast.Return):
            raise _Return(self.expression(node.value, env) if node.value else None)
        if isinstance(node, ast.Assign):
            value = self.expression(node.value, env)
            for target in node.targets:
                self.assign(target, value, env)
        elif isinstance(node, ast.AugAssign):
            old = self.expression(node.target, env)
            right = self.expression(node.value, env)
            if isinstance(node.op, ast.Add) and type(old) is list:
                old.extend(self.iterable(right))
                value = _bounded(old)
            else:
                value = self.binary(node.op, old, right)
            self.assign(node.target, value, env)
        elif isinstance(node, ast.If):
            self.block(node.body if self.expression(node.test, env) else node.orelse, env)
        elif isinstance(node, (ast.For, ast.While)):
            broken = False
            values = iter(self.iterable(self.expression(node.iter, env))) if isinstance(node, ast.For) else None
            while True:
                self.tick()
                if values is not None:
                    try:
                        self.assign(node.target, next(values), env)
                    except StopIteration:
                        break
                elif not self.expression(node.test, env):
                    break
                self.loop_iterations += 1
                try:
                    self.block(node.body, env)
                except _Continue:
                    continue
                except _Break:
                    broken = True
                    break
            if not broken:
                self.block(node.orelse, env)
        elif isinstance(node, ast.Break):
            raise _Break()
        elif isinstance(node, ast.Continue):
            raise _Continue()
        elif isinstance(node, ast.Expr):
            self.expression(node.value, env)
        elif not isinstance(node, ast.Pass):
            raise Rejected("unsupported statement")

    @staticmethod
    def iterable(value):
        if isinstance(value, _Iterator):
            return value
        if type(value) not in (list, tuple, dict, set, str, range) + DICT_VIEWS:
            raise Rejected("only bounded builtin containers can be iterated")
        _bounded(value)
        # A snapshot would hide skipped/repeated elements when code mutates a
        # local list during iteration, producing false correctness passes.
        return value

    def binary(self, operator, left, right):
        if type(left) not in (int, float, bool, list, tuple, str) or type(right) not in (int, float, bool, list, tuple, str):
            raise Rejected("unsupported arithmetic operand")
        if isinstance(operator, ast.Mult):
            if type(left) in (list, tuple, str) and type(right) is int and len(left) * max(0, right) > MAX_ITEMS:
                raise Rejected("container repetition limit exceeded")
            if type(right) in (list, tuple, str) and type(left) is int and len(right) * max(0, left) > MAX_ITEMS:
                raise Rejected("container repetition limit exceeded")
        if isinstance(operator, ast.Mod) and type(left) is str:
            raise Rejected("string formatting is unavailable")
        operations = {
            ast.Add: lambda: left + right, ast.Sub: lambda: left - right,
            ast.Mult: lambda: left * right, ast.FloorDiv: lambda: left // right,
            ast.Div: lambda: left / right, ast.Mod: lambda: left % right,
        }
        return _bounded(operations[type(operator)]())

    def comprehension(self, node, env):
        scope = ChainMap({}, env)
        # Python evaluates a generator expression's first iterable immediately;
        # its body and later iterables remain lazy and see current outer bindings.
        first_values = iter(self.iterable(self.expression(node.generators[0].iter, env)))

        def visit(index):
            self.tick()
            if index == len(node.generators):
                value = (self.expression(node.key, scope), self.expression(node.value, scope)) if isinstance(node, ast.DictComp) else self.expression(node.elt, scope)
                yield value
                return
            generator = node.generators[index]
            values = first_values if index == 0 else iter(self.iterable(self.expression(generator.iter, scope)))
            for item in values:
                self.loop_iterations += 1
                self.assign(generator.target, item, scope)
                if all(self.expression(condition, scope) for condition in generator.ifs):
                    yield from visit(index + 1)

        values = _Iterator(visit(0), self)
        if isinstance(node, ast.GeneratorExp):
            return values
        output = list(values)
        if isinstance(node, ast.DictComp):
            return dict(output)
        if isinstance(node, ast.SetComp):
            return set(output)
        return output

    def expression(self, node, env):
        self.tick()
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            if node.id not in env:
                raise Rejected("unbound name: " + node.id)
            return env[node.id]
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            values = [self.expression(item, env) for item in node.elts]
            return {ast.List: list, ast.Tuple: tuple, ast.Set: set}[type(node)](values)
        if isinstance(node, ast.Dict):
            if any(key is None for key in node.keys):
                raise Rejected("dictionary expansion is unavailable")
            return {self.expression(key, env): self.expression(value, env) for key, value in zip(node.keys, node.values)}
        if isinstance(node, ast.Subscript):
            container = self.expression(node.value, env)
            if type(container) not in (list, tuple, dict, str, range):
                raise Rejected("unsupported subscript container")
            return container[self.expression(node.slice, env)]
        if isinstance(node, ast.Slice):
            return slice(*(self.expression(value, env) if value is not None else None for value in (node.lower, node.upper, node.step)))
        if isinstance(node, ast.BinOp):
            return self.binary(node.op, self.expression(node.left, env), self.expression(node.right, env))
        if isinstance(node, ast.UnaryOp):
            value = self.expression(node.operand, env)
            if isinstance(node.op, ast.Not):
                return not value
            if type(value) not in (int, float, bool):
                raise Rejected("numeric unary operation requires a number")
            return -value if isinstance(node.op, ast.USub) else +value
        if isinstance(node, ast.BoolOp):
            result = None
            for value in node.values:
                result = self.expression(value, env)
                if isinstance(node.op, ast.And) and not result:
                    break
                if isinstance(node.op, ast.Or) and result:
                    break
            return result
        if isinstance(node, ast.Compare):
            left = self.expression(node.left, env)
            for operator, comparator in zip(node.ops, node.comparators):
                right = self.expression(comparator, env)
                comparisons = {
                    ast.Eq: lambda: left == right, ast.NotEq: lambda: left != right,
                    ast.Lt: lambda: left < right, ast.LtE: lambda: left <= right,
                    ast.Gt: lambda: left > right, ast.GtE: lambda: left >= right,
                    ast.In: lambda: left in right, ast.NotIn: lambda: left not in right,
                    ast.Is: lambda: left is right, ast.IsNot: lambda: left is not right,
                }
                if not comparisons[type(operator)]():
                    return False
                left = right
            return True
        if isinstance(node, ast.IfExp):
            return self.expression(node.body if self.expression(node.test, env) else node.orelse, env)
        if isinstance(node, ast.Lambda):
            return _Lambda(self, node, env)
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            return self.comprehension(node, env)
        if isinstance(node, ast.Call):
            args = [self.expression(value, env) for value in node.args]
            kwargs = {keyword.arg: self.expression(keyword.value, env) for keyword in node.keywords}
            if isinstance(node.func, ast.Name):
                return self.builtin(node.func.id, args, kwargs)
            container = self.expression(node.func.value, env)
            return self.method(container, node.func.attr, args, kwargs)
        raise Rejected("unsupported expression")

    def builtin(self, name, args, kwargs):
        self.tick()
        if name == "range":
            if kwargs or any(type(value) is not int for value in args):
                raise Rejected("range requires positional integers")
            return _bounded(range(*args))
        if name in ("enumerate", "reversed", "zip"):
            if kwargs:
                raise Rejected("iterator keyword arguments are unavailable")
            if name == "enumerate":
                return _Iterator(enumerate(self.iterable(args[0]), *args[1:]), self)
            if name == "reversed":
                if len(args) != 1:
                    raise Rejected("reversed requires one argument")
                return _Iterator(reversed(self.iterable(args[0])), self)
            return _Iterator(zip(*(self.iterable(value) for value in args)), self)
        if name in ("sorted", "min", "max"):
            if set(kwargs) - {"key", "reverse", "default"}:
                raise Rejected("unsupported ordering keyword")
            if "key" in kwargs and kwargs["key"] is not None and not isinstance(kwargs["key"], _Lambda):
                raise Rejected("only a local lambda can be an ordering key")
            if name == "sorted":
                if len(args) != 1 or "default" in kwargs:
                    raise Rejected("unsupported sorted signature")
                return sorted(self.iterable(args[0]), **kwargs)
            if "reverse" in kwargs:
                raise Rejected("unsupported min/max signature")
            values = [self.iterable(args[0])] if len(args) == 1 else args
            return _bounded((min if name == "min" else max)(*values, **kwargs))
        if kwargs:
            raise Rejected("unsupported builtin keyword arguments")
        functions = {
            "abs": abs, "all": all, "any": any, "bool": bool, "dict": dict,
            "int": int, "len": len, "list": list, "set": set, "str": str,
            "sum": sum, "tuple": tuple,
        }
        if name not in functions:
            raise Rejected("unknown builtin")
        if name == "str" and any(type(value) not in (str, int, float, bool, type(None)) for value in args):
            # A tiny nested DAG can expand exponentially in native repr/str
            # before a post-conversion length check gets a chance to reject it.
            raise Rejected("str only supports bounded scalar values")
        if name in ("all", "any", "dict", "list", "set", "sum", "tuple") and args:
            if name != "dict" or type(args[0]) is not dict:
                args[0] = self.iterable(args[0])
        if any(isinstance(value, _Lambda) for value in args):
            raise Rejected("callable values are only allowed as ordering keys")
        return _bounded(functions[name](*args))

    def method(self, container, name, args, kwargs):
        self.tick()
        if type(container) is list and name in ("append", "extend", "copy", "sort", "pop"):
            if name == "sort":
                if args or set(kwargs) - {"key", "reverse"}:
                    raise Rejected("unsupported list.sort signature")
                if kwargs.get("key") is not None and not isinstance(kwargs["key"], _Lambda):
                    raise Rejected("only a lambda can be a sort key")
                container.sort(**kwargs)
                return None
            if kwargs:
                raise Rejected("list method keywords are unavailable")
            if name == "extend":
                if len(args) != 1:
                    raise Rejected("extend requires one argument")
                args = [self.iterable(args[0])]
            result = {"append": container.append, "extend": container.extend, "copy": container.copy, "pop": container.pop}[name](*args)
            _bounded(container)
            return _bounded(result)
        if type(container) is dict and name in ("get", "items", "keys", "values"):
            if kwargs:
                raise Rejected("dict method keywords are unavailable")
            result = {"get": container.get, "items": container.items, "keys": container.keys, "values": container.values}[name](*args)
            # Keep live dictionary views; materializing would hide later edits.
            return _bounded(result)
        raise Rejected("method is unavailable for this container type")


def _oracle_intervals(intervals):
    # Independent coverage oracle: sorted unique endpoints, then connected spans.
    normalized = [(min(a, b), max(a, b)) for a, b in intervals]
    pending = list(normalized)
    components = []
    while pending:
        left, right = pending.pop()
        changed = True
        while changed:
            changed = False
            rest = []
            for a, b in pending:
                if a <= right and b >= left:
                    left, right = min(left, a), max(right, b)
                    changed = True
                else:
                    rest.append((a, b))
            pending = rest
        components.append((left, right))
    return sorted(components)


def _tests(case_id):
    if case_id == "merge_intervals":
        samples = [[], [(0, 0)], [(5, 1)], [(1, 2), (3, 4)], [(1, 2), (2, 3)],
                   [(9, 2), (3, 4), (0, 1)], [(-10, -5), (-7, -2), (2, -2)],
                   [(4, 4)] * 4, [(8, 5), (2, 1), (2, 6)], [(10**6, -10**6), (0, 0)]]
        for seed in range(24):
            samples.append([(((seed * 7 + i * 13) % 67) - 33, ((seed * 11 + i * 3) % 67) - 33) for i in range(seed % 11 + 1)])
        return [("intervals-%02d" % i, [sample], _oracle_intervals(sample)) for i, sample in enumerate(samples)]
    if case_id == "lower_bound":
        samples = [([], 0), ([2], 1), ([2], 2), ([2], 3), ([1, 1, 1], 1),
                   ([-9, -4, 0, 0, 7], -10), ([-9, -4, 0, 0, 7], 0),
                   ([-9, -4, 0, 0, 7], 8)]
        for seed in range(16):
            values = sorted(((i * 19 + seed * 7) % 97) - 48 for i in range(seed * 7))
            samples.extend((values, target) for target in (-49, -17, 0, 19, 49))
        large = [i // 3 - 700 for i in range(4096)]
        samples.extend((large, target) for target in (-701, -700, -1, 0, 333, 666, 1000))
        return [("bounds-%03d" % i, [values, target], next((j for j, value in enumerate(values) if value >= target), len(values))) for i, (values, target) in enumerate(samples)]
    if case_id == "select_jobs":
        samples = [([], 0), ([("a", 1, 0)], 0), ([("big", 9, 9), ("small", 1, 2)], 2),
                   ([("a", 1, 3), ("b", 1, 3), ("c", 1, 0)], 3),
                   ([("low", -3, 1), ("high", -1, 1), ("middle", -2, 1)], 2),
                   ([("a", 1, 2), ("b", 9, 2), ("c", 9, 0), ("d", 1, 0)], 2)]
        for seed in range(24):
            jobs = [("job-%02d-%02d" % (seed, i), (i * 7 + seed) % 9 - 4, (i * 11 + seed * 3) % 13) for i in range(seed % 17 + 1)]
            samples.append((jobs, (seed * 7) % 31))
        cases = []
        for i, (jobs, budget) in enumerate(samples):
            remaining = budget
            expected = []
            # Explicit priority buckets give an oracle distinct from a sort key.
            for priority in sorted({job[1] for job in jobs}, reverse=True):
                for job_id, actual_priority, cost in jobs:
                    if actual_priority == priority and cost <= remaining:
                        expected.append(job_id)
                        remaining -= cost
            cases.append(("jobs-%02d" % i, [jobs, budget], expected))
        return cases
    raise ValueError("unknown coding case")


def _same_type_and_value(actual, expected):
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, (list, tuple)):
        return len(actual) == len(expected) and all(_same_type_and_value(a, b) for a, b in zip(actual, expected))
    return actual == expected


def _evaluate(case_id, text, finish_reason):
    cases = _tests(case_id)
    result = {"validator_version": VALIDATOR_VERSION, "passed": False,
              "syntax_valid": False, "contract_valid": False,
              "completion_without_truncation": finish_reason in ("stop", "eos", "end_turn"),
              "tests_passed": 0, "tests_total": len(cases), "failure": None}
    try:
        source = _extract(text)
        ast.parse(source, mode="exec")
        result["syntax_valid"] = True
        function = _parse(source, case_id)
        result["contract_valid"] = True
        for test_name, arguments, expected in cases:
            supplied = copy.deepcopy(arguments)
            evaluator = _Evaluator(function)
            actual = evaluator.run(supplied)
            if not _same_type_and_value(supplied, arguments):
                result["failure"] = test_name + ": input arguments were modified"
                break
            if not _same_type_and_value(actual, expected):
                result["failure"] = test_name + ": output type or value differs from oracle"
                break
            if case_id == "lower_bound" and evaluator.loop_iterations > math.ceil(math.log2(len(arguments[0]) + 1)) + 2:
                result["failure"] = test_name + ": exceeded logarithmic loop-iteration bound"
                break
            result["tests_passed"] += 1
        if not result["completion_without_truncation"] and result["failure"] is None:
            result["failure"] = "generation did not finish normally"
        result["passed"] = result["tests_passed"] == len(cases) and result["completion_without_truncation"]
    except (Rejected, SyntaxError, ValueError, TypeError, KeyError, IndexError, ZeroDivisionError, OverflowError, RecursionError, RuntimeError, _Break, _Continue) as error:
        result["failure"] = type(error).__name__ + ": " + str(error)[:240]
    return result


def _set_resource_limits():
    import resource
    for resource_name, soft, hard in (
        ("RLIMIT_CPU", 3, 4), ("RLIMIT_AS", 768 * 1024 * 1024, 768 * 1024 * 1024),
        ("RLIMIT_FSIZE", 1024 * 1024, 1024 * 1024), ("RLIMIT_NOFILE", 32, 32),
        ("RLIMIT_CORE", 0, 0),
    ):
        resource_id = getattr(resource, resource_name, None)
        if resource_id is None:
            continue
        try:
            resource.setrlimit(resource_id, (soft, hard))
        except (OSError, ValueError):
            # macOS may reject RLIMIT_AS; fuel, size, CPU, and wall limits remain.
            if resource_name != "RLIMIT_AS":
                raise


def validate_response(case, text, finish_reason):
    """Validate in an isolated child; return a JSON-serializable score receipt."""
    started = time.monotonic()
    case_id = case["id"]
    if case_id not in CASE_SIGNATURES:
        raise ValueError("unknown coding case")
    failure = {"validator_version": VALIDATOR_VERSION, "passed": False,
               "syntax_valid": False, "contract_valid": False,
               "completion_without_truncation": finish_reason in ("stop", "eos", "end_turn"),
               "tests_passed": 0, "tests_total": len(_tests(case_id)), "failure": None}
    if not isinstance(text, str) or len(text) > MAX_SOURCE:
        failure["failure"] = "response is not bounded text"
        failure["duration_ms"] = round((time.monotonic() - started) * 1000, 3)
        return failure
    with tempfile.TemporaryDirectory(prefix="camelid-coding-check-") as directory:
        payload = Path(directory) / "response.json"
        payload.write_text(json.dumps({"case_id": case_id, "text": text, "finish_reason": finish_reason}), encoding="utf-8")
        try:
            completed = subprocess.run(
                [sys.executable, "-I", str(Path(__file__).resolve()), "--validate-child", str(payload)],
                cwd=directory, env={"PATH": os.defpath, "LANG": "C", "LC_ALL": "C"},
                stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=8, check=False,
            )
            if completed.returncode != 0:
                failure["failure"] = "validator child failed or exceeded resource limits (exit %d)" % completed.returncode
                result = failure
            else:
                result = json.loads(completed.stdout)
        except subprocess.TimeoutExpired:
            failure["failure"] = "validator wall-time limit exceeded"
            result = failure
        except (ValueError, OSError):
            failure["failure"] = "validator could not produce a valid result"
            result = failure
    result["duration_ms"] = round((time.monotonic() - started) * 1000, 3)
    return result


GOLDENS = {
    "merge_intervals": """def merge_intervals(intervals):
    ordered = sorted((min(a, b), max(a, b)) for a, b in intervals)
    merged = []
    for start, end in ordered:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged
""",
    "lower_bound": """def lower_bound(values, target):
    left, right = 0, len(values)
    while left < right:
        middle = (left + right) // 2
        if values[middle] < target:
            left = middle + 1
        else:
            right = middle
    return left
""",
    "select_jobs": """def select_jobs(jobs, budget):
    chosen = []
    for job_id, priority, cost in sorted(jobs, key=lambda job: -job[1]):
        if cost <= budget:
            chosen.append(job_id)
            budget -= cost
    return chosen
""",
}


def self_check():
    """Run only on the designated test host. Raise AssertionError on regression."""
    checks = []

    def check(name, case_id, code, should_pass, finish_reason="stop", failure_contains=None):
        result = validate_response({"id": case_id}, code, finish_reason)
        if result["passed"] is not should_pass:
            raise AssertionError(name + ": " + json.dumps(result, sort_keys=True))
        if failure_contains is not None and failure_contains not in (result["failure"] or ""):
            raise AssertionError(name + ": wrong rejection: " + json.dumps(result, sort_keys=True))
        checks.append({"name": name, "passed": True})

    for case_id, code in GOLDENS.items():
        check("golden-" + case_id, case_id, code, True)
        check("fenced-" + case_id, case_id, "```python\n" + code + "```", True)
        check("truncated-" + case_id, case_id, code, False, "length")
    check("touching-must-merge", "merge_intervals", GOLDENS["merge_intervals"].replace("start <=", "start <"), False)
    check("normalize-reversed", "merge_intervals", GOLDENS["merge_intervals"].replace("(min(a, b), max(a, b))", "(a, b)"), False)
    check("first-duplicate", "lower_bound", GOLDENS["lower_bound"].replace("values[middle] <", "values[middle] <="), False)
    check("empty-and-tail", "lower_bound", GOLDENS["lower_bound"].replace("0, len(values)", "0, len(values) - 1"), False)
    check("stable-priorities", "select_jobs", GOLDENS["select_jobs"].replace("sorted(jobs, key=lambda job: -job[1])", "jobs"), False)
    check("zero-cost", "select_jobs", GOLDENS["select_jobs"].replace("if cost <= budget:", "if budget > 0 and cost <= budget:"), False)
    check("skip-oversized", "select_jobs", GOLDENS["select_jobs"].replace("        if cost <= budget:", "        if cost > budget:\n            break\n        if cost <= budget:"), False)
    check("import-rejected", "lower_bound", "def lower_bound(values, target):\n    import os\n    return 0", False)
    check("host-call-rejected", "lower_bound", "def lower_bound(values, target):\n    return open('/tmp/forbidden', 'w')", False)
    check("introspection-rejected", "lower_bound", "def lower_bound(values, target):\n    return values.__class__", False)
    check("recursion-rejected", "lower_bound", "def lower_bound(values, target):\n    return lower_bound(values, target)", False)
    check("loop-bounded", "lower_bound", "def lower_bound(values, target):\n    while True:\n        pass", False)
    check("allocation-bounded", "lower_bound", "def lower_bound(values, target):\n    values = [0] * 1000000000\n    return 0", False)
    check("input-mutation-rejected", "select_jobs", "def select_jobs(jobs, budget):\n    jobs.append(('extra', 0, 0))\n    return []", False)
    live_list = GOLDENS["select_jobs"].replace(
        "    chosen = []\n    for job_id, priority, cost in sorted(jobs, key=lambda job: -job[1]):",
        "    chosen = []\n    ordered = sorted(jobs, key=lambda job: -job[1])\n    for job_id, priority, cost in ordered:",
    ).replace("            budget -= cost", "            budget -= cost\n        ordered.pop(0)")
    check("live-list-iteration", "select_jobs", live_list, False, failure_contains="output type or value")
    live_enumerate = live_list.replace(
        "    for job_id, priority, cost in ordered:",
        "    for index, job in enumerate(ordered):\n        job_id, priority, cost = job",
    )
    check("live-enumerate-iteration", "select_jobs", live_enumerate, False, failure_contains="output type or value")
    live_zip = live_list.replace(
        "    for job_id, priority, cost in ordered:",
        "    for job, marker in zip(ordered, ordered):\n        job_id, priority, cost = job",
    )
    check("live-zip-iteration", "select_jobs", live_zip, False, failure_contains="output type or value")
    late_lambda = GOLDENS["select_jobs"].replace(
        "    chosen = []", "    chosen = []\n    direction = -1\n    key = lambda job: direction * job[1]\n    direction = 1",
    ).replace("key=lambda job: -job[1]", "key=key")
    check("lambda-late-binding", "select_jobs", late_lambda, False, failure_contains="output type or value")
    lazy_generator = GOLDENS["select_jobs"].replace(
        "    chosen = []", "    chosen = []\n    direction = 1\n    transformed = ((job[0], direction * job[1], job[2]) for job in jobs)\n    direction = -1",
    ).replace("sorted(jobs,", "sorted(transformed,")
    check("generator-late-binding", "select_jobs", lazy_generator, False, failure_contains="output type or value")
    short_circuit = GOLDENS["select_jobs"].replace(
        "    chosen = []", "    seen = []\n    any(seen.append(i) or i == 1 for i in range(4))\n    if seen != [0, 1]:\n        return ['wrong']\n    chosen = []",
    )
    check("generator-short-circuit", "select_jobs", short_circuit, True)
    live_dict_view = GOLDENS["select_jobs"].replace(
        "    chosen = []", "    mapping = {'first': 1}\n    view = mapping.values()\n    mapping['first'] = 2\n    if list(view) != [2]:\n        return ['wrong']\n    chosen = []",
    )
    check("live-dict-view", "select_jobs", live_dict_view, True)
    float_mutation = GOLDENS["lower_bound"].replace(
        "    left, right", "    if values:\n        values[0] = values[0] + 0.0\n    left, right",
    )
    check("input-float-mutation", "lower_bound", float_mutation, False, failure_contains="input arguments were modified")
    bool_mutation = GOLDENS["lower_bound"].replace(
        "    left, right", "    if len(values) > 2 and values[2] == 0:\n        values[2] = False\n    left, right",
    )
    check("input-bool-mutation", "lower_bound", bool_mutation, False, failure_contains="input arguments were modified")
    nested_string = GOLDENS["lower_bound"].replace(
        "    left, right", "    nested = [0]\n    for level in range(64):\n        nested = [nested, nested]\n    str(nested)\n    left, right",
    )
    check("nested-string-preflight", "lower_bound", nested_string, False, failure_contains="str only supports bounded scalar values")
    return {"validator_version": VALIDATOR_VERSION, "passed": True, "checks_passed": len(checks), "checks": checks}


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--validate-child":
        _set_resource_limits()
        request = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
        print(json.dumps(_evaluate(request["case_id"], request["text"], request["finish_reason"]), sort_keys=True))
    elif sys.argv[1:] == ["--self-check"]:
        print(json.dumps(self_check(), sort_keys=True))
    else:
        raise SystemExit("usage: qwen_coding_fixtures.py --self-check")
