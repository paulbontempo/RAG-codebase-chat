from codebase_chat_tool.graph.ast_visitor import analyze_file


def test_module_level_function_and_class_defs():
    source = """
def foo():
    pass

class Bar:
    def method(self):
        pass
"""
    fa = analyze_file("mod", source)
    qualnames = {d.qualname: d.kind for d in fa.defs}
    assert qualnames == {"mod.foo": "function", "mod.Bar": "class", "mod.Bar.method": "method"}


def test_resolves_call_to_locally_defined_function():
    source = """
def helper():
    pass

def caller():
    helper()
"""
    fa = analyze_file("mod", source)
    call = next(c for c in fa.calls if c.caller_qualname == "mod.caller")
    assert call.callee == "mod.helper"
    assert call.resolved is True


def test_resolves_call_to_imported_symbol():
    source = """
from other import thing

def caller():
    thing()
"""
    fa = analyze_file("mod", source)
    call = next(c for c in fa.calls if c.caller_qualname == "mod.caller")
    assert call.callee == "other.thing"
    assert call.resolved is True


def test_resolves_self_method_call_within_class():
    source = """
class Widget:
    def a(self):
        self.b()

    def b(self):
        pass
"""
    fa = analyze_file("mod", source)
    call = next(c for c in fa.calls if c.caller_qualname == "mod.Widget.a")
    assert call.callee == "mod.Widget.b"
    assert call.resolved is True


def test_call_inside_comprehension_is_still_attributed_to_enclosing_function():
    source = """
def transform(x):
    return x

def caller(items):
    return [transform(i) for i in items]
"""
    fa = analyze_file("mod", source)
    call = next(c for c in fa.calls if c.callee == "mod.transform")
    assert call.caller_qualname == "mod.caller"
    assert call.resolved is True


def test_dynamic_call_on_local_variable_is_unresolved():
    source = """
def caller():
    x = get_thing()
    x.do_something()
"""
    fa = analyze_file("mod", source)
    dynamic_call = next(c for c in fa.calls if c.callee == "x.do_something")
    assert dynamic_call.resolved is False


def test_decorated_function_call_is_still_resolved():
    source = """
def deco(f):
    return f

@deco
def target():
    pass

def caller():
    target()
"""
    fa = analyze_file("mod", source)
    call = next(c for c in fa.calls if c.callee == "mod.target")
    assert call.resolved is True


def test_relative_import_from_ordinary_submodule_resolves_to_parent_package():
    # pkg/sub.py: `from .exceptions import Timeout` -> pkg.exceptions.Timeout
    source = """
from .exceptions import Timeout

def caller():
    Timeout()
"""
    fa = analyze_file("pkg.sub", source)
    call = next(c for c in fa.calls if c.caller_qualname == "pkg.sub.caller")
    assert call.callee == "pkg.exceptions.Timeout"
    assert call.resolved is True


def test_relative_import_from_package_init_resolves_to_same_package():
    # pkg/__init__.py: `from .exceptions import Timeout` -> pkg.exceptions.Timeout
    source = """
from .exceptions import Timeout

def caller():
    Timeout()
"""
    fa = analyze_file("pkg", source, is_package_init=True)
    call = next(c for c in fa.calls if c.caller_qualname == "pkg.caller")
    assert call.callee == "pkg.exceptions.Timeout"
    assert call.resolved is True


def test_relative_import_from_import_only_binds_sibling_submodule():
    # pkg/api.py: `from . import sessions` -> local name "sessions" bound to pkg.sessions
    source = """
from . import sessions

def caller():
    sessions.Session()
"""
    fa = analyze_file("pkg.api", source)
    call = next(c for c in fa.calls if c.caller_qualname == "pkg.api.caller")
    assert call.callee == "pkg.sessions.Session"
    assert call.resolved is True


def test_double_dot_relative_import_goes_up_two_packages():
    # pkg/sub/deep.py: `from ..utils import helper` -> pkg.utils.helper
    source = """
from ..utils import helper

def caller():
    helper()
"""
    fa = analyze_file("pkg.sub.deep", source)
    call = next(c for c in fa.calls if c.caller_qualname == "pkg.sub.deep.caller")
    assert call.callee == "pkg.utils.helper"
    assert call.resolved is True
