from codebase_chat_tool.graph.ast_visitor import analyze_file
from codebase_chat_tool.graph.call_graph import build_graph
from codebase_chat_tool.graph.resolver import GraphResolver


def _build(sources: dict[str, str]) -> GraphResolver:
    analyses = [analyze_file(module, src) for module, src in sources.items()]
    return GraphResolver(build_graph(analyses))


def test_direct_callers_across_modules():
    resolver = _build(
        {
            "utils": "def helper():\n    pass\n",
            "service": "from utils import helper\n\ndef run():\n    helper()\n",
        }
    )
    callers = resolver.direct_callers("utils.helper")
    assert [c.qualname for c in callers] == ["service.run"]


def test_transitive_callers_follows_chain():
    resolver = _build(
        {
            "utils": "def helper():\n    pass\n",
            "service": "from utils import helper\n\ndef mid():\n    helper()\n",
            "main": "from service import mid\n\ndef run():\n    mid()\n",
        }
    )
    transitive = {c.qualname for c in resolver.transitive_callers("utils.helper")}
    assert transitive == {"service.mid", "main.run"}


def test_transitive_callers_is_cycle_safe():
    resolver = _build(
        {
            "a": "from b import g\n\ndef f():\n    g()\n",
            "b": "from a import f\n\ndef g():\n    f()\n",
        }
    )
    # Should terminate and not include the target itself.
    result = {c.qualname for c in resolver.transitive_callers("a.f", max_depth=10)}
    assert "a.f" not in result
    assert "b.g" in result


def test_importers():
    resolver = _build(
        {
            "utils": "def helper():\n    pass\n",
            "service": "import utils\n",
        }
    )
    assert resolver.importers("utils") == ["service"]


def test_unresolved_calls_are_not_treated_as_direct_callers():
    resolver = _build(
        {
            "utils": "def helper():\n    pass\n",
            "service": "def run():\n    something_dynamic()\n",
        }
    )
    assert resolver.direct_callers("utils.helper") == []
