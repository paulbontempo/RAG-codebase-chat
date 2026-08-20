from codebase_chat_tool.ingestion.chunker import chunk_file

SOURCE = '''\
import os

CONST = 1


@decorator
def top_func(x):
    """Top-level docstring."""
    return x + 1


class Foo:
    """Class docstring."""

    def method_a(self):
        return 1

    async def method_b(self):
        return 2
'''


def test_chunk_file_produces_expected_kinds_and_qualnames():
    chunks = chunk_file("mymod", "mymod.py", SOURCE)
    by_qualname = {c.qualname: c for c in chunks}

    assert set(by_qualname) == {
        "mymod.top_func",
        "mymod.Foo",
        "mymod.Foo.method_a",
        "mymod.Foo.method_b",
        "mymod",
    }
    assert by_qualname["mymod.top_func"].kind == "function"
    assert by_qualname["mymod.top_func"].decorators == ["decorator"]
    assert by_qualname["mymod.top_func"].docstring == "Top-level docstring."
    assert by_qualname["mymod.Foo"].kind == "class"
    assert by_qualname["mymod.Foo"].docstring == "Class docstring."
    assert by_qualname["mymod.Foo.method_a"].kind == "method"
    assert by_qualname["mymod.Foo.method_a"].class_name == "Foo"
    assert by_qualname["mymod.Foo.method_b"].kind == "method"


def test_module_level_chunk_contains_imports_and_constants():
    chunks = chunk_file("mymod", "mymod.py", SOURCE)
    module_chunk = next(c for c in chunks if c.kind == "module")
    assert "import os" in module_chunk.text
    assert "CONST = 1" in module_chunk.text


def test_chunk_ids_are_unique():
    chunks = chunk_file("mymod", "mymod.py", SOURCE)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
