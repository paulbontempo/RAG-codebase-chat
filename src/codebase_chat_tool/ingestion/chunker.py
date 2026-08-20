from __future__ import annotations

import tree_sitter_python as tspython
from tree_sitter import Language, Node, Parser

from codebase_chat_tool.ingestion.metadata import Chunk

_LANGUAGE = Language(tspython.language())


def _get_parser() -> Parser:
    return Parser(_LANGUAGE)


def _node_text(source_bytes: bytes, node: Node) -> str:
    return source_bytes[node.start_byte : node.end_byte].decode("utf-8")


def _decorators_and_inner(node: Node) -> tuple[list[Node], Node]:
    """Given a top-level statement node, split off any `decorated_definition` wrapper."""
    if node.type == "decorated_definition":
        decorators = [c for c in node.children if c.type == "decorator"]
        inner = next(
            c for c in node.children if c.type in ("function_definition", "class_definition")
        )
        return decorators, inner
    return [], node


def _decorator_texts(source_bytes: bytes, decorators: list[Node]) -> list[str]:
    return [_node_text(source_bytes, d).lstrip("@").strip() for d in decorators]


def _def_name(node: Node) -> str:
    name_node = node.child_by_field_name("name")
    return name_node.text.decode("utf-8") if name_node else "<anonymous>"


def _docstring(source_bytes: bytes, body_node: Node | None) -> str | None:
    if body_node is None:
        return None
    for child in body_node.children:
        if child.type == "expression_statement":
            expr = child.children[0] if child.children else None
            if expr is not None and expr.type == "string":
                text = _node_text(source_bytes, expr)
                return text.strip("\"'").strip()
        break  # only the first statement counts as a docstring
    return None


def chunk_file(module: str, rel_path: str, source: str) -> list[Chunk]:
    """Structure-aware chunking: one chunk per top-level function/class/method, plus
    a single chunk for remaining module-level code (imports, constants, etc.)."""
    source_bytes = source.encode("utf-8")
    tree = _get_parser().parse(source_bytes)
    root = tree.root_node

    chunks: list[Chunk] = []
    module_level_lines: set[int] = set(range(1, len(source.splitlines()) + 2))

    for top_node in root.children:
        decorators, inner = _decorators_and_inner(top_node)
        span_start, span_end = top_node.start_point.row + 1, top_node.end_point.row + 1

        if inner.type == "function_definition":
            name = _def_name(inner)
            body = inner.child_by_field_name("body")
            chunks.append(
                Chunk(
                    file_path=rel_path,
                    qualname=f"{module}.{name}",
                    kind="function",
                    start_line=span_start,
                    end_line=span_end,
                    text=_node_text(source_bytes, top_node),
                    docstring=_docstring(source_bytes, body),
                    decorators=_decorator_texts(source_bytes, decorators),
                )
            )
            module_level_lines -= set(range(span_start, span_end + 1))

        elif inner.type == "class_definition":
            class_name = _def_name(inner)
            class_body = inner.child_by_field_name("body")

            chunks.append(
                Chunk(
                    file_path=rel_path,
                    qualname=f"{module}.{class_name}",
                    kind="class",
                    start_line=span_start,
                    end_line=span_end,
                    text=_node_text(source_bytes, top_node),
                    docstring=_docstring(source_bytes, class_body),
                    decorators=_decorator_texts(source_bytes, decorators),
                )
            )
            module_level_lines -= set(range(span_start, span_end + 1))

            if class_body is not None:
                for member in class_body.children:
                    m_decorators, m_inner = _decorators_and_inner(member)
                    if m_inner.type != "function_definition":
                        continue
                    m_name = _def_name(m_inner)
                    m_body = m_inner.child_by_field_name("body")
                    m_start, m_end = member.start_point.row + 1, member.end_point.row + 1
                    chunks.append(
                        Chunk(
                            file_path=rel_path,
                            qualname=f"{module}.{class_name}.{m_name}",
                            kind="method",
                            start_line=m_start,
                            end_line=m_end,
                            text=_node_text(source_bytes, member),
                            docstring=_docstring(source_bytes, m_body),
                            decorators=_decorator_texts(source_bytes, m_decorators),
                            class_name=class_name,
                        )
                    )

    if module_level_lines:
        lines = source.splitlines()
        module_text = "\n".join(
            lines[i - 1] for i in sorted(module_level_lines) if 0 < i <= len(lines)
        ).strip()
        if module_text:
            chunks.append(
                Chunk(
                    file_path=rel_path,
                    qualname=module,
                    kind="module",
                    start_line=min(i for i in module_level_lines if i <= len(lines)),
                    end_line=max(i for i in module_level_lines if i <= len(lines)),
                    text=module_text,
                    docstring=_docstring(source_bytes, root),
                )
            )

    return chunks
