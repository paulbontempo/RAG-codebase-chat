from functools import partial

from langgraph.graph import END, StateGraph

from codebase_chat_tool.agent.context import RepoContext
from codebase_chat_tool.agent.prompts import RISK_SYSTEM_PROMPT
from codebase_chat_tool.agent.state import ImpactState
from codebase_chat_tool.agent.tools import find_tests_for
from codebase_chat_tool.llm.base import LLMProvider, Message


def _resolve_target_node(state: ImpactState, ctx: RepoContext) -> ImpactState:
    target = state["target"]
    if target in ctx.chunks_by_qualname or target in ctx.resolver.graph.nodes:
        return {"resolved_target": target, "error": None}

    candidates = [q for q in ctx.chunks_by_qualname if q == target or q.endswith(f".{target}")]
    if len(candidates) == 1:
        return {"resolved_target": candidates[0], "error": None}
    if len(candidates) > 1:
        return {"error": f"Ambiguous target {target!r}, matches: {', '.join(sorted(candidates))}"}
    return {"error": f"Symbol {target!r} not found in the index."}


def _find_direct_callers_node(state: ImpactState, ctx: RepoContext) -> ImpactState:
    if state.get("error"):
        return {}
    infos = ctx.resolver.direct_callers(state["resolved_target"])
    return {"direct_callers": sorted({c.qualname for c in infos})}


def _find_transitive_callers_node(state: ImpactState, ctx: RepoContext) -> ImpactState:
    if state.get("error"):
        return {}
    infos = ctx.resolver.transitive_callers(state["resolved_target"])
    direct = set(state.get("direct_callers", []))
    transitive_only = sorted({c.qualname for c in infos} - direct)
    return {"transitive_callers": transitive_only}


def _find_dependent_modules_node(state: ImpactState, ctx: RepoContext) -> ImpactState:
    if state.get("error"):
        return {}
    target = state["resolved_target"]
    node_data = ctx.resolver.graph.nodes.get(target, {})
    module = node_data.get("module") or (target.rsplit(".", 1)[0] if "." in target else target)
    return {"target_module": module, "dependent_modules": sorted(ctx.resolver.importers(module))}


def _find_related_tests_node(state: ImpactState, ctx: RepoContext) -> ImpactState:
    if state.get("error"):
        return {}
    return {"related_tests": find_tests_for(ctx, state["resolved_target"])}


def _assess_risk_node(state: ImpactState, ctx: RepoContext, provider: LLMProvider) -> ImpactState:
    if state.get("error"):
        return {"risk_summary": None}

    facts = (
        f"Target: {state['resolved_target']}\n"
        f"Direct callers ({len(state.get('direct_callers', []))}): "
        f"{', '.join(state.get('direct_callers', [])) or 'none'}\n"
        f"Additional transitive callers ({len(state.get('transitive_callers', []))}): "
        f"{', '.join(state.get('transitive_callers', [])) or 'none'}\n"
        f"Modules importing {state.get('target_module')} "
        f"({len(state.get('dependent_modules', []))}): "
        f"{', '.join(state.get('dependent_modules', [])) or 'none'}\n"
        f"Tests referencing this symbol ({len(state.get('related_tests', []))}): "
        f"{', '.join(t['qualname'] for t in state.get('related_tests', [])) or 'none'}\n"
    )
    messages = [
        Message(role="system", content=RISK_SYSTEM_PROMPT),
        Message(role="user", content=facts),
    ]
    response = provider.generate(messages, max_tokens=400)
    return {"risk_summary": response.content}


def build_impact_graph(ctx: RepoContext, provider: LLMProvider):
    graph = StateGraph(ImpactState)
    graph.add_node("resolve_target", partial(_resolve_target_node, ctx=ctx))
    graph.add_node("direct_callers", partial(_find_direct_callers_node, ctx=ctx))
    graph.add_node("transitive_callers", partial(_find_transitive_callers_node, ctx=ctx))
    graph.add_node("dependent_modules", partial(_find_dependent_modules_node, ctx=ctx))
    graph.add_node("related_tests", partial(_find_related_tests_node, ctx=ctx))
    graph.add_node("assess_risk", partial(_assess_risk_node, ctx=ctx, provider=provider))

    graph.set_entry_point("resolve_target")
    graph.add_edge("resolve_target", "direct_callers")
    graph.add_edge("direct_callers", "transitive_callers")
    graph.add_edge("transitive_callers", "dependent_modules")
    graph.add_edge("dependent_modules", "related_tests")
    graph.add_edge("related_tests", "assess_risk")
    graph.add_edge("assess_risk", END)
    return graph.compile()


def run_impact_analysis(ctx: RepoContext, provider: LLMProvider, target: str) -> ImpactState:
    compiled = build_impact_graph(ctx, provider)
    return compiled.invoke({"target": target})
