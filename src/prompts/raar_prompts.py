"""Prompts used by RAAR (decomposition + critic)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def _truncate_text(text: str, max_chars: Optional[int]) -> str:
    if max_chars is None:
        return text
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def format_contexts_for_critic(
    contexts: List[Dict[str, Any]],
    max_chars: int = 6000,
) -> str:
    """Format numbered contexts for the critic prompt."""
    lines: List[str] = []
    for i, ctx in enumerate(contexts, start=1):
        title = (ctx.get("title") or "").strip()
        text = (ctx.get("text") or "").strip()
        if title:
            lines.append(f"[{i}] {title}\n{text}")
        else:
            lines.append(f"[{i}]\n{text}")

    block = "\n\n".join(lines) if lines else "(no contexts provided)"
    return _truncate_text(block, max_chars=max_chars)


def build_decompose_sys_prompt() -> str:
    return (
        "You are a careful multi-hop question decomposition assistant.\n"
        "Given a QUESTION, produce a structured decomposition for retrieval.\n"
        "Return ONLY JSON.\n\n"
        "JSON schema:\n"
        "{\n"
        "  \"hops\": [\"subquestion 1\", \"subquestion 2\", ...],\n"
        "  \"query_variants\": [\"search query 1\", \"search query 2\", ...]\n"
        "}\n\n"
        "Rules:\n"
        "- 2 to 4 hops is typical.\n"
        "- query_variants should include the original QUESTION plus improved queries.\n"
        "- Avoid overly long queries.\n"
        "- Output valid JSON only.\n"
    )


def build_decompose_user_prompt(question: str) -> str:
    return (
        "QUESTION:\n"
        f"{question}\n\n"
        "Return ONLY the JSON object."
    )


def build_critic_sys_prompt() -> str:
    return (
        "You are a Retrieval-Aware Adversarial RAG critic and planner.\n"
        "You will be given:\n"
        "- QUESTION\n"
        "- CURRENT_ANSWER\n"
        "- CONTEXTS (numbered sources)\n\n"
        "Your job:\n"
        "1) Check if CURRENT_ANSWER is fully supported by CONTEXTS and answers QUESTION.\n"
        "2) If incomplete/unsupported, propose adversarial retrieval queries to fill gaps.\n"
        "3) Be strict and transparent.\n\n"
        "Return ONLY valid JSON with this schema:\n"
        "{\n"
        "  \"decision\": \"accept\" | \"revise\" | \"give_up\",\n"
        "  \"issues\": [\"...\"],\n"
        "  \"missing_info\": [\"...\"],\n"
        "  \"new_queries\": [\"...\"],\n"
        "  \"suggested_strategy\": \"none\" | \"increase_k\" | \"decompose\" |\n"
        "    \"disambiguate\" | \"counter_check\",\n"
        "  \"scores\": {\n"
        "    \"groundedness\": 0.0,\n"
        "    \"completeness\": 0.0,\n"
        "    \"overall_confidence\": 0.0\n"
        "  }\n"
        "}\n\n"
        "Guidelines:\n"
        "- groundedness: claims supported by CONTEXTS.\n"
        "- completeness: covers all parts of QUESTION.\n"
        "- new_queries should be short, retrieval-friendly.\n"
        "- If CONTEXTS empty and cannot ground, choose give_up or revise with queries.\n"
        "- Only choose \"accept\" if groundedness, completeness, and "
        "overall_confidence are all >= 0.85.\n"
        "- If you choose \"revise\", you MUST provide at least 1 item in new_queries.\n"
    )


def build_critic_user_prompt(
    question: str,
    current_answer: str,
    contexts: List[Dict[str, Any]],
    max_chars: int = 6000,
) -> str:
    ctx_block = format_contexts_for_critic(contexts, max_chars=max_chars)
    return (
        "QUESTION:\n"
        f"{question}\n\n"
        "CURRENT_ANSWER:\n"
        f"{current_answer}\n\n"
        "CONTEXTS:\n"
        f"{ctx_block}\n\n"
        "Return ONLY the JSON object."
    )
