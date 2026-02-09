"""Prompts for response generation."""

from __future__ import annotations

from typing import Any, Dict, List


def build_resp_sys_prompt(use_rag: bool) -> str:
    """Build the system prompt for response generation."""
    if use_rag:
        return (
            "You are a careful, factual question-answering assistant.\n"
            "You must answer using ONLY the information in CONTEXT.\n"
            "If CONTEXT is insufficient, reply exactly: I do not know.\n\n"
            "Rules:\n"
            "- Do not use outside knowledge.\n"
            "- Do not guess or infer beyond what is directly supported.\n"
            "- If the question has multiple parts, answer only supported parts;\n"
            "  otherwise reply exactly: I do not know.\n"
            "- Include citations in square brackets referring to sources (e.g., [1]).\n"
            "- If you cannot cite at least one source, reply exactly: I do not know.\n\n"
            "Output:\n"
            "- Output ONLY the answer text.\n"
            "- No headings, no preamble, no explanations.\n"
        )

    return (
        "You are a careful, factual question-answering assistant.\n"
        "Answer the question as accurately as you can.\n"
        "If you are unsure, reply exactly: I do not know.\n\n"
        "Output:\n"
        "- Output ONLY the answer text.\n"
        "- No headings, no preamble, no explanations.\n"
    )


def build_resp_user_prompt(query: str, contexts: List[Dict[str, Any]]) -> str:
    """Build the user prompt with a structured, delimited context block."""
    lines: List[str] = []
    for i, ctx in enumerate(contexts, start=1):
        title = (ctx.get("title") or "").strip()
        text = (ctx.get("text") or "").strip()
        if title:
            lines.append(f"[{i}] {title}\n{text}")
        else:
            lines.append(f"[{i}]\n{text}")

    context_block = "\n\n".join(lines) if lines else "(no context provided)"

    return (
        "QUESTION:\n"
        f"{query}\n\n"
        "CONTEXT (numbered sources):\n"
        "---BEGIN CONTEXT---\n"
        f"{context_block}\n"
        "---END CONTEXT---\n\n"
        "ANSWER:\n"
    )
