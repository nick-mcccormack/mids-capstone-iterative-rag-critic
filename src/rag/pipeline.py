"""Unified pipeline for No-RAG, RAG, RAG+Rerank, and RAAR.

This module is intentionally backend-only (no Streamlit dependencies). It emits a
structured `trace` that the Streamlit app can visualize as a narrative.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from src.prompts.evaluation_prompts import build_eval_sys_prompt, build_eval_user_prompt
from src.prompts.raar_prompts import (
    build_critic_sys_prompt,
    build_critic_user_prompt,
    build_decompose_sys_prompt,
    build_decompose_user_prompt,
)
from src.prompts.response_prompts import build_resp_sys_prompt, build_resp_user_prompt
from src.rag.generator import call_llm
from src.rag.reranker import rerank_contexts
from src.rag.retriever import retrieve_contexts


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration for the pipeline."""

    mode: str = "rag"
    retrieve_top_k: Optional[int] = 5
    rerank_top_k: Optional[int] = None

    max_contexts_final: int = 5
    temperature: float = 0.2

    # RAAR settings
    raar_max_iters: int = 3
    raar_max_query_variants_per_iter: int = 3
    raar_prompt_context_max_chars: int = 6000
    raar_answer_stagnation_window: int = 1


def make_config_no_rag() -> PipelineConfig:
    return PipelineConfig(mode="no_rag", retrieve_top_k=None, rerank_top_k=None)


def make_config_rag(retrieve_top_k: int = 5) -> PipelineConfig:
    return PipelineConfig(mode="rag", retrieve_top_k=retrieve_top_k, rerank_top_k=None)


def make_config_rag_rerank(retrieve_top_k: int = 30, rerank_top_k: int = 5) -> PipelineConfig:
    return PipelineConfig(
        mode="rag_rerank",
        retrieve_top_k=retrieve_top_k,
        rerank_top_k=rerank_top_k,
    )


def make_config_raar(
    retrieve_top_k: int = 30,
    rerank_top_k: int = 5,
    raar_max_iters: int = 3,
) -> PipelineConfig:
    return PipelineConfig(
        mode="raar",
        retrieve_top_k=retrieve_top_k,
        rerank_top_k=rerank_top_k,
        raar_max_iters=raar_max_iters,
    )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _coerce_list_str(value: Any) -> List[str]:
    """Coerce a value into a list of non-empty strings.

    Models occasionally emit strings where a list is requested (e.g., a newline-
    separated list). This helper is tolerant to those cases.

    Parameters
    ----------
    value:
        Candidate value (list/tuple/str/other).

    Returns
    -------
    list of str
        Cleaned, non-empty strings.
    """
    if value is None:
        return []

    if isinstance(value, str):
        parts = re.split(r"[\n;,]+", value)
        return [p.strip() for p in parts if p and p.strip()]

    if isinstance(value, (list, tuple)):
        out: List[str] = []
        for x in value:
            if isinstance(x, str) and x.strip():
                out.append(x.strip())
            elif isinstance(x, dict):
                # Common pattern: [{"query": "..."}]
                q = x.get("query") or x.get("text") or x.get("value")
                if isinstance(q, str) and q.strip():
                    out.append(q.strip())
            else:
                try:
                    s = str(x).strip()
                except Exception:  # noqa: BLE001
                    continue
                if s:
                    out.append(s)
        return out

    return []



def _strip_code_fences(text: str) -> str:
    """Remove Markdown code fences from a model output string."""
    s = (text or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9_-]*\n", "", s)
        s = re.sub(r"\n```\s*$", "", s)
    return s.strip()


def _safe_json_load(text: str) -> Optional[Dict[str, Any]]:
    """Parse a JSON object from model output.

    Prompts request JSON-only outputs, but models sometimes add code fences or a
    short preamble. This helper is tolerant:
    1) Try parsing the whole string.
    2) If that fails, parse the substring between the first '{' and last '}'.

    Parameters
    ----------
    text:
        Raw model output.

    Returns
    -------
    dict or None
        Parsed JSON object, or ``None`` if parsing fails.
    """
    raw = _strip_code_fences(text)
    if not raw:
        return None

    def _try_parse(s: str) -> Optional[Dict[str, Any]]:
        try:
            obj = json.loads(s)
        except json.JSONDecodeError:
            return None
        return obj if isinstance(obj, dict) else None

    obj = _try_parse(raw)
    if obj is not None:
        return obj

    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    return _try_parse(raw[start : end + 1])


class TraceLogger:
    """Lightweight structured trace logger."""

    def __init__(self) -> None:
        self._events: List[Dict[str, Any]] = []
        self._idx = 0

    @property
    def events(self) -> List[Dict[str, Any]]:
        return self._events

    def log(
        self,
        component: str,
        action: str,
        stage: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._events.append(
            {
                "idx": self._idx,
                "ts": _now_iso(),
                "component": component,
                "action": action,
                "stage": stage,
                "data": data or {},
            }
        )
        self._idx += 1


def _contexts_preview(
    contexts: List[Dict[str, Any]],
    max_items: int = 5,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for c in contexts[:max_items]:
        text = (c.get("text") or "").strip()
        out.append(
            {
                "title": (c.get("title") or "").strip(),
                "doc_id": c.get("doc_id"),
                "score": c.get("score"),
                "rerank_score": c.get("rerank_score"),
                "text_preview": text[:400] + ("…" if len(text) > 400 else ""),
                "retrieval_query": c.get("retrieval_query"),
                "retrieval_strategy": c.get("retrieval_strategy"),
            }
        )
    return out


def _dedupe_contexts(contexts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Dedupe contexts by (doc_id, title, text hash)."""
    seen: set[str] = set()
    out: List[Dict[str, Any]] = []
    for c in contexts:
        doc_id = str(c.get("doc_id") or "")
        title = str(c.get("title") or "")
        text = str(c.get("text") or "")
        key = f"{doc_id}|{title}|{_hash_text(text)}"
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def _safe_call_llm(
    logger: TraceLogger,
    tag: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
) -> Tuple[str, Optional[str]]:
    """Call the LLM with trace logging. Returns (text, error)."""
    logger.log(
        component="llm",
        action="call_start",
        stage="llm",
        data={
            "tag": tag,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
        },
    )
    try:
        text = call_llm(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
        )
    except Exception as exc:  # noqa: BLE001
        err = f"{type(exc).__name__}: {exc}"
        logger.log(
            component="llm",
            action="call_error",
            stage="llm",
            data={"tag": tag, "error": err},
        )
        return "", err

    logger.log(
        component="llm",
        action="call_success",
        stage="llm",
        data={"tag": tag, "output": text},
    )
    return text, None


def _retrieve_for_queries(
    logger: TraceLogger,
    queries: List[str],
    top_k: int,
    strategy: str,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Retrieve for each query and concatenate results."""
    contexts: List[Dict[str, Any]] = []
    errors: List[str] = []

    for q in queries:
        logger.log(
            component="retriever",
            action="retrieve_start",
            stage="retrieval",
            data={"query": q, "top_k": top_k, "strategy": strategy},
        )
        try:
            hits = retrieve_contexts(query=q, top_k=top_k)
        except Exception as exc:  # noqa: BLE001
            err = f"{type(exc).__name__}: {exc}"
            errors.append(err)
            logger.log(
                component="retriever",
                action="retrieve_error",
                stage="retrieval",
                data={"query": q, "top_k": top_k, "strategy": strategy, "error": err},
            )
            continue

        for h in hits:
            item = dict(h)
            item["retrieval_query"] = q
            item["retrieval_strategy"] = strategy
            contexts.append(item)

        logger.log(
            component="retriever",
            action="retrieve_success",
            stage="retrieval",
            data={
                "query": q,
                "top_k": top_k,
                "strategy": strategy,
                "num_hits": len(hits),
                "preview": _contexts_preview(contexts[-min(5, len(hits)) :]),
            },
        )

    return contexts, errors


def _maybe_rerank(
    logger: TraceLogger,
    query: str,
    contexts: List[Dict[str, Any]],
    rerank_top_k: Optional[int],
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    if not rerank_top_k:
        return contexts, None

    logger.log(
        component="reranker",
        action="rerank_start",
        stage="rerank",
        data={"query": query, "num_candidates": len(contexts), "top_k": rerank_top_k},
    )
    try:
        reranked = rerank_contexts(query=query, candidates=contexts, top_k=rerank_top_k)
    except Exception as exc:  # noqa: BLE001
        err = f"{type(exc).__name__}: {exc}"
        logger.log(
            component="reranker",
            action="rerank_error",
            stage="rerank",
            data={"error": err},
        )
        return contexts, err

    logger.log(
        component="reranker",
        action="rerank_success",
        stage="rerank",
        data={
            "kept": len(reranked),
            "preview": _contexts_preview(reranked),
        },
    )
    return reranked, None


def _generate_answer(
    logger: TraceLogger,
    question: str,
    contexts: List[Dict[str, Any]],
    temperature: float,
    tag: str,
    use_rag: bool,
) -> Tuple[str, Optional[str]]:
    sys_prompt = build_resp_sys_prompt(use_rag=use_rag)
    usr_prompt = build_resp_user_prompt(query=question, contexts=contexts)
    text, err = _safe_call_llm(
        logger=logger,
        tag=tag,
        system_prompt=sys_prompt,
        user_prompt=usr_prompt,
        temperature=temperature,
    )
    if err:
        return "", err

    answer = (text or "").strip()
    logger.log(
        component="generator",
        action="answer_generated",
        stage="generation",
        data={
            "tag": tag,
            "answer": answer,
            "num_contexts": len(contexts),
            "contexts_preview": _contexts_preview(contexts),
        },
    )
    return answer, None


def _evaluate_answer(
    logger: TraceLogger,
    question: str,
    model_answer: str,
    gold_answer: str,
    contexts: Optional[List[Dict[str, Any]]],
    temperature: float,
) -> Tuple[Dict[str, Any], str, Optional[str]]:
    sys_prompt = build_eval_sys_prompt()
    usr_prompt = build_eval_user_prompt(
        question=question,
        model_answer=model_answer,
        gold_answer=gold_answer,
        contexts=contexts,
    )
    text, err = _safe_call_llm(
        logger=logger,
        tag="judge_eval",
        system_prompt=sys_prompt,
        user_prompt=usr_prompt,
        temperature=temperature,
    )
    if err:
        return {}, "", err

    obj = _safe_json_load(text)
    if obj is None:
        parse_err = "Judge output was not valid JSON."
        logger.log(
            component="judge",
            action="eval_parse_error",
            stage="evaluation",
            data={"error": parse_err, "raw": text},
        )
        return {}, text, parse_err

    logger.log(
        component="judge",
        action="eval_success",
        stage="evaluation",
        data={"metrics": obj},
    )
    return obj, text, None


def _decompose_question(
    logger: TraceLogger,
    question: str,
    temperature: float,
) -> Dict[str, Any]:
    sys_prompt = build_decompose_sys_prompt()
    usr_prompt = build_decompose_user_prompt(question=question)

    out, err = _safe_call_llm(
        logger=logger,
        tag="decompose",
        system_prompt=sys_prompt,
        user_prompt=usr_prompt,
        temperature=temperature,
    )
    if err:
        logger.log(
            component="raar_decompose",
            action="decompose_fallback",
            stage="raar",
            data={"reason": err},
        )
        return {"hops": [], "query_variants": [question]}

    obj = _safe_json_load(out)
    if not obj:
        logger.log(
            component="raar_decompose",
            action="decompose_parse_error",
            stage="raar",
            data={"raw": out},
        )
        return {"hops": [], "query_variants": [question]}

    hops = _coerce_list_str(obj.get("hops"))
    qv = _coerce_list_str(obj.get("query_variants"))
    if not qv:
        qv = [question]

    result = {"hops": hops, "query_variants": qv}
    logger.log(
        component="raar_decompose",
        action="decompose_success",
        stage="raar",
        data=result,
    )
    return result


def _critic_plan(
    logger: TraceLogger,
    question: str,
    current_answer: str,
    contexts: List[Dict[str, Any]],
    temperature: float,
    max_ctx_chars: int,
) -> Dict[str, Any]:
    sys_prompt = build_critic_sys_prompt()
    usr_prompt = build_critic_user_prompt(
        question=question,
        current_answer=current_answer,
        contexts=contexts,
        max_chars=max_ctx_chars,
    )
    out, err = _safe_call_llm(
        logger=logger,
        tag="raar_critic",
        system_prompt=sys_prompt,
        user_prompt=usr_prompt,
        temperature=temperature,
    )
    if err:
        logger.log(
            component="raar_critic",
            action="critic_fallback",
            stage="raar",
            data={"reason": err},
        )
        return {
            "decision": "revise",
            "issues": ["critic_error"],
            "missing_info": [],
            "new_queries": [],
            "suggested_strategy": "none",
            "scores": {"groundedness": 0.0, "completeness": 0.0, "overall_confidence": 0.0},
        }

    obj = _safe_json_load(out)
    if not obj:
        logger.log(
            component="raar_critic",
            action="critic_parse_error",
            stage="raar",
            data={"raw": out},
        )
        return {
            "decision": "revise",
            "issues": ["critic_parse_error"],
            "missing_info": [],
            "new_queries": [],
            "suggested_strategy": "none",
            "scores": {"groundedness": 0.0, "completeness": 0.0, "overall_confidence": 0.0},
        }

    logger.log(
        component="raar_critic",
        action="critic_success",
        stage="raar",
        data=obj,
    )
    return obj


def _should_stop_for_stagnation(
    prev_answer: str,
    new_answer: str,
    window: int,
) -> bool:
    if window <= 0:
        return False
    return _hash_text(prev_answer) == _hash_text(new_answer)


def _raar_loop(
    logger: TraceLogger,
    question: str,
    base_contexts: List[Dict[str, Any]],
    config: PipelineConfig,
) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
    """Run RAAR: draft -> critic -> targeted retrieval -> revision."""
    raar_meta: Dict[str, Any] = {"iterations": []}
    contexts = list(base_contexts)

    contexts = _dedupe_contexts(contexts)
    contexts = contexts[: config.max_contexts_final]
    answer, _ = _generate_answer(
        logger=logger,
        question=question,
        contexts=contexts,
        temperature=config.temperature,
        tag="raar_draft_0",
        use_rag=True,
    )

    if not contexts and answer.strip().lower() == "i do not know.":
        raar_meta["stop_reason"] = "empty_context_and_idk"
        raar_meta["best_iter_idx"] = 0
        logger.log(
            component="raar",
            action="stop",
            stage="raar",
            data={"reason": "empty_context_and_idk"},
        )
        return answer, contexts, raar_meta

    best_iter_idx = 0
    best_score = -1.0

    for iter_idx in range(1, config.raar_max_iters + 1):
        critic = _critic_plan(
            logger=logger,
            question=question,
            current_answer=answer,
            contexts=contexts,
            temperature=config.temperature,
            max_ctx_chars=config.raar_prompt_context_max_chars,
        )

        scores = critic.get("scores") if isinstance(critic.get("scores"), dict) else {}
        overall = float(scores.get("overall_confidence") or 0.0)
        if overall > best_score:
            best_score = overall
            best_iter_idx = iter_idx - 1

        raar_meta["iterations"].append(
            {
                "iter_idx": iter_idx,
                "answer": answer,
                "num_contexts": len(contexts),
                "critic": critic,
                "internal_score": overall,
            }
        )

        decision = (critic.get("decision") or "").strip().lower()

        # Be conservative about "accept": require high confidence.
        accept_threshold = 0.85
        groundedness = float(scores.get("groundedness") or 0.0)
        completeness = float(scores.get("completeness") or 0.0)
        overall_conf = float(scores.get("overall_confidence") or 0.0)

        if decision == "accept" and (
            groundedness < accept_threshold
            or completeness < accept_threshold
            or overall_conf < accept_threshold
        ):
            decision = "revise"
            critic = dict(critic)
            critic["decision"] = "revise"
            issues = _coerce_list_str(critic.get("issues"))
            issues.append("accept_overridden_low_confidence")
            critic["issues"] = issues
            if raar_meta.get("iterations"):
                raar_meta["iterations"][-1]["critic"] = critic

        if decision == "accept":
            raar_meta["stop_reason"] = "critic_accept"
            break
        if decision == "give_up":
            raar_meta["stop_reason"] = "critic_give_up"
            break

        strategy = (critic.get("suggested_strategy") or "none").strip()

        new_queries = _coerce_list_str(critic.get("new_queries"))
        new_queries = new_queries[: config.raar_max_query_variants_per_iter]

        if not new_queries and strategy == "decompose":
            decomp = _decompose_question(
                logger=logger,
                question=question,
                temperature=config.temperature,
            )
            new_queries = _coerce_list_str(decomp.get("query_variants"))
            new_queries = [q for q in new_queries if q.strip() and q.strip() != question]
            new_queries = new_queries[: config.raar_max_query_variants_per_iter]
            logger.log(
                component="raar",
                action="fallback_decompose_queries",
                stage="raar",
                data={"iter_idx": iter_idx, "queries": new_queries},
            )

        if not new_queries:
            missing = _coerce_list_str(critic.get("missing_info"))
            if missing:
                new_queries = missing[: config.raar_max_query_variants_per_iter]
            else:
                new_queries = [question]
            logger.log(
                component="raar",
                action="fallback_queries",
                stage="raar",
                data={"iter_idx": iter_idx, "queries": new_queries},
            )

        if not new_queries:
            raar_meta["stop_reason"] = "no_new_queries"
            break

        logger.log(
            component="raar",
            action="plan",
            stage="raar",
            data={"iter_idx": iter_idx, "strategy": strategy, "new_queries": new_queries},
        )

        top_k = int(config.retrieve_top_k or 0)
        if strategy == "increase_k":
            top_k = max(top_k, config.max_contexts_final) + 10

        extra_contexts, _errs = _retrieve_for_queries(
            logger=logger,
            queries=new_queries,
            top_k=top_k,
            strategy=f"raar_iter_{iter_idx}",
        )

        merged = _dedupe_contexts(contexts + extra_contexts)
        merged, _ = _maybe_rerank(
            logger=logger,
            query=question,
            contexts=merged,
            rerank_top_k=config.rerank_top_k,
        )
        merged = merged[: config.max_contexts_final]

        new_answer, _ = _generate_answer(
            logger=logger,
            question=question,
            contexts=merged,
            temperature=config.temperature,
            tag=f"raar_draft_{iter_idx}",
            use_rag=True,
        )

        if _should_stop_for_stagnation(
            answer,
            new_answer,
            config.raar_answer_stagnation_window,
        ):
            raar_meta["stop_reason"] = "answer_stagnation"
            contexts = merged
            answer = new_answer
            break

        contexts = merged
        answer = new_answer
    if "stop_reason" not in raar_meta:
        raar_meta["stop_reason"] = "max_iters_reached"

    raar_meta["best_iter_idx"] = best_iter_idx
    logger.log(
        component="raar",
        action="stop",
        stage="raar",
        data={"reason": raar_meta["stop_reason"], "best_iter_idx": best_iter_idx},
    )
    return answer, contexts, raar_meta


def run_pipeline(
    query: str,
    gold_answer: str,
    config: Optional[PipelineConfig] = None,
) -> Dict[str, Any]:
    """Run the configured pipeline and return results + trace."""
    cfg = config or make_config_rag()
    logger = TraceLogger()
    t0 = time.time()

    logger.log(
        component="pipeline",
        action="start",
        stage="control",
        data={"mode": cfg.mode, "config": cfg.__dict__},
    )

    contexts: Optional[List[Dict[str, Any]]] = None
    retrieval_errors: List[str] = []
    raar_meta: Optional[Dict[str, Any]] = None
    answer: str = ""

    if cfg.mode == "no_rag":
        answer, _ = _generate_answer(
            logger=logger,
            question=query,
            contexts=[],
            temperature=cfg.temperature,
            tag="no_rag_answer",
            use_rag=False,
        )
    else:
        if cfg.retrieve_top_k is None:
            raise ValueError("retrieve_top_k must be set for RAG modes.")

        if cfg.mode == "raar":
            decomp = _decompose_question(
                logger=logger,
                question=query,
                temperature=cfg.temperature,
            )
            qv = _coerce_list_str(decomp.get("query_variants")) or [query]
            variants = [query] + [q for q in qv if q != query]
            variants = variants[: max(1, cfg.raar_max_query_variants_per_iter + 1)]

            logger.log(
                component="raar",
                action="query_variants",
                stage="raar",
                data={"variants": variants, "hops": decomp.get("hops")},
            )

            base_contexts, retrieval_errors = _retrieve_for_queries(
                logger=logger,
                queries=variants,
                top_k=cfg.retrieve_top_k,
                strategy="base_question",
            )
            base_contexts, _ = _maybe_rerank(
                logger=logger,
                query=query,
                contexts=base_contexts,
                rerank_top_k=cfg.rerank_top_k,
            )
            answer, contexts, raar_meta = _raar_loop(
                logger=logger,
                question=query,
                base_contexts=base_contexts,
                config=cfg,
            )
        elif cfg.mode in {"rag", "rag_rerank"}:
            base_contexts, retrieval_errors = _retrieve_for_queries(
                logger=logger,
                queries=[query],
                top_k=cfg.retrieve_top_k,
                strategy="base_question",
            )
            if cfg.mode == "rag_rerank":
                base_contexts, _ = _maybe_rerank(
                    logger=logger,
                    query=query,
                    contexts=base_contexts,
                    rerank_top_k=cfg.rerank_top_k,
                )

            contexts = _dedupe_contexts(base_contexts)[: cfg.max_contexts_final]
            answer, _ = _generate_answer(
                logger=logger,
                question=query,
                contexts=contexts,
                temperature=cfg.temperature,
                tag=cfg.mode + "_answer",
                use_rag=True,
            )
        else:
            raise ValueError(f"Unknown mode: {cfg.mode}")

    eval_obj, eval_raw, eval_err = _evaluate_answer(
        logger=logger,
        question=query,
        model_answer=answer,
        gold_answer=gold_answer,
        contexts=contexts,
        temperature=cfg.temperature,
    )

    out: Dict[str, Any] = {
        "mode": cfg.mode,
        "question": query,
        "gold_answer": gold_answer,
        "answer": answer,
        "contexts": contexts,
        "retrieval_errors": retrieval_errors,
        "evaluation": eval_obj,
        "evaluation_raw": eval_raw,
        "trace": logger.events,
    }
    if eval_err:
        out["evaluation_error"] = eval_err
    if raar_meta is not None:
        out["raar_meta"] = raar_meta

    logger.log(
        component="pipeline",
        action="end",
        stage="control",
        data={"elapsed_sec": round(time.time() - t0, 3)},
    )
    out["trace"] = logger.events
    return out
