from typing import Any, List, Dict, Optional

def _sanitize(obj: Any, max_chars: int, show_prompts: bool) -> Any:
    if isinstance(obj, dict):
        out: Dict[str, Any] = {}
        for k, v in obj.items():
            if not show_prompts and k in {"system_prompt", "user_prompt"}:
                out[k] = "(hidden)"
                continue
            out[k] = _sanitize(v, max_chars=max_chars, show_prompts=show_prompts)
        return out

    if isinstance(obj, list):
        return [_sanitize(x, max_chars=max_chars, show_prompts=show_prompts) for x in obj]

    if isinstance(obj, str):
        if len(obj) <= max_chars:
            return obj
        return obj[: max_chars - 1] + "…"

    return obj

def _extract_events(
    trace: List[Dict[str, Any]],
    component: Optional[str] = None,
    action: Optional[str] = None,
    stage: Optional[str] = None,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for e in trace:
        if component and e.get("component") != component:
            continue
        if action and e.get("action") != action:
            continue
        if stage and e.get("stage") != stage:
            continue
        out.append(e)
    return out


def _extract_retrieval_rows(trace: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for e in _extract_events(trace, component="retriever", action="retrieve_success"):
        d = e.get("data") or {}
        rows.append(
            {
                "strategy": d.get("strategy"),
                "query": d.get("query"),
                "top_k": d.get("top_k"),
                "num_hits": d.get("num_hits"),
            }
        )

    for e in _extract_events(trace, component="retriever", action="retrieve_error"):
        d = e.get("data") or {}
        rows.append(
            {
                "strategy": d.get("strategy"),
                "query": d.get("query"),
                "top_k": d.get("top_k"),
                "num_hits": 0,
                "error": d.get("error"),
            }
        )

    return rows


def _extract_rerank_info(trace: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    evs = _extract_events(trace, component="reranker", action="rerank_success")
    if not evs:
        return None
    return evs[-1].get("data") or None


def _extract_generation_events(trace: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return _extract_events(trace, component="generator", action="answer_generated")


def _extract_answer_by_tag(trace: List[Dict[str, Any]], tag: str) -> Optional[str]:
    for e in _extract_generation_events(trace):
        d = e.get("data") if isinstance(e.get("data"), dict) else {}
        if d.get("tag") == tag:
            return d.get("answer")
    return None


def _extract_judge_metrics(trace: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    evs = _extract_events(trace, component="judge", action="eval_success")
    if not evs:
        return None
    d = evs[-1].get("data") if isinstance(evs[-1].get("data"), dict) else {}
    metrics = d.get("metrics") if isinstance(d.get("metrics"), dict) else None
    return metrics
