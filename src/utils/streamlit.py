"""Streamlit formatting helpers."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import hashlib
import pandas as pd
import streamlit as st

from src.utils.helpers import (
    _sanitize,
    _extract_retrieval_rows,
    _extract_rerank_info,
    _extract_generation_events,
    _extract_judge_metrics,
    _extract_events,
    _extract_answer_by_tag
)

from src.rag.pipeline import (
    make_config_no_rag,
    make_config_raar,
    make_config_rag,
    make_config_rag_rerank,
    run_pipeline,
)

def center_header(text: str) -> None:
    st.markdown(
        (
            "<div style='text-align:center; padding-top:10px;'>"
            f"<h1 style='margin-bottom:0;'>{text}</h1>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def run_all_pipelines(query: str, gold_answer: str):
    results: Dict[str, Dict[str, Any]] = {}

    with st.status("Running Pipelines..."):
        results["no_rag"] = run_pipeline(
            query=query,
            gold_answer=gold_answer,
            config=make_config_no_rag(),
        )
        st.success("No RAG Completed")
        results["rag"] = run_pipeline(
            query=query,
            gold_answer=gold_answer,
            config=make_config_rag(retrieve_top_k=5),
        )
        st.success("Retrieval-Only Completed")
        results["rag_rerank"] = run_pipeline(
            query=query,
            gold_answer=gold_answer,
            config=make_config_rag_rerank(retrieve_top_k=30, rerank_top_k=5),
        )
        st.success("Retrieval-Rerank Completed")
        results["raar"] = run_pipeline(
            query=query,
            gold_answer=gold_answer,
            config=make_config_raar(retrieve_top_k=30, rerank_top_k=5, raar_max_iters=3),
        )
        st.success("RAAR Completed")
    st.session_state["run_results"] = results


def render_contexts(
    contexts: List[Dict[str, Any]],
    show_rerank: bool = False,
    show_retrieval_meta: bool = False,
) -> None:
    """Render retrieval contexts as formatted Markdown blocks."""
    if not contexts:
        st.info("No contexts found.")
        return

    blocks: List[str] = []
    for i, ctx in enumerate(contexts, start=1):
        title = (ctx.get("title") or "").strip() or "Untitled"
        text = (ctx.get("text") or "").strip() or "_No text provided._"
        retrieval_score = float(ctx.get("score") or 0.0)
        rerank_score = float(ctx.get("rerank_score") or 0.0)

        meta_parts: List[str] = [f"Retrieval: `{retrieval_score:.4f}`"]
        if show_rerank:
            meta_parts.append(f"Rerank: `{rerank_score:.4f}`")

        if show_retrieval_meta:
            rq = (ctx.get("retrieval_query") or "").strip()
            rs = (ctx.get("retrieval_strategy") or "").strip()
            if rq:
                meta_parts.append(f"Query: `{rq}`")
            if rs:
                meta_parts.append(f"Strategy: `{rs}`")

        blocks.append(
            f"**{i}. {title}**\n\n"
            f"{' · '.join(meta_parts)}\n\n"
            f"{text}"
        )

    st.markdown("\n\n---\n\n".join(blocks))


def render_raw_trace(trace: List[Dict[str, Any]], mode: str) -> None:
    if not trace:
        st.write("No trace events were recorded.")
        return

    col_a, col_b = st.columns([1, 2])
    with col_a:
        show_prompts = st.checkbox("Show prompts", value=False, key=f"prompts_checkbox_{mode}")
    with col_b:
        max_chars = st.slider("Max chars per field", 200, 20000, 4000, 200, key=f"max_chars_slider_{mode}")

    df = pd.DataFrame(
        [
            {
                "idx": e.get("idx"),
                "stage": e.get("stage"),
                "component": e.get("component"),
                "action": e.get("action"),
                "tag": (e.get("data") or {}).get("tag"),
            }
            for e in trace
        ]
    )
    st.dataframe(df, hide_index=True, use_container_width=True, key=f"trace_df_{mode}")

    idxs = [int(x) for x in df["idx"].dropna().tolist()] if "idx" in df else []
    if not idxs:
        return

    selected = st.selectbox("Inspect event", options=idxs, index=0)
    for e in trace:
        if e.get("idx") == selected:
            st.json(_sanitize(e, max_chars=max_chars, show_prompts=show_prompts))
            break


def render_story_chapter(title: str, narrative_md: str) -> None:
    st.markdown(f"### {title}")
    st.markdown(narrative_md)


def render_storyboard(steps: List[Dict[str, Any]]) -> None:
    """Render a semantic storyboard.

    Each step dict may contain:
    - title (str): chapter title
    - narrative (str): markdown narrative
    - df (pd.DataFrame): optional table
    - json (Any): optional JSON object
    - expander_title (str): optional details toggle label
    - expander_body (Any): optional details content

    Notes
    -----
    Streamlit does not allow expanders nested inside other expanders. The
    storyboard is typically rendered inside a parent expander, so we use a
    checkbox-style toggle for step details instead of nested expanders.

    Parameters
    ----------
    steps:
        Ordered storyboard steps.

    Returns
    -------
    None
        Renders content to the Streamlit app.
    """
    for i, step in enumerate(steps, start=1):
        title = step.get("title") or f"Step {i}"
        narrative = step.get("narrative") or ""
        render_story_chapter(f"{i}. {title}", narrative)

        df = step.get("df")
        if isinstance(df, pd.DataFrame) and not df.empty:
            st.dataframe(df, hide_index=True, use_container_width=True)

        json_obj = step.get("json")
        if json_obj is not None:
            st.json(json_obj)

        exp_title = step.get("expander_title")
        exp_body = step.get("expander_body")
        if exp_title and exp_body is not None:
            key_seed = f"{i}|{title}|{exp_title}"
            key = "sb_details_" + hashlib.sha256(key_seed.encode("utf-8")).hexdigest()[:10]
            show = st.checkbox(exp_title, value=False, key=key)
            if show:
                if isinstance(exp_body, pd.DataFrame):
                    st.dataframe(exp_body, hide_index=True, use_container_width=True)
                elif isinstance(exp_body, (dict, list)):
                    st.json(exp_body)
                else:
                    st.write(exp_body)

        st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)


def _build_storyboard_rag(
    resp: Dict[str, Any],
    show_rerank: bool,
) -> List[Dict[str, Any]]:
    trace = resp.get("trace") or []
    retrieval_rows = _extract_retrieval_rows(trace)
    retrieval_df = pd.DataFrame(retrieval_rows)

    steps: List[Dict[str, Any]] = []

    steps.append(
        {
            "title": "Retrieve evidence",
            "narrative": (
                "We embed the question and retrieve candidate passages from Qdrant. "
                "This creates the evidence pool used for generation."
            ),
            "df": retrieval_df,
        }
    )

    if show_rerank:
        rr = _extract_rerank_info(trace) or {}
        steps.append(
            {
                "title": "Rerank evidence",
                "narrative": (
                    "We rerank the retrieved candidates with Voyage to prioritize passages "
                    "most relevant to the question."
                ),
                "json": {"kept": rr.get("kept"), "preview": rr.get("preview")},
            }
        )

    final_answer = resp.get("answer", "")
    steps.append(
        {
            "title": "Answer from evidence",
            "narrative": (
                "We ask the model to answer using only the selected contexts. "
                "If it cannot cite evidence, it must reply exactly: `I do not know.`\n\n"
                f"**Final answer:**\n\n{final_answer}"
            ),
            "expander_title": "Draft answers (from trace)",
            "expander_body": pd.DataFrame(
                [
                    {
                        "tag": (e.get("data") or {}).get("tag"),
                        "num_contexts": (e.get("data") or {}).get("num_contexts"),
                        "answer": (e.get("data") or {}).get("answer"),
                    }
                    for e in _extract_generation_events(trace)
                ]
            ),
        }
    )

    metrics = resp.get("evaluation") or _extract_judge_metrics(trace) or {}
    steps.append(
        {
            "title": "Judge scoring",
            "narrative": (
                "We evaluate answer quality (precision/completeness) and, when contexts "
                "exist, context quality (precision/completeness/faithfulness)."
            ),
            "json": metrics,
        }
    )

    return steps


def _build_storyboard_raar(resp: Dict[str, Any]) -> List[Dict[str, Any]]:
    trace = resp.get("trace") or []
    raar_meta = resp.get("raar_meta") or {}

    steps: List[Dict[str, Any]] = []

    variants_ev = _extract_events(trace, component="raar", action="query_variants")
    variants_data = (variants_ev[-1].get("data") or {}) if variants_ev else {}

    steps.append(
        {
            "title": "Plan the search (decompose + query variants)",
            "narrative": (
                "RAAR starts by decomposing the question and proposing multiple retrieval "
                "queries that target the required hops."
            ),
            "json": {
                "hops": variants_data.get("hops"),
                "query_variants": variants_data.get("variants"),
            },
        }
    )

    retrieval_rows = _extract_retrieval_rows(trace)
    retrieval_df = pd.DataFrame(retrieval_rows)

    if retrieval_df.empty:
        base_df = pd.DataFrame()
    else:
        base_df = retrieval_df[retrieval_df["strategy"] == "base_question"]
    steps.append(
        {
            "title": "Retrieve initial evidence",
            "narrative": (
                "We retrieve evidence for the query variants, merge results, and dedupe to "
                "form the starting evidence set."
            ),
            "df": base_df,
        }
    )

    rr = _extract_rerank_info(trace) or {}
    steps.append(
        {
            "title": "Rerank initial evidence",
            "narrative": (
                "We rerank the merged evidence pool so the initial draft uses the strongest "
                "passages first."
            ),
            "json": {"kept": rr.get("kept"), "preview": rr.get("preview")},
        }
    )

    draft0 = _extract_answer_by_tag(trace, "raar_draft_0") or ""
    steps.append(
        {
            "title": "Draft 0 (answer from initial evidence)",
            "narrative": (
                "We generate an initial answer strictly from the selected contexts. "
                "This is the starting point for adversarial critique.\n\n"
                f"**Draft 0:**\n\n{draft0}"
            ),
        }
    )

    iters_raw = raar_meta.get("iterations")
    iters = iters_raw if isinstance(iters_raw, list) else []
    plans = _extract_events(trace, component="raar", action="plan")
    plan_by_iter: Dict[int, Dict[str, Any]] = {}
    for p in plans:
        d = p.get("data") if isinstance(p.get("data"), dict) else {}
        it = d.get("iter_idx")
        if isinstance(it, int):
            plan_by_iter[it] = d

    for it in iters:
        iter_idx = it.get("iter_idx")
        if not isinstance(iter_idx, int):
            continue

        critic_raw = it.get("critic")
        critic = critic_raw if isinstance(critic_raw, dict) else {}
        scores_raw = critic.get("scores")
        scores = scores_raw if isinstance(scores_raw, dict) else {}
        decision = critic.get("decision")
        strategy = critic.get("suggested_strategy")

        plan = plan_by_iter.get(iter_idx, {})
        new_q_raw = plan.get("new_queries")
        new_queries = new_q_raw if isinstance(new_q_raw, list) else []

        target_key = f"raar_iter_{iter_idx}"
        targ_df = (
            retrieval_df[retrieval_df["strategy"] == target_key]
            if not retrieval_df.empty
            else pd.DataFrame()
        )

        revised = _extract_answer_by_tag(trace, f"raar_draft_{iter_idx}") or ""

        steps.append(
            {
                "title": f"Iteration {iter_idx}: critique -> targeted retrieval -> revision",
                "narrative": (
                    "The critic checks whether the current draft is grounded and complete. "
                    "If not, it proposes *targeted* retrieval queries "
                    "to fill evidence gaps.\n\n"
                    f"- Critic decision: **{decision}**\n"
                    f"- Suggested strategy: **{strategy}**\n"
                    f"- Overall confidence: **{scores.get('overall_confidence')}**\n"
                    f"- Groundedness: **{scores.get('groundedness')}**\n"
                    f"- Completeness: **{scores.get('completeness')}**\n\n"
                    "**Targeted queries:**\n"
                    f"{new_queries if new_queries else 'None'}"
                ),
                "df": targ_df,
                "expander_title": "Critic JSON (full)",
                "expander_body": critic,
            }
        )

        if revised:
            steps.append(
                {
                    "title": f"Iteration {iter_idx}: revised draft",
                    "narrative": (
                        "Using the augmented evidence set, we generate a revised answer.\n\n"
                        f"**Draft {iter_idx}:**\n\n{revised}"
                    ),
                }
            )

    stop_reason = raar_meta.get("stop_reason")
    best_iter = raar_meta.get("best_iter_idx")
    steps.append(
        {
            "title": "Stop and finalize",
            "narrative": (
                "RAAR stops when one of the configured criteria fires. We keep the final "
                "draft from the stopping point.\n\n"
                f"- Stop reason: **{stop_reason}**\n"
                f"- Best iteration (internal confidence): **{best_iter}**"
            ),
        }
    )

    final_answer = resp.get("answer", "")
    steps.append(
        {
            "title": "Final answer",
            "narrative": f"**Final answer:**\n\n{final_answer}",
        }
    )

    metrics = resp.get("evaluation") or _extract_judge_metrics(trace) or {}
    steps.append(
        {
            "title": "Judge scoring",
            "narrative": "We score the final RAAR output using the judge rubric.",
            "json": metrics,
        }
    )

    return steps


def render_method_tab(title: str, resp: Dict[str, Any], mode: str) -> None:
    center_header(title)

    with st.expander("Question", expanded=True):
        st.markdown(f"**Query:** {resp.get('question', '')}")
        st.markdown(f"**Gold Answer:** {resp.get('gold_answer', '')}")

    with st.expander("Semantic storyboard", expanded=True):
        if mode == "rag":
            steps = _build_storyboard_rag(resp, show_rerank=False)
            render_storyboard(steps)
        elif mode == "rag_rerank":
            steps = _build_storyboard_rag(resp, show_rerank=True)
            render_storyboard(steps)
        elif mode == "raar":
            steps = _build_storyboard_raar(resp)
            render_storyboard(steps)
        else:
            st.write("Narrative walkthrough is available for RAG methods only.")

    with st.expander("Evidence used (final contexts)", expanded=True):
        render_contexts(
            resp.get("contexts") or [],
            show_rerank=(mode in {"rag_rerank", "raar"}),
            show_retrieval_meta=True,
        )

    with st.expander("Raw trace (debug)", expanded=False):
        render_raw_trace(resp.get("trace") or [], mode=mode)
