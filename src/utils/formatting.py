import streamlit as st

def render_contexts(contexts: list[dict], show_rerank: bool = False) -> None:
	"""Render retrieval contexts as formatted Markdown blocks in a Streamlit app.

	This helper formats each context (title, scores, and text) into a readable
	Markdown "card" and renders the full list via ``st.markdown``. Contexts are
	numbered starting at 1 and separated by a horizontal rule (``---``).

	If ``contexts`` is empty, an informational message is shown and no Markdown
	is rendered.

	Parameters
	----------
	contexts
		Sequence of context records. Each record is expected to be a mapping that
		may contain the following keys:

		- ``"title"`` : str, optional
		Human-readable source title.
		- ``"text"`` : str, optional
		Retrieved passage text.
		- ``"score"`` : float, optional
		Retrieval score for the passage (displayed as ``Retrieval``).
		- ``"rerank_score"`` : float, optional
		Reranker score for the passage (displayed as ``Rerank`` when enabled).

		Missing fields default to empty strings for text/title and ``0.0`` for
		scores.
	show_rerank
		Whether to display the reranker score (``"rerank_score"``) alongside the
		retrieval score. Defaults to ``False``.

	Returns
	-------
	None
	"""
	if not contexts:
		st.info("No contexts found.")
		return

	blocks: list[str] = []
	for i, ctx in enumerate(contexts, start=1):
		title = (ctx.get("title") or "").strip()
		text = (ctx.get("text") or "").strip()
		retrieval_score = float(ctx.get("score") or 0.0)
		rerank_score = float(ctx.get("rerank_score") or 0.0)

		title_line = f"**{i}. {title if title else 'Untitled'}**"
		scores = (
			f"Retrieval: `{retrieval_score:.4f}`"
			+ (f" · Rerank: `{rerank_score:.4f}`" if show_rerank else "")
		)

		blocks.append(
			f"{title_line}\n\n"
			f"{scores}\n\n"
			f"{text if text else '_No text provided._'}"
		)

	st.markdown("\n\n---\n\n".join(blocks))
	