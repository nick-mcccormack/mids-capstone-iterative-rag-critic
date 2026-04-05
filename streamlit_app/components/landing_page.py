import os
import base64
import mimetypes
from textwrap import dedent

import streamlit as st


WORKFLOW_PATH = os.path.join(os.getcwd(), "images", "workflow.jpg")


def _image_to_base64(path: str) -> tuple[str, str]:
	"""Convert image file to a base64 string and detect MIME type.

	Parameters
	----------
	path : str
		Path to the image file.

	Returns
	-------
	tuple[str, str]
		A tuple containing the MIME type and base64-encoded image content.
	"""
	mime_type, _ = mimetypes.guess_type(path)
	if mime_type is None:
		mime_type = "image/jpeg"

	with open(path, "rb") as file_obj:
		img_b64 = base64.b64encode(file_obj.read()).decode()

	return mime_type, img_b64


def render_landing_page() -> None:
	"""Render the landing page for the iterative RAG workflow.

	Returns
	-------
	None
		Render-only function. Outputs directly to Streamlit.
	"""
	if not os.path.exists(WORKFLOW_PATH):
		st.error(f"Image not found: {WORKFLOW_PATH}")
		return

	mime_type, img_b64 = _image_to_base64(WORKFLOW_PATH)

	st.html(
		dedent(
			"""
			<style>
			.landing-page {
				max-width: 1000px;
			}

			.landing-hero {
				margin-bottom: 1.5rem;
			}

			.landing-kicker {
				font-size: 0.88rem;
				font-weight: 700;
				letter-spacing: 0.08em;
				text-transform: uppercase;
				color: #6b7280;
				margin-bottom: 0.35rem;
			}

			.landing-title {
				font-size: 2.0rem;
				font-weight: 800;
				line-height: 1.1;
				color: #111827;
				margin-bottom: 0.55rem;
			}

			.landing-subtitle {
				font-size: 1.0rem;
				line-height: 1.55;
				color: #4b5563;
				max-width: 900px;
			}

			.landing-section {
				margin-top: 1.6rem;
				padding: 1.1rem 1.2rem;
				border: 1px solid #e5e7eb;
				border-radius: 14px;
				background: #ffffff;
				box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04);
			}

			.landing-section.problem {
				border-left: 6px solid #003262;
				background: #f0f6ff;
			}

			.landing-section.solution {
				border-left: 6px solid #5B2C83;
				background: #ede9fe;
			}

			.landing-section-label {
				font-size: 0.84rem;
				font-weight: 700;
				letter-spacing: 0.08em;
				text-transform: uppercase;
				color: #6b7280;
				margin-bottom: 0.35rem;
			}

			.landing-section-title {
				font-size: 1.25rem;
				font-weight: 800;
				line-height: 1.2;
				color: #111827;
				margin-bottom: 0.75rem;
			}

			.landing-highlight {
				font-size: 1.02rem;
				font-weight: 800;
				line-height: 1.45;
				color: #003262;
				margin-bottom: 0.9rem;
			}

			.landing-copy {
				font-size: 0.98rem;
				line-height: 1.65;
				color: #374151;
				margin: 0;
			}

			.landing-copy p {
				margin: 0 0 1rem 0;
			}

			.landing-copy p:last-child {
				margin-bottom: 0;
			}

			.landing-image-wrap {
				margin-top: 1.5rem;
				padding: 0.9rem;
				border-radius: 14px;
				background: #ffffff;
				box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04);
			}
			</style>
			"""
		)
	)

	st.html(
		dedent(
			f"""
<div class="landing-page">
	<div class="landing-hero">
		<div class="landing-kicker">Generate–Critique–Decompose</div>
		<div class="landing-title">
			Critic-Guided Decomposition for Multi-Hop RAG
		</div>
		<div class="landing-subtitle">
			A retrieval-augmented generation workflow that moves
			beyond single-pass answering by evaluating, refining,
			and regenerating responses when the initial answer is
			incomplete or insufficiently grounded.
		</div>
	</div>

	<div class="landing-section problem">
		<div class="landing-section-label">
			Problem &amp; Motivation
		</div>
		<div class="landing-section-title">
			Standard RAG is effective but limited by single-pass
			generation and lack of post-hoc validation
		</div>
		<div class="landing-copy">
			<p>
				Most RAG systems follow a retrieve → generate
				paradigm, producing an answer in a single forward
				pass. While effective for simple queries, this
				approach lacks mechanisms for post-hoc evaluation or
				targeted refinement after the initial generation. As
				a result, the system cannot determine whether an
				answer is complete, accurate, or sufficiently
				grounded in the retrieved contexts.
			</p>

			<p>
				These limitations are more pronounced for complex,
				multi-hop queries that require synthesizing
				information across multiple documents. Standard RAG
				often struggles with connecting evidence across
				documents, handling multi-step reasoning, and
				identifying answers that are incomplete or
				insufficiently grounded. Even when relevant
				documents are retrieved, the model may fail to
				integrate them correctly or omit necessary
				intermediate reasoning steps.
			</p>

			<p>
				Without an explicit feedback mechanism, errors in
				retrieval or reasoning propagate directly to the
				final answer. The system cannot identify when an
				answer is insufficient, nor can it trigger
				additional retrieval or refinement steps. This leads
				to missing context, incomplete answers, and
				unverified reasoning, reducing reliability—
				particularly on queries that require multi-hop
				reasoning or deeper evidence aggregation.
			</p>
		</div>
	</div>

	<div class="landing-section solution">
		<div class="landing-section-label">Our Solution</div>

		<div class="landing-section-title">
			Critic-Guided Iterative RAG with Conditional Decomposition
		</div>

		<div class="landing-highlight">
			Retrieve → Rerank → Generate → Critique →
			(Decompose + Regenerate)
		</div>

		<div class="landing-copy">
			<p>
				The system extends a standard RAG pipeline by introducing a
				critic-driven feedback loop on top of an initial retrieve →
				rerank → generate pass. As shown in the workflow, the process
				begins with initial retrieval and reranking, followed by
				generation of an initial answer using the retrieved contexts.
			</p>

			<p>
				An LLM-based critic then evaluates the answer for grounding
				(faithfulness to retrieved contexts) and completeness. If the
				answer passes the critic check, it is returned as the final
				answer.
			</p>

			<p>
				If the answer fails, the system enters the critic loop, where
				it performs decomposition and regeneration. The original query
				is broken into targeted sub-queries, additional evidence is
				retrieved, and a new answer is generated using the expanded
				context. This updated answer is then re-evaluated by the
				critic.
			</p>

			<p>
				This retrieve → generate → critique →
				(decompose + regenerate) loop continues for a bounded number
				of iterations (up to three rounds) or until the critic passes
				the answer.
			</p>

			<p>
				By introducing this structured feedback mechanism, the system
				moves beyond single-pass generation and enables targeted
				recovery from common RAG failure modes, including missing or
				incomplete context and errors in multi-hop reasoning. The
				result is a more reliable and grounded answer generation
				process, particularly for complex queries requiring
				multi-step reasoning.
			</p>
		</div>

		<div class="landing-image-wrap">
			<img
				src="data:{mime_type};base64,{img_b64}"
				style="
					width:100%;
					height:auto;
					border-radius:12px;
					display:block;
				"
			/>
		</div>
	</div>
</div>
			"""
		)
	)
