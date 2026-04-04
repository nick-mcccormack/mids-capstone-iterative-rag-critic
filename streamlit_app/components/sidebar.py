import streamlit as st

from data.doc_loader import get_formatted_results

def get_sidebar():
	"""Render sidebar filters and return the filtered results.

	Loads the formatted evaluation results, renders sidebar controls for
	question type, critic outcome, and accuracy thresholds, and returns the
	filtered DataFrame.

	Returns
	-------
	pandas.DataFrame
		Filtered formatted results based on the selected sidebar controls.
	"""
	with st.sidebar:
		st.space("small")

		formatted_results = get_formatted_results()

		decomposed = st.toggle("**Failed Critic Check**")
		if decomposed:
			formatted_results = formatted_results.loc[
				formatted_results["critic_outcome"]=="decompose"
			]

		st.divider()

		st.markdown("**Initial Accuracy - Human**")
		col1, col2 = st.columns(2)
		with col1:
			min_initial_answer_accuracy_human = st.number_input(
				"Min",
				min_value=0.0,
				max_value=1.0,
				value=0.0,
				step=0.25,
				format="%.2f",
				key="min_initial_accuracy_human_widget",
			)
		with col2:
			max_initial_answer_accuracy_human = st.number_input(
				"Max",
				min_value=min_initial_answer_accuracy_human,
				max_value=1.0,
				value=1.0,
				step=0.25,
				format="%.2f",
				key="max_initial_accuracy_human_widget",
			)

		formatted_results = formatted_results.loc[
			(formatted_results["initial_answer_accuracy_human"] >=
				min_initial_answer_accuracy_human)
			& (formatted_results["initial_answer_accuracy_human"] <=
				max_initial_answer_accuracy_human)
		]

		st.markdown("**Final Accuracy - Human**")
		col1, col2 = st.columns(2)
		with col1:
			min_final_answer_accuracy_human = st.number_input(
				"Min",
				min_value=0.0,
				max_value=1.0,
				value=0.0,
				step=0.25,
				format="%.2f",
				key="min_final_accuracy_human_widget",
			)
		with col2:
			max_final_answer_accuracy_human = st.number_input(
				"Max",
				min_value=min_final_answer_accuracy_human,
				max_value=1.0,
				value=1.0,
				step=0.25,
				format="%.2f",
				key="max_final_accuracy_human_widget",
			)

		formatted_results = formatted_results.loc[
			(formatted_results["final_answer_accuracy"] >=
				min_final_answer_accuracy_human)
			& (formatted_results["final_answer_accuracy"] <=
				max_final_answer_accuracy_human)
		]

		st.divider()

		st.markdown("**Initial Accuracy - LLM**")
		col1, col2 = st.columns(2)
		with col1:
			min_initial_answer_accuracy = st.number_input(
				"Min",
				min_value=0.0,
				max_value=1.0,
				value=0.0,
				step=0.25,
				format="%.2f",
				key="min_initial_accuracy_widget",
			)
		with col2:
			max_initial_answer_accuracy = st.number_input(
				"Max",
				min_value=min_initial_answer_accuracy,
				max_value=1.0,
				value=1.0,
				step=0.25,
				format="%.2f",
				key="max_initial_accuracy_widget",
			)

		formatted_results = formatted_results.loc[
			(formatted_results["initial_answer_accuracy"] >=
				min_initial_answer_accuracy)
			& (formatted_results["initial_answer_accuracy"] <=
				max_initial_answer_accuracy)
		]

		st.markdown("**Final Accuracy - LLM**")
		col1, col2 = st.columns(2)
		with col1:
			min_final_answer_accuracy = st.number_input(
				"Min",
				min_value=0.0,
				max_value=1.0,
				value=0.0,
				step=0.25,
				format="%.2f",
				key="min_final_accuracy_widget",
			)
		with col2:
			max_final_answer_accuracy = st.number_input(
				"Max",
				min_value=min_final_answer_accuracy,
				max_value=1.0,
				value=1.0,
				step=0.25,
				format="%.2f",
				key="max_final_accuracy_widget",
			)

		formatted_results = formatted_results.loc[
			(formatted_results["final_answer_accuracy"] >=
				min_final_answer_accuracy)
			& (formatted_results["final_answer_accuracy"] <=
				max_final_answer_accuracy)
		]

	return formatted_results
