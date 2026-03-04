# HotpotQA Evaluation Guide

This guide explains how to evaluate the Critic System on the stratified 300-question HotpotQA dataset.

**Requirements:** 7.1, 7.2, 7.3

## Overview

The evaluation system provides:
- **Automated evaluation** on the 300-question HotpotQA dataset
- **Accuracy metrics** comparing to the 60.17% baseline
- **Iteration tracking** to analyze critic loop behavior
- **Validation outcomes** to understand failure modes
- **Detailed results** for error analysis

## Quick Start

### Option 1: Using the Jupyter Notebook (Recommended)

The easiest way to run the evaluation is using the provided Jupyter notebook:

```bash
jupyter notebook Notebooks/hotpotqa_evaluation.ipynb
```

The notebook provides:
- Step-by-step evaluation workflow
- Interactive result analysis
- Visualization of results
- Small-scale testing before full evaluation

### Option 2: Using Python Script

For programmatic evaluation, use the evaluation script:

```python
from src.critic.evaluate_hotpotqa import evaluate_hotpotqa

# Your RAG pipeline components
from your_rag_pipeline import retriever, llm

# Run evaluation
metrics = evaluate_hotpotqa(
    dataset_path="Dataset/hotpot_eval_300.json",
    retriever=retriever,
    llm=llm,
    output_dir="evaluation_results"
)

print(f"Accuracy: {metrics.accuracy:.2f}%")
print(f"Improvement: {metrics.accuracy_improvement:+.2f}%")
```

### Option 3: Using the Example Script

See the example script for a complete workflow:

```bash
python src/critic/evaluation_example.py
```

Edit the script to add your RAG pipeline components, then uncomment the function you want to run.

## Evaluation Process

The evaluation follows these steps:

1. **Load Dataset**: Load the stratified 300-question HotpotQA dataset
2. **Process Questions**: For each question:
   - Get baseline answer (from provided dict or generate)
   - Process through Critic System
   - Compare final answer to gold answer
   - Track metrics (iterations, validation outcomes)
3. **Calculate Metrics**: Compute accuracy and compare to baseline
4. **Save Results**: Save detailed results and metrics summary

## Dataset

The evaluation uses the stratified 300-question HotpotQA dataset located at:
```
Dataset/hotpot_eval_300.json
```

Each question includes:
- `id`: Unique question identifier
- `question`: Question text
- `answer`: Ground truth answer
- `type`: Question type (bridge, comparison)
- `level`: Difficulty level (easy, medium, hard)
- `context`: Retrieved documents with titles and sentences

## Baseline Answers

You can provide baseline answers in two ways:

### 1. Pre-computed Baseline Answers

If you have baseline answers from a previous evaluation:

```python
import json

# Load baseline answers
with open("baseline_answers.json", 'r') as f:
    baseline_answers = json.load(f)

# baseline_answers format: {"question_id": "answer", ...}

metrics = evaluate_hotpotqa(
    dataset_path="Dataset/hotpot_eval_300.json",
    retriever=retriever,
    llm=llm,
    baseline_answers=baseline_answers,
    output_dir="evaluation_results"
)
```

### 2. Generate During Evaluation

If you don't provide baseline answers, they will be generated during evaluation using your RAG pipeline without critic enhancement.

## Output Files

The evaluation generates two files in the output directory:

### 1. Detailed Results (`detailed_results_TIMESTAMP.json`)

Contains full results for each question:

```json
[
  {
    "question_id": "5ac4fbe255429924173fb53b",
    "question": "Name one of the Judges...",
    "gold_answer": "Jeffrey Adam \"Duff\" Goldman",
    "baseline_answer": "Duff Goldman",
    "final_answer": "Jeffrey Adam \"Duff\" Goldman",
    "is_correct": true,
    "iteration_count": 1,
    "validation_history": ["length_exceeded"]
  },
  ...
]
```

### 2. Metrics Summary (`metrics_summary_TIMESTAMP.json`)

Contains overall evaluation metrics:

```json
{
  "evaluation_timestamp": "20240115_143022",
  "total_questions": 300,
  "correct_answers": 195,
  "accuracy": 65.00,
  "baseline_accuracy": 60.17,
  "accuracy_improvement": 4.83,
  "avg_iteration_count": 1.2,
  "validation_outcomes": {
    "length_exceeded": 120,
    "hedging_detected": 80,
    "multiple_entities": 40,
    "unknown_answer": 60
  },
  "processing_time_seconds": 1800.5,
  "avg_time_per_question": 6.0
}
```

## Metrics Explained

### Accuracy Metrics

- **Total Questions**: Number of questions processed (should be 300)
- **Correct Answers**: Number of questions answered correctly
- **Accuracy**: Percentage of correct answers
- **Baseline Accuracy**: Baseline system accuracy (60.17%)
- **Accuracy Improvement**: Difference from baseline (positive = improvement)

### Performance Metrics

- **Processing Time**: Total time to process all questions
- **Avg Time per Question**: Average processing time per question
- **Avg Iteration Count**: Average number of critic loop iterations

### Validation Outcomes

Count of each validation rule that triggered the critic loop:
- `length_exceeded`: Answer exceeded 90 words
- `hedging_detected`: Answer contained hedging phrases
- `multiple_entities`: Answer contained multiple entities with conjunctions
- `unknown_answer`: Answer was "I do not know."

## Answer Matching

The evaluation uses exact match scoring with normalization:

1. Convert to lowercase
2. Remove articles (a, an, the)
3. Remove punctuation
4. Remove extra whitespace
5. Compare normalized strings

This follows the standard HotpotQA evaluation protocol.

## Small-Scale Testing

Before running the full 300-question evaluation, test on a small subset:

```python
from src.critic.evaluation_example import run_small_scale_test

# Test on 10 questions
run_small_scale_test()
```

This is useful for:
- Verifying the evaluation pipeline works
- Quick iteration during development
- Debugging issues with specific questions

## Analyzing Results

### Load and Analyze Results

```python
import json
from pathlib import Path

# Load detailed results
results_dir = Path("evaluation_results")
results_file = sorted(results_dir.glob("detailed_results_*.json"))[-1]

with open(results_file, 'r') as f:
    results = json.load(f)

# Find incorrect answers
incorrect = [r for r in results if not r['is_correct']]

print(f"Incorrect answers: {len(incorrect)}")

# Analyze by validation outcome
from collections import Counter
outcomes = Counter()
for r in incorrect:
    if r['validation_history']:
        outcomes[r['validation_history'][0]] += 1

print("Validation outcomes for incorrect answers:")
for outcome, count in outcomes.most_common():
    print(f"  {outcome}: {count}")
```

### Common Analysis Tasks

1. **Identify failure patterns**: Group incorrect answers by validation outcome
2. **Compare baseline vs final**: See which answers improved/regressed
3. **Analyze iteration counts**: Understand critic loop behavior
4. **Review specific questions**: Deep dive into challenging questions

## Troubleshooting

### Issue: Low Accuracy

**Possible causes:**
- Retriever not returning relevant documents
- LLM generating poor quality answers
- Critic system parameters need tuning

**Solutions:**
- Review detailed results to identify failure patterns
- Test retriever and LLM independently
- Adjust validation rules or decomposition prompts

### Issue: Slow Processing

**Possible causes:**
- LLM API rate limits
- Slow retrieval
- Too many critic iterations

**Solutions:**
- Use faster LLM model
- Optimize retrieval (caching, indexing)
- Reduce max iterations (currently 2)

### Issue: Errors During Evaluation

**Possible causes:**
- Missing dependencies
- Invalid RAG pipeline components
- Dataset file not found

**Solutions:**
- Check error logs for specific issues
- Verify retriever and llm are properly configured
- Ensure dataset file exists at correct path

## Integration with Existing Pipeline

The evaluation script integrates with your existing RAG pipeline through adapters:

```python
from src.critic.integration import create_retrieval_adapter, create_generation_adapter

# Create adapters for your components
retrieval_function = create_retrieval_adapter(
    your_retriever,
    doc_text_field="page_content",
    doc_source_field="metadata"
)

generation_function = create_generation_adapter(
    your_llm,
    prompt_template="Answer: {question}\nContext: {context}\nAnswer:"
)

# Use in evaluation
metrics = evaluate_hotpotqa(
    dataset_path="Dataset/hotpot_eval_300.json",
    retriever=your_retriever,
    llm=your_llm,
    retrieval_adapter_kwargs={"doc_text_field": "page_content"},
    generation_adapter_kwargs={"prompt_template": custom_template}
)
```

See `INTEGRATION_GUIDE.md` for more details on integration.

## Next Steps

After running the evaluation:

1. **Review Results**: Analyze detailed results to understand performance
2. **Identify Patterns**: Look for common failure modes
3. **Tune Parameters**: Adjust critic system parameters based on findings
4. **Iterate**: Re-run evaluation to measure improvements
5. **Compare**: Track accuracy improvements over time

## References

- **Requirements**: See `.kiro/specs/llm-critic-enhancement/requirements.md`
- **Design**: See `.kiro/specs/llm-critic-enhancement/design.md`
- **Integration**: See `src/critic/INTEGRATION_GUIDE.md`
- **HotpotQA**: https://hotpotqa.github.io/

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the integration guide
3. Examine the example scripts and notebook
4. Check the detailed error logs
