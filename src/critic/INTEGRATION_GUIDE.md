# Critic System Integration Guide

This guide explains how to integrate the Critic System with your existing Jupyter notebook RAG pipeline.

## Overview

The Critic System enhances RAG pipeline answers through iterative improvement:
1. Validates initial answers against quality criteria
2. Decomposes complex queries into sub-queries
3. Retrieves targeted context for each sub-query
4. Validates reasoning and intermediate facts
5. Iterates up to 2 times before returning final answer

## Quick Start

### 1. Import the Integration Module

```python
import sys
sys.path.append('/path/to/project/root')  # Adjust to your project root

from src.critic.integration import integrate_critic_with_rag
import logging

# Enable logging to see progress
logging.basicConfig(level=logging.INFO)
```

### 2. Process a Single Question

```python
# Assuming you have:
# - retriever: Your LangChain retriever
# - llm: Your LangChain LLM
# - your_rag_chain: Your existing RAG chain

question = "Where was the director of Inception born?"

# Get baseline answer from existing pipeline
baseline_answer = your_rag_chain.invoke(question)

# Enhance with critic system
final_answer = integrate_critic_with_rag(
    question=question,
    baseline_answer=baseline_answer,
    retriever=retriever,
    llm=llm,
    enable_logging=True
)

print(f"Baseline: {baseline_answer}")
print(f"Final: {final_answer}")
```

### 3. Batch Processing for Evaluation

```python
from src.critic.integration import process_questions_with_critic

# Process multiple questions
final_answers = process_questions_with_critic(
    questions=questions_list,
    baseline_answers=baseline_answers_list,
    retriever=retriever,
    llm=llm,
    enable_logging=False  # Disable verbose logging for batch
)
```

## Integration with Existing RAG Pipelines

### LangChain-based Pipelines

The integration module automatically handles LangChain components:

```python
# Works with LangChain retrievers
retriever = qdrant_vectorstore.as_retriever(search_kwargs={"k": 5})

# Works with LangChain LLMs
from langchain_community.llms import HuggingFacePipeline
llm = HuggingFacePipeline(pipeline=mistral_pipeline)

# Or with LangChain chat models
from langchain_cohere import ChatCohere
llm = ChatCohere(model="command-r-plus")

# Use directly
final_answer = integrate_critic_with_rag(
    question=question,
    baseline_answer=baseline_answer,
    retriever=retriever,
    llm=llm
)
```

### Custom Retrieval Functions

If you have a custom retrieval function:

```python
def my_custom_retriever(query: str):
    # Your custom retrieval logic
    # Must return list of documents
    return documents

# Use with custom adapter settings
final_answer = integrate_critic_with_rag(
    question=question,
    baseline_answer=baseline_answer,
    retriever=my_custom_retriever,
    llm=llm,
    retrieval_adapter_kwargs={
        "doc_text_field": "content",    # Field name for text
        "doc_source_field": "doc_id",   # Field name for source
        "doc_score_field": "relevance"  # Field name for score (optional)
    }
)
```

### Custom Generation Functions

If you have a custom generation function:

```python
def my_custom_generator(prompt: str):
    # Your custom generation logic
    return answer

# Use with custom prompt template
final_answer = integrate_critic_with_rag(
    question=question,
    baseline_answer=baseline_answer,
    retriever=retriever,
    llm=my_custom_generator,
    generation_adapter_kwargs={
        "prompt_template": "Q: {question}\nContext: {context}\nA:"
    }
)
```

## Integration with EnhancedRAGPipeline Class

If you're using the `EnhancedRAGPipeline` class from the baseline notebook:

```python
# Assuming you have:
# rag_pipeline = EnhancedRAGPipeline(vectorstore, retriever, llm_mistral, llm_cohere)

def run_with_critic(question, user_type="engineer", llm_choice="mistral", k=5):
    """Run question through baseline pipeline, then enhance with critic."""
    
    # Get baseline answer
    result = rag_pipeline.generate_with_details(question, user_type, llm_choice, k)
    baseline_answer = result['answer']
    
    # Select LLM
    llm = rag_pipeline.llm_mistral if llm_choice == "mistral" else rag_pipeline.llm_cohere
    
    # Get retriever
    retriever = rag_pipeline.get_retriever_with_params(k=k)
    
    # Enhance with critic
    final_answer = integrate_critic_with_rag(
        question=question,
        baseline_answer=baseline_answer,
        retriever=retriever,
        llm=llm
    )
    
    return {
        'baseline_answer': baseline_answer,
        'final_answer': final_answer,
        'changed': baseline_answer != final_answer
    }

# Use it
result = run_with_critic("Where was the director of Inception born?")
```

## Output Format Compatibility

The critic system maintains output format compatibility with the baseline RAG pipeline:

- **Input**: Accepts question (string) and baseline answer (string)
- **Output**: Returns final answer (string)
- **Format**: Same string format as baseline pipeline
- **Evaluation**: Compatible with HotpotQA official evaluation script

```python
# The output is a plain string, just like baseline
final_answer = integrate_critic_with_rag(...)
assert isinstance(final_answer, str)

# Can be used directly in evaluation
from hotpotqa_evaluation import evaluate
accuracy = evaluate(predictions=[final_answer], gold_answers=[gold_answer])
```

## Logging and Debugging

### Enable Detailed Logging

```python
import logging

# Set up detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Run with logging enabled
final_answer = integrate_critic_with_rag(
    question=question,
    baseline_answer=baseline_answer,
    retriever=retriever,
    llm=llm,
    enable_logging=True
)
```

### What Gets Logged

The critic system logs:
- Initial validation results (pass/fail, triggered rules)
- Query decomposition (sub-queries, reasoning plan)
- Retrieval results (number of documents per sub-query)
- Answer generation progress
- Reasoning validation results
- Iteration progress
- Final answer

### Example Log Output

```
2024-01-15 10:30:00 - src.critic.critic_system - INFO - Starting critic system for question: Where was the director of Inception born?
2024-01-15 10:30:00 - src.critic.critic_system - INFO - Initial answer: Christopher Nolan
2024-01-15 10:30:00 - src.critic.critic_system - INFO - Initial validation: passed=False, triggered_rule=length_exceeded
2024-01-15 10:30:01 - src.critic.critic_system - INFO - Starting critic iteration 1/2
2024-01-15 10:30:01 - src.critic.critic_system - INFO - Decomposing query into sub-queries
2024-01-15 10:30:02 - src.critic.critic_system - INFO - Decomposed into 2 sub-queries
2024-01-15 10:30:02 - src.critic.critic_system - INFO - Retrieving context for sub-queries
2024-01-15 10:30:03 - src.critic.critic_system - INFO - Retrieved 5 documents for: Who directed Inception?
2024-01-15 10:30:04 - src.critic.critic_system - INFO - Retrieved 5 documents for: Where was Christopher Nolan born?
2024-01-15 10:30:04 - src.critic.critic_system - INFO - Generating new answer with reasoning
2024-01-15 10:30:06 - src.critic.critic_system - INFO - Generated answer: London, England
2024-01-15 10:30:06 - src.critic.critic_system - INFO - Checking reasoning and facts
2024-01-15 10:30:07 - src.critic.critic_system - INFO - Reasoning check action: PASS
2024-01-15 10:30:07 - src.critic.critic_system - INFO - Re-validation: passed=True
2024-01-15 10:30:07 - src.critic.critic_system - INFO - Answer passed validation on iteration 1, returning final answer
```

## Evaluation on HotpotQA Dataset

### Process All Questions

```python
import json
from tqdm import tqdm

# Load HotpotQA data
with open('data/hotpotqa_stratified_300.json', 'r') as f:
    hotpotqa_data = json.load(f)

# Process each question
results = []
for item in tqdm(hotpotqa_data):
    question = item['question']
    gold_answer = item['answer']
    
    # Get baseline answer
    baseline_answer = your_rag_chain.invoke(question)
    
    # Enhance with critic
    final_answer = integrate_critic_with_rag(
        question=question,
        baseline_answer=baseline_answer,
        retriever=retriever,
        llm=llm,
        enable_logging=False  # Disable for batch processing
    )
    
    results.append({
        'question': question,
        'gold_answer': gold_answer,
        'baseline_answer': baseline_answer,
        'final_answer': final_answer
    })

# Save results
with open('critic_system_results.json', 'w') as f:
    json.dump(results, f, indent=2)
```

### Calculate Accuracy

```python
# Use HotpotQA official evaluation script
from hotpotqa_evaluation import evaluate

predictions = [r['final_answer'] for r in results]
gold_answers = [r['gold_answer'] for r in results]

accuracy = evaluate(predictions, gold_answers)
print(f"Critic System Accuracy: {accuracy:.2%}")

# Compare with baseline
baseline_predictions = [r['baseline_answer'] for r in results]
baseline_accuracy = evaluate(baseline_predictions, gold_answers)
print(f"Baseline Accuracy: {baseline_accuracy:.2%}")
print(f"Improvement: {(accuracy - baseline_accuracy):.2%}")
```

## Troubleshooting

### Issue: "retriever_or_function must be callable or have .invoke() method"

**Solution**: Make sure your retriever is either:
- A LangChain retriever with `.invoke()` method
- A callable function that takes a query string

```python
# Good: LangChain retriever
retriever = vectorstore.as_retriever()

# Good: Custom function
def my_retriever(query):
    return documents

# Bad: Not callable
retriever = "not a function"  # This will fail
```

### Issue: "Length mismatch" in batch processing

**Solution**: Ensure questions and baseline_answers lists have the same length:

```python
assert len(questions) == len(baseline_answers), "Lists must have same length"
```

### Issue: Empty or "I do not know." answers

**Solution**: Check that:
1. Your retriever is returning documents
2. Your LLM is generating responses
3. Enable logging to see where the issue occurs

```python
# Enable debug logging
logging.basicConfig(level=logging.DEBUG)
```

### Issue: Slow performance

**Solution**: 
1. Disable logging for batch processing: `enable_logging=False`
2. Reduce retrieval k value
3. Use faster LLM for sub-query processing

## API Reference

### `integrate_critic_with_rag()`

Main integration function.

**Parameters:**
- `question` (str): Original question to answer
- `baseline_answer` (str): Initial answer from baseline RAG pipeline
- `retriever` (Any): RAG pipeline's retriever (LangChain or custom)
- `llm` (Any): RAG pipeline's LLM (LangChain or custom)
- `retrieval_adapter_kwargs` (dict, optional): Kwargs for retrieval adapter
- `generation_adapter_kwargs` (dict, optional): Kwargs for generation adapter
- `enable_logging` (bool): Enable detailed logging (default: True)

**Returns:**
- `str`: Final answer after critic system processing

### `process_questions_with_critic()`

Batch processing function.

**Parameters:**
- `questions` (List[str]): List of questions
- `baseline_answers` (List[str]): List of baseline answers
- `retriever` (Any): RAG pipeline's retriever
- `llm` (Any): RAG pipeline's LLM
- `retrieval_adapter_kwargs` (dict, optional): Kwargs for retrieval adapter
- `generation_adapter_kwargs` (dict, optional): Kwargs for generation adapter
- `enable_logging` (bool): Enable detailed logging (default: False)

**Returns:**
- `List[str]`: List of final answers

### `create_retrieval_adapter()`

Create adapter for custom retrieval functions.

**Parameters:**
- `retriever_or_function` (Any): Retriever object or function
- `doc_text_field` (str): Field name for document text (default: "page_content")
- `doc_source_field` (str): Field name for document source (default: "metadata")
- `doc_score_field` (str, optional): Field name for relevance score

**Returns:**
- `Callable`: Adapted retrieval function

### `create_generation_adapter()`

Create adapter for custom generation functions.

**Parameters:**
- `llm_or_chain` (Any): LLM object or function
- `prompt_template` (str, optional): Custom prompt template with {question} and {context}

**Returns:**
- `Callable`: Adapted generation function

## Requirements

The integration satisfies the following requirements:

- **Requirement 6.1**: Accepts initial answer from baseline RAG pipeline
- **Requirement 6.2**: Integrates with existing retrieval and generation components
- **Requirement 6.3**: Outputs answers in same format as baseline pipeline
- **Requirement 6.4**: Supports evaluation using HotpotQA official evaluation script

## Next Steps

1. Copy example code from `notebook_integration_example.py` into your Jupyter notebook
2. Adjust paths and variable names to match your pipeline
3. Test on a single question first
4. Run batch evaluation on HotpotQA dataset
5. Compare accuracy with baseline

For more examples, see `notebook_integration_example.py`.
