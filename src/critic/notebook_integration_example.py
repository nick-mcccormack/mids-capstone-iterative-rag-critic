"""
Example code for integrating the Critic System into a Jupyter notebook RAG pipeline.

This file contains example code snippets that can be copied into Jupyter notebook cells
to integrate the critic system with an existing RAG pipeline.

Requirements: 6.1, 6.2, 6.3
"""

# ============================================================================
# CELL 1: Import the critic system integration module
# ============================================================================

"""
# Import the critic system integration
import sys
sys.path.append('/path/to/project/root')  # Adjust path as needed

from src.critic.integration import integrate_critic_with_rag, process_questions_with_critic
import logging

# Enable logging to see critic system progress
logging.basicConfig(level=logging.INFO)
"""

# ============================================================================
# CELL 2: Single question example with LangChain components
# ============================================================================

"""
# Example: Process a single question with the critic system
# Assumes you have already set up:
# - retriever: Your LangChain retriever (e.g., qdrant_vectorstore.as_retriever())
# - llm: Your LangChain LLM (e.g., mistral_llm_lc or cohere_chat_model)

question = "Where was the director of Inception born?"

# Get baseline answer from your existing RAG pipeline
baseline_answer = your_rag_chain.invoke(question)

# Process with critic system
final_answer = integrate_critic_with_rag(
    question=question,
    baseline_answer=baseline_answer,
    retriever=retriever,  # Your existing retriever
    llm=llm,              # Your existing LLM
    enable_logging=True   # Set to False to reduce output
)

print(f"Baseline answer: {baseline_answer}")
print(f"Final answer: {final_answer}")
"""

# ============================================================================
# CELL 3: Batch processing example for evaluation
# ============================================================================

"""
# Example: Process multiple questions for evaluation
# Useful for running on the HotpotQA dataset

# Load your questions and get baseline answers
questions = [
    "Where was the director of Inception born?",
    "What year was the company that created the iPhone founded?",
    # ... more questions
]

# Get baseline answers from your existing RAG pipeline
baseline_answers = []
for question in questions:
    answer = your_rag_chain.invoke(question)
    baseline_answers.append(answer)

# Process all questions with critic system
final_answers = process_questions_with_critic(
    questions=questions,
    baseline_answers=baseline_answers,
    retriever=retriever,
    llm=llm,
    enable_logging=False  # Disable verbose logging for batch processing
)

# Compare results
for i, (q, baseline, final) in enumerate(zip(questions, baseline_answers, final_answers)):
    print(f"\\nQuestion {i+1}: {q}")
    print(f"Baseline: {baseline}")
    print(f"Final: {final}")
"""

# ============================================================================
# CELL 4: Custom adapter configuration example
# ============================================================================

"""
# Example: Use custom adapter settings if your RAG pipeline uses different field names

# If your documents use different field names
final_answer = integrate_critic_with_rag(
    question=question,
    baseline_answer=baseline_answer,
    retriever=retriever,
    llm=llm,
    retrieval_adapter_kwargs={
        "doc_text_field": "content",      # If your docs use 'content' instead of 'page_content'
        "doc_source_field": "doc_id",     # If your docs use 'doc_id' instead of 'metadata'
        "doc_score_field": "relevance"    # If your docs have a 'relevance' score field
    },
    generation_adapter_kwargs={
        "prompt_template": "Q: {question}\\nContext: {context}\\nA:"  # Custom prompt format
    }
)
"""

# ============================================================================
# CELL 5: Integration with EnhancedRAGPipeline class (from baseline notebook)
# ============================================================================

"""
# Example: Integrate with the EnhancedRAGPipeline class from the baseline notebook

# Assuming you have an EnhancedRAGPipeline instance
# rag_pipeline = EnhancedRAGPipeline(vectorstore, retriever, llm_mistral, llm_cohere)

def run_with_critic(question, user_type="engineer", llm_choice="mistral", k=5):
    '''
    Run a question through the baseline RAG pipeline, then enhance with critic system.
    
    Args:
        question: Question to answer
        user_type: "engineer" or "marketing"
        llm_choice: "mistral" or "cohere"
        k: Number of documents to retrieve
    
    Returns:
        dict with baseline_answer, final_answer, and metadata
    '''
    # Get baseline answer from existing pipeline
    result = rag_pipeline.generate_with_details(question, user_type, llm_choice, k)
    baseline_answer = result['answer']
    
    # Select the appropriate LLM
    llm = rag_pipeline.llm_mistral if llm_choice == "mistral" else rag_pipeline.llm_cohere
    
    # Get retriever with specified k
    retriever = rag_pipeline.get_retriever_with_params(k=k)
    
    # Process with critic system
    final_answer = integrate_critic_with_rag(
        question=question,
        baseline_answer=baseline_answer,
        retriever=retriever,
        llm=llm,
        enable_logging=True
    )
    
    return {
        'question': question,
        'baseline_answer': baseline_answer,
        'final_answer': final_answer,
        'user_type': user_type,
        'llm_choice': llm_choice,
        'k': k,
        'retrieved_docs': result.get('retrieved_docs', [])
    }

# Use it
result = run_with_critic(
    "Where was the director of Inception born?",
    user_type="engineer",
    llm_choice="mistral",
    k=5
)

print(f"Baseline: {result['baseline_answer']}")
print(f"Final: {result['final_answer']}")
"""

# ============================================================================
# CELL 6: Evaluation on HotpotQA dataset
# ============================================================================

"""
# Example: Run evaluation on the HotpotQA dataset with critic system

import json
from tqdm import tqdm

# Load HotpotQA questions (adjust path as needed)
with open('data/hotpotqa_stratified_300.json', 'r') as f:
    hotpotqa_data = json.load(f)

# Process each question
results = []
for item in tqdm(hotpotqa_data):
    question = item['question']
    gold_answer = item['answer']
    
    # Get baseline answer
    baseline_answer = your_rag_chain.invoke(question)
    
    # Process with critic system
    final_answer = integrate_critic_with_rag(
        question=question,
        baseline_answer=baseline_answer,
        retriever=retriever,
        llm=llm,
        enable_logging=False
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

# Calculate accuracy (using HotpotQA evaluation script)
# ... evaluation code here ...
"""

# ============================================================================
# CELL 7: Debugging and analysis
# ============================================================================

"""
# Example: Enable detailed logging for debugging

import logging

# Set up detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Run with detailed logging
final_answer = integrate_critic_with_rag(
    question="Where was the director of Inception born?",
    baseline_answer="Christopher Nolan",
    retriever=retriever,
    llm=llm,
    enable_logging=True
)

# The logs will show:
# - Initial validation results
# - Query decomposition into sub-queries
# - Retrieval for each sub-query
# - Answer generation with reasoning
# - Reasoning validation
# - Iteration progress
# - Final answer
"""

# ============================================================================
# CELL 8: Comparison function for analysis
# ============================================================================

"""
# Example: Compare baseline vs critic system answers

def compare_answers(question, retriever, llm, rag_chain):
    '''
    Compare baseline RAG answer with critic-enhanced answer.
    
    Returns detailed comparison for analysis.
    '''
    # Get baseline answer
    baseline_answer = rag_chain.invoke(question)
    
    # Get critic-enhanced answer
    final_answer = integrate_critic_with_rag(
        question=question,
        baseline_answer=baseline_answer,
        retriever=retriever,
        llm=llm,
        enable_logging=True
    )
    
    # Print comparison
    print("=" * 80)
    print(f"QUESTION: {question}")
    print("=" * 80)
    print(f"\\nBASELINE ANSWER:\\n{baseline_answer}")
    print(f"\\nCRITIC-ENHANCED ANSWER:\\n{final_answer}")
    print("=" * 80)
    
    return {
        'question': question,
        'baseline': baseline_answer,
        'final': final_answer,
        'changed': baseline_answer != final_answer
    }

# Use it
comparison = compare_answers(
    "Where was the director of Inception born?",
    retriever,
    llm,
    your_rag_chain
)
"""
