"""Generator component for the LLM Critic Enhancement system.

This module implements the Generator component that produces answers using
sub-query contexts and reasoning plans by wrapping the existing RAG pipeline
generation function with an augmented prompt.
"""

from typing import Callable, Dict, List

from src.critic.models import Document


def generate_with_reasoning(
    question: str,
    sub_query_contexts: Dict[str, List[Document]],
    reasoning_plan: str,
    generation_function: Callable[[str, str], str]
) -> str:
    """Generate answer using sub-query contexts and reasoning plan.
    
    This function wraps the existing RAG pipeline generation function by
    formatting the retrieved context to show sub-query associations and
    creating an augmented prompt that includes the reasoning plan. It handles
    empty answer generation and LLM timeout/error gracefully.
    
    Args:
        question: Original question to answer
        sub_query_contexts: Dictionary mapping sub-queries to their retrieved documents
        reasoning_plan: Description of how to combine sub-answers to produce final answer
        generation_function: Existing RAG pipeline generation function that takes
                            a question and context string, returns an answer string
    
    Returns:
        Generated answer string. Returns "I do not know." if generation fails,
        times out, or produces an empty answer.
    
    Example:
        >>> def mock_generation(question: str, context: str) -> str:
        ...     return "Paris"
        >>> contexts = {
        ...     "What is the capital?": [Document("Paris is capital", "doc1", 0.9)]
        ... }
        >>> answer = generate_with_reasoning(
        ...     "What is the capital of France?",
        ...     contexts,
        ...     "Use the capital information to answer",
        ...     mock_generation
        ... )
        >>> answer
        'Paris'
    """
    try:
        # Format context with sub-query associations
        formatted_context = _format_context_with_subqueries(
            sub_query_contexts,
            reasoning_plan
        )
        
        # Create augmented prompt
        augmented_question = _create_augmented_prompt(
            question,
            reasoning_plan
        )
        
        # Call existing generation function
        answer = generation_function(augmented_question, formatted_context)
        
        # Handle empty answer generation
        if not answer or not answer.strip():
            return "I do not know."
        
        return answer.strip()
        
    except Exception as e:
        # Handle LLM timeout/error
        print(f"Warning: Generation failed for question '{question}': {e}")
        return "I do not know."


def _format_context_with_subqueries(
    sub_query_contexts: Dict[str, List[Document]],
    reasoning_plan: str
) -> str:
    """Format retrieved context to show sub-query associations.
    
    Args:
        sub_query_contexts: Dictionary mapping sub-queries to their retrieved documents
        reasoning_plan: Reasoning plan to include in context
    
    Returns:
        Formatted context string with sub-query associations
    """
    context_parts = []
    
    # Add reasoning plan at the top
    context_parts.append(f"Reasoning Plan: {reasoning_plan}\n")
    context_parts.append("Retrieved Information:\n")
    
    # Format each sub-query with its documents
    for sub_query, documents in sub_query_contexts.items():
        context_parts.append(f"\nSub-query: {sub_query}")
        
        if not documents:
            context_parts.append("  No documents retrieved.")
        else:
            for i, doc in enumerate(documents, 1):
                context_parts.append(f"  Document {i} (source: {doc.source}, score: {doc.score:.3f}):")
                context_parts.append(f"    {doc.text}")
    
    return "\n".join(context_parts)


def _create_augmented_prompt(question: str, reasoning_plan: str) -> str:
    """Create augmented prompt with reasoning plan.
    
    Args:
        question: Original question
        reasoning_plan: How to combine sub-answers
    
    Returns:
        Augmented question string
    """
    return (
        f"For the question: {question}\n\n"
        f"Follow this reasoning plan: {reasoning_plan}\n\n"
        f"Generate a concise answer following the reasoning plan."
    )
