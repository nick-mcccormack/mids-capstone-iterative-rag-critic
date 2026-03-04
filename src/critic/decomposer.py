"""Query decomposition component for the LLM Critic Enhancement system.

This module implements LLM-based query decomposition that breaks complex multi-hop
queries into 1-3 sub-queries with a reasoning plan. The decomposer uses a single
LLM prompt to generate both sub-queries and reasoning plan in JSON format, with
retry logic for malformed responses and fallback handling for invalid sub-query counts.
"""

import json
import logging
from typing import Optional

from src.critic.models import DecompositionResult
from src.rag.generator import call_llm


logger = logging.getLogger(__name__)


# Maximum number of retries for malformed JSON responses
MAX_RETRIES = 2


def decompose_query(question: str, max_retries: int = MAX_RETRIES) -> DecompositionResult:
    """
    Decomposes a multi-hop question into sub-queries with a reasoning plan.
    
    Uses an LLM to break complex queries into 1-3 independently answerable sub-queries
    and generates a reasoning plan showing how the sub-answers combine to produce the
    final answer. The function makes a single LLM call per attempt, parses the JSON
    response, validates constraints, and handles errors with retry logic and fallbacks.
    
    Args:
        question: The original multi-hop question to decompose
        max_retries: Maximum number of retry attempts for malformed JSON (default: 2)
    
    Returns:
        DecompositionResult containing 1-3 sub-queries and reasoning plan
    
    Raises:
        ValueError: If question is empty or None
    
    Error Handling:
        - Malformed JSON: Retries with clarified prompt (max 2 retries)
        - Invalid sub-query count (<1 or >3): Falls back to treating original question
          as single sub-query with default reasoning plan
        - Missing reasoning plan: Generates default plan "Answer the question using
          retrieved information"
        - LLM timeout/error: Raises exception to caller (should trigger fallback to
          baseline RAG pipeline)
    
    Examples:
        >>> result = decompose_query("Where was the director of Inception born?")
        >>> len(result.sub_queries)
        2
        >>> result.sub_queries[0]
        "Who directed Inception?"
        >>> result.sub_queries[1]
        "Where was [director name] born?"
        >>> "director" in result.reasoning_plan.lower()
        True
    """
    if not question or not question.strip():
        raise ValueError("Question cannot be empty")
    
    # Try to decompose with retries for malformed JSON
    for attempt in range(max_retries + 1):
        try:
            # Call LLM with decomposition prompt
            response_text = _call_decomposition_llm(question, attempt)
            
            # Parse JSON response
            parsed = json.loads(response_text)
            
            # Extract sub-queries and reasoning plan
            sub_queries = parsed.get("sub_queries", [])
            reasoning_plan = parsed.get("reasoning_plan", "")
            
            # Validate and handle missing reasoning plan
            if not reasoning_plan or not reasoning_plan.strip():
                logger.warning("Missing reasoning plan in LLM response, using default")
                reasoning_plan = "Answer the question using retrieved information"
            
            # Validate sub-query count
            if not isinstance(sub_queries, list) or len(sub_queries) < 1 or len(sub_queries) > 3:
                logger.warning(
                    f"Invalid sub-query count: {len(sub_queries) if isinstance(sub_queries, list) else 'not a list'}. "
                    f"Using fallback."
                )
                return _create_fallback_decomposition(question)
            
            # Validate sub-queries are non-empty strings
            if not all(isinstance(sq, str) and sq.strip() for sq in sub_queries):
                logger.warning("Sub-queries contain empty or non-string values. Using fallback.")
                return _create_fallback_decomposition(question)
            
            # Create and return result (will validate constraints in __post_init__)
            return DecompositionResult(
                sub_queries=sub_queries,
                reasoning_plan=reasoning_plan
            )
            
        except json.JSONDecodeError as e:
            logger.warning(f"Malformed JSON on attempt {attempt + 1}/{max_retries + 1}: {e}")
            if attempt >= max_retries:
                logger.error("Max retries reached for JSON parsing. Using fallback.")
                return _create_fallback_decomposition(question)
            # Continue to next retry attempt
            
        except ValueError as e:
            # This catches validation errors from DecompositionResult.__post_init__
            logger.warning(f"Validation error on attempt {attempt + 1}: {e}. Using fallback.")
            return _create_fallback_decomposition(question)
        
        except Exception as e:
            # Catch any other unexpected errors (LLM timeout, network issues, etc.)
            logger.error(f"Unexpected error during decomposition: {e}")
            raise
    
    # Should not reach here, but fallback just in case
    return _create_fallback_decomposition(question)


def _call_decomposition_llm(question: str, retry_attempt: int = 0) -> str:
    """
    Calls the LLM with the decomposition prompt.
    
    Args:
        question: The question to decompose
        retry_attempt: Current retry attempt number (0 for first attempt)
    
    Returns:
        Raw LLM response text (should be JSON)
    """
    system_prompt = _create_system_prompt(retry_attempt)
    user_prompt = _create_user_prompt(question)
    
    # Use temperature=0.2 for more deterministic decomposition
    return call_llm(system_prompt, user_prompt, temperature=0.2)


def _create_system_prompt(retry_attempt: int = 0) -> str:
    """
    Creates the system prompt for query decomposition.
    
    Args:
        retry_attempt: Current retry attempt number (0 for first attempt)
    
    Returns:
        System prompt string
    """
    base_prompt = """You are analyzing a multi-hop question that requires multiple pieces of information to answer.

Your task:
1. Break this question into 1-3 sub-questions that each retrieve a specific piece of information
2. Create a reasoning plan showing how the answers to sub-questions combine to produce the final answer

Requirements:
- Generate 1-3 sub-queries (no more, no less)
- Each sub-query should be independently answerable
- Reasoning plan must reference each sub-query's answer
- Use clear, specific language"""
    
    if retry_attempt > 0:
        # Add clarification for retry attempts
        base_prompt += """

IMPORTANT: Your previous response had formatting issues. Please ensure you output ONLY valid JSON with no additional text."""
    
    return base_prompt


def _create_user_prompt(question: str) -> str:
    """
    Creates the user prompt for query decomposition.
    
    Args:
        question: The question to decompose
    
    Returns:
        User prompt string with question and output format
    """
    return f"""Question: {question}

Output format (JSON):
{{
  "sub_queries": [
    "sub-question 1",
    "sub-question 2",
    ...
  ],
  "reasoning_plan": "Step-by-step explanation of how sub-answers combine: [sub-answer 1] provides X, [sub-answer 2] provides Y, combining them gives final answer Z"
}}

Respond with ONLY the JSON object, no additional text."""


def _create_fallback_decomposition(question: str) -> DecompositionResult:
    """
    Creates a fallback decomposition when LLM fails or returns invalid results.
    
    Treats the original question as a single sub-query with a default reasoning plan.
    
    Args:
        question: The original question
    
    Returns:
        DecompositionResult with single sub-query
    """
    logger.info("Using fallback decomposition: treating question as single sub-query")
    return DecompositionResult(
        sub_queries=[question],
        reasoning_plan="Answer the question using retrieved information"
    )
