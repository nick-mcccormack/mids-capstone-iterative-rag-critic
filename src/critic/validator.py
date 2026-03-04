"""Answer validation component for the LLM Critic Enhancement system.

This module implements rule-based quality checks to determine if an answer
requires critic intervention. The validator detects:
- Exact "I do not know." responses
- Overly long answers (>90 words)
- Hedging phrases indicating uncertainty
- Multiple entities with conjunctions for certain question types
"""

from typing import List
from src.critic.models import ValidationResult


# Hedging phrases that indicate uncertainty or over-answering
HEDGING_PHRASES = [
    "however",
    "but the context does not",
    "some also",
    "specifically",
    "but also"
]

# Question types that should have single entity answers
INTERROGATIVE_PREFIXES = ["Where", "Who", "What year", "When"]

# Conjunctions that indicate multiple entities in an answer
ENTITY_CONJUNCTIONS = ["and", "or", "but also"]


def validate_answer(answer: str, question: str) -> ValidationResult:
    """
    Checks answer against quality criteria to determine if critic intervention is needed.
    
    The validator applies four rule-based checks:
    1. Exact match check: answer == "I do not know."
    2. Length check: answer exceeds 90 words
    3. Hedging phrase check: answer contains uncertainty indicators
    4. Multiple entity check: answer contains multiple entities with conjunctions
       for questions that should have single answers
    
    Args:
        answer: The answer text to validate
        question: The original question (needed for entity check)
    
    Returns:
        ValidationResult with pass/fail status and triggered rule name.
        If passed=False, triggered_rule contains the name of the first rule that failed.
        If passed=True, triggered_rule is None.
    
    Examples:
        >>> validate_answer("I do not know.", "What is the capital?")
        ValidationResult(passed=False, triggered_rule="unknown_answer")
        
        >>> validate_answer("A very long answer with more than 90 words...", "What?")
        ValidationResult(passed=False, triggered_rule="length_exceeded")
        
        >>> validate_answer("The answer is X, however the context does not specify Y", "What?")
        ValidationResult(passed=False, triggered_rule="hedging_detected")
        
        >>> validate_answer("Paris and London", "Where was the event held?")
        ValidationResult(passed=False, triggered_rule="multiple_entities")
        
        >>> validate_answer("Paris", "Where was the event held?")
        ValidationResult(passed=True, triggered_rule=None)
    """
    # Rule 1: Check for exact "I do not know." response
    if answer == "I do not know.":
        return ValidationResult(passed=False, triggered_rule="unknown_answer")
    
    # Rule 2: Check if answer exceeds 90 words
    word_count = len(answer.split())
    if word_count > 90:
        return ValidationResult(passed=False, triggered_rule="length_exceeded")
    
    # Rule 3: Check for hedging phrases
    for phrase in HEDGING_PHRASES:
        if phrase in answer:
            return ValidationResult(passed=False, triggered_rule="hedging_detected")
    
    # Rule 4: Check for multiple entities with conjunctions
    # Only applies to questions starting with specific interrogatives
    if _question_starts_with_interrogative(question):
        if _has_multiple_entities(answer):
            return ValidationResult(passed=False, triggered_rule="multiple_entities")
    
    # All checks passed
    return ValidationResult(passed=True, triggered_rule=None)


def _question_starts_with_interrogative(question: str) -> bool:
    """
    Checks if question starts with an interrogative that expects a single entity answer.
    
    Args:
        question: The question text to check
    
    Returns:
        True if question starts with "Where", "Who", "What year", or "When"
    """
    question_stripped = question.strip()
    for prefix in INTERROGATIVE_PREFIXES:
        if question_stripped.startswith(prefix):
            return True
    return False


def _has_multiple_entities(answer: str) -> bool:
    """
    Checks if answer contains multiple entities joined by conjunctions.
    
    Args:
        answer: The answer text to check
    
    Returns:
        True if answer contains "and", "or", or "but also"
    """
    for conjunction in ENTITY_CONJUNCTIONS:
        if conjunction in answer:
            return True
    return False
