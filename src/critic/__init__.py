"""LLM Critic Enhancement for RAG Pipeline.

This module provides a critic system that validates answers, decomposes queries,
retrieves targeted context, and validates reasoning to improve RAG answer quality.
"""

from src.critic.models import (
    ValidationResult,
    DecompositionResult,
    Document,
    ReasoningCheckResult,
    CriticState,
)
from src.critic.validator import validate_answer
from src.critic.decomposer import decompose_query
from src.critic.retriever import retrieve_for_subqueries
from src.critic.generator import generate_with_reasoning
from src.critic.reasoning_checker import check_reasoning
from src.critic.critic_system import run_critic_system

__all__ = [
    "ValidationResult",
    "DecompositionResult",
    "Document",
    "ReasoningCheckResult",
    "CriticState",
    "validate_answer",
    "decompose_query",
    "retrieve_for_subqueries",
    "generate_with_reasoning",
    "check_reasoning",
    "run_critic_system",
]
