"""Simple validation script for data models (no pytest required).

This script validates that all data models work correctly without requiring
pytest to be installed. Useful for quick validation during development.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.critic.models import (
    ValidationResult,
    DecompositionResult,
    Document,
    ReasoningCheckResult,
    CriticState,
)


def test_validation_result():
    """Test ValidationResult dataclass."""
    # Test passed validation
    result = ValidationResult(passed=True, triggered_rule=None)
    assert result.passed is True
    assert result.triggered_rule is None
    
    # Test failed validation
    result = ValidationResult(passed=False, triggered_rule="length_exceeded")
    assert result.passed is False
    assert result.triggered_rule == "length_exceeded"
    
    print("✓ ValidationResult tests passed")


def test_decomposition_result():
    """Test DecompositionResult dataclass."""
    # Test valid single sub-query
    result = DecompositionResult(
        sub_queries=["What is the capital?"],
        reasoning_plan="Answer directly from sub-query 1"
    )
    assert len(result.sub_queries) == 1
    
    # Test valid multiple sub-queries
    result = DecompositionResult(
        sub_queries=["Who is the author?", "When was it published?"],
        reasoning_plan="Combine author from sub-query 1 with date from sub-query 2"
    )
    assert len(result.sub_queries) == 2
    
    # Test valid max sub-queries
    result = DecompositionResult(
        sub_queries=["Query 1", "Query 2", "Query 3"],
        reasoning_plan="Combine all three"
    )
    assert len(result.sub_queries) == 3
    
    # Test invalid zero sub-queries
    try:
        DecompositionResult(sub_queries=[], reasoning_plan="Invalid")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "must contain 1-3 elements" in str(e)
    
    # Test invalid too many sub-queries
    try:
        DecompositionResult(
            sub_queries=["Q1", "Q2", "Q3", "Q4"],
            reasoning_plan="Too many"
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "must contain 1-3 elements" in str(e)
    
    # Test invalid empty sub-query string
    try:
        DecompositionResult(
            sub_queries=["Valid", ""],
            reasoning_plan="Has empty"
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "must be non-empty strings" in str(e)
    
    # Test invalid empty reasoning plan
    try:
        DecompositionResult(
            sub_queries=["Valid query"],
            reasoning_plan=""
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "must be a non-empty string" in str(e)
    
    print("✓ DecompositionResult tests passed")


def test_document():
    """Test Document dataclass."""
    doc = Document(
        text="This is document content",
        source="doc_123",
        score=0.95
    )
    assert doc.text == "This is document content"
    assert doc.source == "doc_123"
    assert doc.score == 0.95
    
    print("✓ Document tests passed")


def test_reasoning_check_result():
    """Test ReasoningCheckResult dataclass."""
    # Test PASS action
    result = ReasoningCheckResult(
        action="PASS",
        missing_facts=[],
        reasoning_explanation="All checks passed"
    )
    assert result.action == "PASS"
    assert result.missing_facts == []
    
    # Test RETRY_RETRIEVAL action
    result = ReasoningCheckResult(
        action="RETRY_RETRIEVAL",
        missing_facts=["sub-query 1"],
        reasoning_explanation="Missing facts for sub-query 1"
    )
    assert result.action == "RETRY_RETRIEVAL"
    assert result.missing_facts == ["sub-query 1"]
    
    # Test REGENERATE action
    result = ReasoningCheckResult(
        action="REGENERATE",
        missing_facts=[],
        reasoning_explanation="Reasoning invalid"
    )
    assert result.action == "REGENERATE"
    
    # Test invalid action
    try:
        ReasoningCheckResult(
            action="INVALID_ACTION",
            missing_facts=[],
            reasoning_explanation="Invalid"
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "action must be one of" in str(e)
    
    print("✓ ReasoningCheckResult tests passed")


def test_critic_state():
    """Test CriticState dataclass."""
    # Test initial state
    state = CriticState(
        question="What is the capital of France?",
        current_answer="Paris"
    )
    assert state.question == "What is the capital of France?"
    assert state.current_answer == "Paris"
    assert state.iteration_count == 0
    assert state.sub_queries is None
    assert state.reasoning_plan is None
    assert state.sub_query_contexts is None
    assert state.validation_history == []
    
    # Test state with decomposition
    state = CriticState(
        question="Test question",
        current_answer="Test answer",
        iteration_count=1,
        sub_queries=["Sub-query 1", "Sub-query 2"],
        reasoning_plan="Combine sub-answers"
    )
    assert state.sub_queries == ["Sub-query 1", "Sub-query 2"]
    assert state.reasoning_plan == "Combine sub-answers"
    assert state.iteration_count == 1
    
    # Test state with validation history
    validation1 = ValidationResult(passed=False, triggered_rule="length_exceeded")
    validation2 = ValidationResult(passed=True, triggered_rule=None)
    
    state = CriticState(
        question="Test",
        current_answer="Answer",
        validation_history=[validation1, validation2]
    )
    assert len(state.validation_history) == 2
    assert state.validation_history[0].passed is False
    assert state.validation_history[1].passed is True
    
    print("✓ CriticState tests passed")


if __name__ == "__main__":
    print("Running data model validation tests...\n")
    
    test_validation_result()
    test_decomposition_result()
    test_document()
    test_reasoning_check_result()
    test_critic_state()
    
    print("\n✅ All data model tests passed successfully!")
