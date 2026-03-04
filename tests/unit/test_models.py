"""Unit tests for critic system data models.

Tests the basic functionality and validation of all data model classes.
"""

import pytest
from src.critic.models import (
    ValidationResult,
    DecompositionResult,
    Document,
    ReasoningCheckResult,
    CriticState,
)


class TestValidationResult:
    """Tests for ValidationResult dataclass."""
    
    def test_passed_validation(self):
        """Test ValidationResult with passed validation."""
        result = ValidationResult(passed=True, triggered_rule=None)
        assert result.passed is True
        assert result.triggered_rule is None
    
    def test_failed_validation(self):
        """Test ValidationResult with failed validation."""
        result = ValidationResult(passed=False, triggered_rule="length_exceeded")
        assert result.passed is False
        assert result.triggered_rule == "length_exceeded"


class TestDecompositionResult:
    """Tests for DecompositionResult dataclass."""
    
    def test_valid_single_subquery(self):
        """Test DecompositionResult with single sub-query."""
        result = DecompositionResult(
            sub_queries=["What is the capital?"],
            reasoning_plan="Answer directly from sub-query 1"
        )
        assert len(result.sub_queries) == 1
        assert result.reasoning_plan == "Answer directly from sub-query 1"
    
    def test_valid_multiple_subqueries(self):
        """Test DecompositionResult with multiple sub-queries."""
        result = DecompositionResult(
            sub_queries=["Who is the author?", "When was it published?"],
            reasoning_plan="Combine author from sub-query 1 with date from sub-query 2"
        )
        assert len(result.sub_queries) == 2
    
    def test_valid_max_subqueries(self):
        """Test DecompositionResult with maximum 3 sub-queries."""
        result = DecompositionResult(
            sub_queries=["Query 1", "Query 2", "Query 3"],
            reasoning_plan="Combine all three"
        )
        assert len(result.sub_queries) == 3
    
    def test_invalid_zero_subqueries(self):
        """Test DecompositionResult rejects empty sub-queries list."""
        with pytest.raises(ValueError, match="must contain 1-3 elements"):
            DecompositionResult(
                sub_queries=[],
                reasoning_plan="Invalid"
            )
    
    def test_invalid_too_many_subqueries(self):
        """Test DecompositionResult rejects more than 3 sub-queries."""
        with pytest.raises(ValueError, match="must contain 1-3 elements"):
            DecompositionResult(
                sub_queries=["Q1", "Q2", "Q3", "Q4"],
                reasoning_plan="Too many"
            )
    
    def test_invalid_empty_subquery_string(self):
        """Test DecompositionResult rejects empty sub-query strings."""
        with pytest.raises(ValueError, match="must be non-empty strings"):
            DecompositionResult(
                sub_queries=["Valid", ""],
                reasoning_plan="Has empty"
            )
    
    def test_invalid_empty_reasoning_plan(self):
        """Test DecompositionResult rejects empty reasoning plan."""
        with pytest.raises(ValueError, match="must be a non-empty string"):
            DecompositionResult(
                sub_queries=["Valid query"],
                reasoning_plan=""
            )


class TestDocument:
    """Tests for Document dataclass."""
    
    def test_valid_document(self):
        """Test Document with valid fields."""
        doc = Document(
            text="This is document content",
            source="doc_123",
            score=0.95
        )
        assert doc.text == "This is document content"
        assert doc.source == "doc_123"
        assert doc.score == 0.95


class TestReasoningCheckResult:
    """Tests for ReasoningCheckResult dataclass."""
    
    def test_pass_action(self):
        """Test ReasoningCheckResult with PASS action."""
        result = ReasoningCheckResult(
            action="PASS",
            missing_facts=[],
            reasoning_explanation="All checks passed"
        )
        assert result.action == "PASS"
        assert result.missing_facts == []
    
    def test_retry_retrieval_action(self):
        """Test ReasoningCheckResult with RETRY_RETRIEVAL action."""
        result = ReasoningCheckResult(
            action="RETRY_RETRIEVAL",
            missing_facts=["sub-query 1"],
            reasoning_explanation="Missing facts for sub-query 1"
        )
        assert result.action == "RETRY_RETRIEVAL"
        assert result.missing_facts == ["sub-query 1"]
    
    def test_regenerate_action(self):
        """Test ReasoningCheckResult with REGENERATE action."""
        result = ReasoningCheckResult(
            action="REGENERATE",
            missing_facts=[],
            reasoning_explanation="Reasoning invalid"
        )
        assert result.action == "REGENERATE"
    
    def test_invalid_action(self):
        """Test ReasoningCheckResult rejects invalid action."""
        with pytest.raises(ValueError, match="action must be one of"):
            ReasoningCheckResult(
                action="INVALID_ACTION",
                missing_facts=[],
                reasoning_explanation="Invalid"
            )


class TestCriticState:
    """Tests for CriticState dataclass."""
    
    def test_initial_state(self):
        """Test CriticState with initial values."""
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
    
    def test_state_with_decomposition(self):
        """Test CriticState with decomposition results."""
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
    
    def test_state_with_validation_history(self):
        """Test CriticState with validation history."""
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
