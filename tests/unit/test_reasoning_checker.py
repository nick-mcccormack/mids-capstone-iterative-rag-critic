"""Unit tests for the Reasoning Checker component.

Tests the check_reasoning function including LLM validation, JSON parsing,
decision logic, error handling, and fallback behavior.
"""

import json
import pytest
from unittest.mock import Mock, patch

from src.critic.models import Document, ReasoningCheckResult
from src.critic.reasoning_checker import check_reasoning


class TestCheckReasoning:
    """Test suite for check_reasoning function."""
    
    def test_pass_action_when_both_valid(self):
        """Test that PASS action is returned when both facts and reasoning are valid."""
        # Mock LLM response with both validations passing
        mock_response = json.dumps({
            "facts_retrieved": True,
            "missing_facts": [],
            "reasoning_valid": True,
            "reasoning_explanation": "All facts present and reasoning is correct"
        })
        
        with patch('src.critic.reasoning_checker.call_llm', return_value=mock_response):
            result = check_reasoning(
                question="Where was the director of Inception born?",
                sub_queries=["Who directed Inception?", "Where was Christopher Nolan born?"],
                sub_query_contexts={
                    "Who directed Inception?": [Document("Christopher Nolan directed Inception", "doc1", 0.9)],
                    "Where was Christopher Nolan born?": [Document("Christopher Nolan was born in London", "doc2", 0.9)]
                },
                reasoning_plan="Find the director, then find their birthplace",
                answer="Christopher Nolan was born in London"
            )
        
        assert result.action == "PASS"
        assert result.missing_facts == []
        assert "correct" in result.reasoning_explanation.lower()
    
    def test_retry_retrieval_action_when_facts_missing(self):
        """Test that RETRY_RETRIEVAL action is returned when facts are missing."""
        # Mock LLM response with missing facts
        mock_response = json.dumps({
            "facts_retrieved": False,
            "missing_facts": ["Where was Christopher Nolan born?"],
            "reasoning_valid": True,
            "reasoning_explanation": "Birthplace information not found in documents"
        })
        
        with patch('src.critic.reasoning_checker.call_llm', return_value=mock_response):
            result = check_reasoning(
                question="Where was the director of Inception born?",
                sub_queries=["Who directed Inception?", "Where was Christopher Nolan born?"],
                sub_query_contexts={
                    "Who directed Inception?": [Document("Christopher Nolan directed Inception", "doc1", 0.9)],
                    "Where was Christopher Nolan born?": []  # Empty documents
                },
                reasoning_plan="Find the director, then find their birthplace",
                answer="I do not know."
            )
        
        assert result.action == "RETRY_RETRIEVAL"
        assert "Where was Christopher Nolan born?" in result.missing_facts
        assert len(result.missing_facts) == 1
    
    def test_regenerate_action_when_reasoning_invalid(self):
        """Test that REGENERATE action is returned when reasoning is invalid."""
        # Mock LLM response with invalid reasoning
        mock_response = json.dumps({
            "facts_retrieved": True,
            "missing_facts": [],
            "reasoning_valid": False,
            "reasoning_explanation": "Answer does not follow the reasoning plan"
        })
        
        with patch('src.critic.reasoning_checker.call_llm', return_value=mock_response):
            result = check_reasoning(
                question="Where was the director of Inception born?",
                sub_queries=["Who directed Inception?", "Where was Christopher Nolan born?"],
                sub_query_contexts={
                    "Who directed Inception?": [Document("Christopher Nolan directed Inception", "doc1", 0.9)],
                    "Where was Christopher Nolan born?": [Document("Christopher Nolan was born in London", "doc2", 0.9)]
                },
                reasoning_plan="Find the director, then find their birthplace",
                answer="Inception was released in 2010"  # Wrong answer
            )
        
        assert result.action == "REGENERATE"
        assert result.missing_facts == []
        assert "not follow" in result.reasoning_explanation.lower()
    
    def test_malformed_json_retry_logic(self):
        """Test that malformed JSON triggers retry logic."""
        # First two attempts return malformed JSON, third returns valid
        mock_responses = [
            "This is not JSON",
            "{invalid json}",
            json.dumps({
                "facts_retrieved": True,
                "missing_facts": [],
                "reasoning_valid": True,
                "reasoning_explanation": "Valid after retries"
            })
        ]
        
        with patch('src.critic.reasoning_checker.call_llm', side_effect=mock_responses):
            result = check_reasoning(
                question="Test question?",
                sub_queries=["Sub-query 1"],
                sub_query_contexts={"Sub-query 1": []},
                reasoning_plan="Test plan",
                answer="Test answer"
            )
        
        # Should succeed after retries
        assert result.action == "PASS"
        assert "Valid after retries" in result.reasoning_explanation
    
    def test_max_retries_fallback(self):
        """Test that max retries triggers fallback to PASS."""
        # All attempts return malformed JSON
        with patch('src.critic.reasoning_checker.call_llm', return_value="Not JSON"):
            result = check_reasoning(
                question="Test question?",
                sub_queries=["Sub-query 1"],
                sub_query_contexts={"Sub-query 1": []},
                reasoning_plan="Test plan",
                answer="Test answer",
                max_retries=2
            )
        
        # Should fallback to PASS after max retries
        assert result.action == "PASS"
        assert "default" in result.reasoning_explanation.lower()
    
    def test_missing_validation_fields_use_defaults(self):
        """Test that missing validation fields use default values."""
        # Mock response missing some fields
        mock_response = json.dumps({
            "facts_retrieved": True
            # missing: missing_facts, reasoning_valid, reasoning_explanation
        })
        
        with patch('src.critic.reasoning_checker.call_llm', return_value=mock_response):
            result = check_reasoning(
                question="Test question?",
                sub_queries=["Sub-query 1"],
                sub_query_contexts={"Sub-query 1": []},
                reasoning_plan="Test plan",
                answer="Test answer"
            )
        
        # Should use defaults: reasoning_valid=False -> REGENERATE
        assert result.action == "REGENERATE"
        assert result.missing_facts == []
    
    def test_llm_exception_fallback(self):
        """Test that LLM exceptions trigger fallback to PASS."""
        with patch('src.critic.reasoning_checker.call_llm', side_effect=Exception("LLM timeout")):
            result = check_reasoning(
                question="Test question?",
                sub_queries=["Sub-query 1"],
                sub_query_contexts={"Sub-query 1": []},
                reasoning_plan="Test plan",
                answer="Test answer"
            )
        
        # Should fallback to PASS on exception
        assert result.action == "PASS"
        assert "default" in result.reasoning_explanation.lower()
    
    def test_empty_sub_query_contexts(self):
        """Test handling of empty sub-query contexts."""
        mock_response = json.dumps({
            "facts_retrieved": False,
            "missing_facts": ["Sub-query 1"],
            "reasoning_valid": False,
            "reasoning_explanation": "No documents retrieved"
        })
        
        with patch('src.critic.reasoning_checker.call_llm', return_value=mock_response):
            result = check_reasoning(
                question="Test question?",
                sub_queries=["Sub-query 1"],
                sub_query_contexts={},  # Empty contexts
                reasoning_plan="Test plan",
                answer="I do not know."
            )
        
        assert result.action == "RETRY_RETRIEVAL"
        assert "Sub-query 1" in result.missing_facts
    
    def test_single_llm_call_per_attempt(self):
        """Test that only one LLM call is made per attempt."""
        mock_response = json.dumps({
            "facts_retrieved": True,
            "missing_facts": [],
            "reasoning_valid": True,
            "reasoning_explanation": "Valid"
        })
        
        with patch('src.critic.reasoning_checker.call_llm', return_value=mock_response) as mock_llm:
            check_reasoning(
                question="Test question?",
                sub_queries=["Sub-query 1"],
                sub_query_contexts={"Sub-query 1": []},
                reasoning_plan="Test plan",
                answer="Test answer"
            )
        
        # Should make exactly one LLM call
        assert mock_llm.call_count == 1
    
    def test_decision_logic_all_combinations(self):
        """Test decision logic for all combinations of facts_retrieved and reasoning_valid."""
        test_cases = [
            # (facts_retrieved, reasoning_valid, expected_action)
            (True, True, "PASS"),
            (True, False, "REGENERATE"),
            (False, True, "RETRY_RETRIEVAL"),
            (False, False, "RETRY_RETRIEVAL"),  # facts take precedence
        ]
        
        for facts_retrieved, reasoning_valid, expected_action in test_cases:
            mock_response = json.dumps({
                "facts_retrieved": facts_retrieved,
                "missing_facts": [] if facts_retrieved else ["Sub-query 1"],
                "reasoning_valid": reasoning_valid,
                "reasoning_explanation": f"Test: facts={facts_retrieved}, reasoning={reasoning_valid}"
            })
            
            with patch('src.critic.reasoning_checker.call_llm', return_value=mock_response):
                result = check_reasoning(
                    question="Test question?",
                    sub_queries=["Sub-query 1"],
                    sub_query_contexts={"Sub-query 1": []},
                    reasoning_plan="Test plan",
                    answer="Test answer"
                )
            
            assert result.action == expected_action, \
                f"Expected {expected_action} for facts={facts_retrieved}, reasoning={reasoning_valid}"
    
    def test_missing_facts_not_list_defaults_to_empty(self):
        """Test that non-list missing_facts defaults to empty list."""
        mock_response = json.dumps({
            "facts_retrieved": False,
            "missing_facts": "not a list",  # Invalid type
            "reasoning_valid": True,
            "reasoning_explanation": "Test"
        })
        
        with patch('src.critic.reasoning_checker.call_llm', return_value=mock_response):
            result = check_reasoning(
                question="Test question?",
                sub_queries=["Sub-query 1"],
                sub_query_contexts={"Sub-query 1": []},
                reasoning_plan="Test plan",
                answer="Test answer"
            )
        
        assert result.action == "RETRY_RETRIEVAL"
        assert result.missing_facts == []  # Should default to empty list
