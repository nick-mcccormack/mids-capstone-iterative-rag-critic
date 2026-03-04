"""Unit tests for the Query Decomposer component.

Tests the decompose_query function including:
- Successful decomposition with 1, 2, and 3 sub-queries
- Malformed JSON handling with retry logic
- Invalid sub-query count handling (0, 4+)
- Missing reasoning plan handling
- Empty question validation
"""

import json
import pytest
from unittest.mock import patch, MagicMock

from src.critic.decomposer import decompose_query, _create_fallback_decomposition
from src.critic.models import DecompositionResult


class TestDecomposeQuery:
    """Test suite for decompose_query function."""
    
    def test_successful_decomposition_with_two_subqueries(self):
        """Test successful decomposition with 2 sub-queries."""
        mock_response = json.dumps({
            "sub_queries": [
                "Who directed Inception?",
                "Where was Christopher Nolan born?"
            ],
            "reasoning_plan": "First find the director from sub-query 1, then find their birthplace from sub-query 2"
        })
        
        with patch('src.critic.decomposer.call_llm', return_value=mock_response):
            result = decompose_query("Where was the director of Inception born?")
        
        assert isinstance(result, DecompositionResult)
        assert len(result.sub_queries) == 2
        assert result.sub_queries[0] == "Who directed Inception?"
        assert result.sub_queries[1] == "Where was Christopher Nolan born?"
        assert "director" in result.reasoning_plan.lower()
    
    def test_successful_decomposition_with_one_subquery(self):
        """Test successful decomposition with 1 sub-query."""
        mock_response = json.dumps({
            "sub_queries": ["What is the capital of France?"],
            "reasoning_plan": "Answer directly from the retrieved information"
        })
        
        with patch('src.critic.decomposer.call_llm', return_value=mock_response):
            result = decompose_query("What is the capital of France?")
        
        assert len(result.sub_queries) == 1
        assert result.sub_queries[0] == "What is the capital of France?"
        assert len(result.reasoning_plan) > 0
    
    def test_successful_decomposition_with_three_subqueries(self):
        """Test successful decomposition with 3 sub-queries (maximum)."""
        mock_response = json.dumps({
            "sub_queries": [
                "Who wrote the novel 1984?",
                "When was George Orwell born?",
                "Where was George Orwell born?"
            ],
            "reasoning_plan": "Find the author, then find their birth date and location"
        })
        
        with patch('src.critic.decomposer.call_llm', return_value=mock_response):
            result = decompose_query("When and where was the author of 1984 born?")
        
        assert len(result.sub_queries) == 3
        assert all(isinstance(sq, str) for sq in result.sub_queries)
        assert len(result.reasoning_plan) > 0
    
    def test_malformed_json_with_retry_success(self):
        """Test that malformed JSON triggers retry and succeeds on second attempt."""
        malformed_response = "This is not JSON {sub_queries: []}"
        valid_response = json.dumps({
            "sub_queries": ["What is the answer?"],
            "reasoning_plan": "Answer the question"
        })
        
        with patch('src.critic.decomposer.call_llm', side_effect=[malformed_response, valid_response]):
            result = decompose_query("What is the answer?")
        
        assert len(result.sub_queries) == 1
        assert result.sub_queries[0] == "What is the answer?"
    
    def test_malformed_json_max_retries_fallback(self):
        """Test that max retries for malformed JSON triggers fallback."""
        malformed_response = "Not JSON at all"
        
        with patch('src.critic.decomposer.call_llm', return_value=malformed_response):
            result = decompose_query("What is the answer?", max_retries=2)
        
        # Should fall back to single sub-query
        assert len(result.sub_queries) == 1
        assert result.sub_queries[0] == "What is the answer?"
        assert result.reasoning_plan == "Answer the question using retrieved information"
    
    def test_invalid_subquery_count_zero(self):
        """Test that 0 sub-queries triggers fallback."""
        mock_response = json.dumps({
            "sub_queries": [],
            "reasoning_plan": "Some plan"
        })
        
        with patch('src.critic.decomposer.call_llm', return_value=mock_response):
            result = decompose_query("What is the answer?")
        
        # Should fall back to single sub-query
        assert len(result.sub_queries) == 1
        assert result.sub_queries[0] == "What is the answer?"
    
    def test_invalid_subquery_count_four(self):
        """Test that 4 sub-queries triggers fallback."""
        mock_response = json.dumps({
            "sub_queries": ["Q1", "Q2", "Q3", "Q4"],
            "reasoning_plan": "Too many queries"
        })
        
        with patch('src.critic.decomposer.call_llm', return_value=mock_response):
            result = decompose_query("What is the answer?")
        
        # Should fall back to single sub-query
        assert len(result.sub_queries) == 1
        assert result.sub_queries[0] == "What is the answer?"
    
    def test_missing_reasoning_plan(self):
        """Test that missing reasoning plan uses default."""
        mock_response = json.dumps({
            "sub_queries": ["What is the answer?"],
            "reasoning_plan": ""
        })
        
        with patch('src.critic.decomposer.call_llm', return_value=mock_response):
            result = decompose_query("What is the answer?")
        
        assert len(result.sub_queries) == 1
        assert result.reasoning_plan == "Answer the question using retrieved information"
    
    def test_missing_reasoning_plan_field(self):
        """Test that missing reasoning_plan field uses default."""
        mock_response = json.dumps({
            "sub_queries": ["What is the answer?"]
        })
        
        with patch('src.critic.decomposer.call_llm', return_value=mock_response):
            result = decompose_query("What is the answer?")
        
        assert len(result.sub_queries) == 1
        assert result.reasoning_plan == "Answer the question using retrieved information"
    
    def test_empty_question_raises_error(self):
        """Test that empty question raises ValueError."""
        with pytest.raises(ValueError, match="Question cannot be empty"):
            decompose_query("")
    
    def test_none_question_raises_error(self):
        """Test that None question raises ValueError."""
        with pytest.raises(ValueError, match="Question cannot be empty"):
            decompose_query(None)
    
    def test_whitespace_only_question_raises_error(self):
        """Test that whitespace-only question raises ValueError."""
        with pytest.raises(ValueError, match="Question cannot be empty"):
            decompose_query("   ")
    
    def test_empty_subquery_strings_trigger_fallback(self):
        """Test that empty strings in sub-queries trigger fallback."""
        mock_response = json.dumps({
            "sub_queries": ["Valid question", "", "Another valid"],
            "reasoning_plan": "Some plan"
        })
        
        with patch('src.critic.decomposer.call_llm', return_value=mock_response):
            result = decompose_query("What is the answer?")
        
        # Should fall back to single sub-query
        assert len(result.sub_queries) == 1
        assert result.sub_queries[0] == "What is the answer?"
    
    def test_non_string_subqueries_trigger_fallback(self):
        """Test that non-string sub-queries trigger fallback."""
        mock_response = json.dumps({
            "sub_queries": ["Valid question", 123, None],
            "reasoning_plan": "Some plan"
        })
        
        with patch('src.critic.decomposer.call_llm', return_value=mock_response):
            result = decompose_query("What is the answer?")
        
        # Should fall back to single sub-query
        assert len(result.sub_queries) == 1
        assert result.sub_queries[0] == "What is the answer?"


class TestFallbackDecomposition:
    """Test suite for fallback decomposition."""
    
    def test_fallback_creates_single_subquery(self):
        """Test that fallback creates a single sub-query from the original question."""
        question = "What is the meaning of life?"
        result = _create_fallback_decomposition(question)
        
        assert isinstance(result, DecompositionResult)
        assert len(result.sub_queries) == 1
        assert result.sub_queries[0] == question
        assert result.reasoning_plan == "Answer the question using retrieved information"
