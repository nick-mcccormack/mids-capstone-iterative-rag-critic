"""Simple validation script for decomposer (no pytest required).

This script validates that the decompose_query function works correctly without
requiring pytest to be installed. Uses mocking to avoid LLM API calls.
"""

import sys
import json
from pathlib import Path
from unittest.mock import patch

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.critic.decomposer import decompose_query, _create_fallback_decomposition
from src.critic.models import DecompositionResult


def test_successful_decomposition_two_subqueries():
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
    
    print("✓ Successful decomposition with 2 sub-queries")


def test_successful_decomposition_one_subquery():
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
    
    print("✓ Successful decomposition with 1 sub-query")


def test_successful_decomposition_three_subqueries():
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
    
    print("✓ Successful decomposition with 3 sub-queries")


def test_malformed_json_with_retry():
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
    
    print("✓ Malformed JSON retry logic works")


def test_malformed_json_max_retries_fallback():
    """Test that max retries for malformed JSON triggers fallback."""
    malformed_response = "Not JSON at all"
    
    with patch('src.critic.decomposer.call_llm', return_value=malformed_response):
        result = decompose_query("What is the answer?", max_retries=2)
    
    # Should fall back to single sub-query
    assert len(result.sub_queries) == 1
    assert result.sub_queries[0] == "What is the answer?"
    assert result.reasoning_plan == "Answer the question using retrieved information"
    
    print("✓ Max retries fallback works")


def test_invalid_subquery_count_zero():
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
    
    print("✓ Invalid count (0) triggers fallback")


def test_invalid_subquery_count_four():
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
    
    print("✓ Invalid count (4) triggers fallback")


def test_missing_reasoning_plan():
    """Test that missing reasoning plan uses default."""
    mock_response = json.dumps({
        "sub_queries": ["What is the answer?"],
        "reasoning_plan": ""
    })
    
    with patch('src.critic.decomposer.call_llm', return_value=mock_response):
        result = decompose_query("What is the answer?")
    
    assert len(result.sub_queries) == 1
    assert result.reasoning_plan == "Answer the question using retrieved information"
    
    print("✓ Missing reasoning plan uses default")


def test_empty_question_raises_error():
    """Test that empty question raises ValueError."""
    try:
        decompose_query("")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Question cannot be empty" in str(e)
    
    print("✓ Empty question raises ValueError")


def test_none_question_raises_error():
    """Test that None question raises ValueError."""
    try:
        decompose_query(None)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Question cannot be empty" in str(e)
    
    print("✓ None question raises ValueError")


def test_fallback_decomposition():
    """Test that fallback creates a single sub-query from the original question."""
    question = "What is the meaning of life?"
    result = _create_fallback_decomposition(question)
    
    assert isinstance(result, DecompositionResult)
    assert len(result.sub_queries) == 1
    assert result.sub_queries[0] == question
    assert result.reasoning_plan == "Answer the question using retrieved information"
    
    print("✓ Fallback decomposition works correctly")


if __name__ == "__main__":
    print("Running decomposer validation tests...\n")
    
    try:
        test_successful_decomposition_two_subqueries()
        test_successful_decomposition_one_subquery()
        test_successful_decomposition_three_subqueries()
        test_malformed_json_with_retry()
        test_malformed_json_max_retries_fallback()
        test_invalid_subquery_count_zero()
        test_invalid_subquery_count_four()
        test_missing_reasoning_plan()
        test_empty_question_raises_error()
        test_none_question_raises_error()
        test_fallback_decomposition()
        
        print("\n✅ All decomposer tests passed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
