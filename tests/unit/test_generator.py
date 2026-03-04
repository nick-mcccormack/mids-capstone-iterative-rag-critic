"""Unit tests for the Generator component.

Tests cover:
- Generation with sub-query contexts
- Context formatting
- Empty answer handling
- LLM timeout/error handling
"""

import pytest
from typing import Dict, List

from src.critic.generator import generate_with_reasoning
from src.critic.models import Document


class TestGenerateWithReasoning:
    """Test suite for generate_with_reasoning function."""
    
    def test_generation_with_single_subquery_context(self):
        """Test generation with a single sub-query context."""
        # Arrange
        question = "What is the capital of France?"
        sub_query_contexts = {
            "What is the capital of France?": [
                Document(
                    text="Paris is the capital and largest city of France.",
                    source="France Geography",
                    score=0.95
                )
            ]
        }
        reasoning_plan = "Use the capital information to answer the question."
        
        def mock_generation(question: str, context: str) -> str:
            # Verify context contains sub-query and reasoning plan
            assert "Reasoning Plan:" in context
            assert "Sub-query:" in context
            assert "Paris is the capital" in context
            return "Paris"
        
        # Act
        answer = generate_with_reasoning(
            question,
            sub_query_contexts,
            reasoning_plan,
            mock_generation
        )
        
        # Assert
        assert answer == "Paris"
    
    def test_generation_with_multiple_subquery_contexts(self):
        """Test generation with multiple sub-query contexts."""
        # Arrange
        question = "Where was the director of Inception born?"
        sub_query_contexts = {
            "Who directed Inception?": [
                Document(
                    text="Inception was directed by Christopher Nolan.",
                    source="Movie Database",
                    score=0.98
                )
            ],
            "Where was Christopher Nolan born?": [
                Document(
                    text="Christopher Nolan was born in London, England.",
                    source="Biography",
                    score=0.96
                )
            ]
        }
        reasoning_plan = (
            "First find who directed Inception, then find where that person was born."
        )
        
        def mock_generation(question: str, context: str) -> str:
            # Verify context contains both sub-queries
            assert "Who directed Inception?" in context
            assert "Where was Christopher Nolan born?" in context
            assert "Christopher Nolan" in context
            assert "London, England" in context
            return "London, England"
        
        # Act
        answer = generate_with_reasoning(
            question,
            sub_query_contexts,
            reasoning_plan,
            mock_generation
        )
        
        # Assert
        assert answer == "London, England"
    
    def test_context_formatting_includes_reasoning_plan(self):
        """Test that formatted context includes reasoning plan."""
        # Arrange
        question = "Test question"
        sub_query_contexts = {
            "Sub-query 1": [
                Document("Doc 1", "Source 1", 0.9)
            ]
        }
        reasoning_plan = "This is the reasoning plan."
        
        captured_context = {"value": None}
        
        def mock_generation(question: str, context: str) -> str:
            captured_context["value"] = context
            return "Answer"
        
        # Act
        generate_with_reasoning(
            question,
            sub_query_contexts,
            reasoning_plan,
            mock_generation
        )
        
        # Assert
        context = captured_context["value"]
        assert "Reasoning Plan: This is the reasoning plan." in context
    
    def test_context_formatting_includes_subquery_associations(self):
        """Test that formatted context shows sub-query associations."""
        # Arrange
        question = "Test question"
        sub_query_contexts = {
            "Sub-query A": [
                Document("Doc A1", "Source A1", 0.95),
                Document("Doc A2", "Source A2", 0.90)
            ],
            "Sub-query B": [
                Document("Doc B1", "Source B1", 0.85)
            ]
        }
        reasoning_plan = "Combine A and B."
        
        captured_context = {"value": None}
        
        def mock_generation(question: str, context: str) -> str:
            captured_context["value"] = context
            return "Answer"
        
        # Act
        generate_with_reasoning(
            question,
            sub_query_contexts,
            reasoning_plan,
            mock_generation
        )
        
        # Assert
        context = captured_context["value"]
        assert "Sub-query: Sub-query A" in context
        assert "Sub-query: Sub-query B" in context
        assert "Doc A1" in context
        assert "Doc A2" in context
        assert "Doc B1" in context
        assert "Source A1" in context
        assert "0.950" in context  # Score formatting
    
    def test_context_formatting_with_empty_documents(self):
        """Test context formatting when a sub-query has no documents."""
        # Arrange
        question = "Test question"
        sub_query_contexts = {
            "Sub-query with docs": [
                Document("Doc 1", "Source 1", 0.9)
            ],
            "Sub-query without docs": []
        }
        reasoning_plan = "Use available information."
        
        captured_context = {"value": None}
        
        def mock_generation(question: str, context: str) -> str:
            captured_context["value"] = context
            return "Answer"
        
        # Act
        generate_with_reasoning(
            question,
            sub_query_contexts,
            reasoning_plan,
            mock_generation
        )
        
        # Assert
        context = captured_context["value"]
        assert "Sub-query: Sub-query with docs" in context
        assert "Sub-query: Sub-query without docs" in context
        assert "No documents retrieved." in context
    
    def test_augmented_prompt_includes_reasoning_plan(self):
        """Test that augmented prompt includes reasoning plan."""
        # Arrange
        question = "Original question"
        sub_query_contexts = {
            "Sub-query": [Document("Doc", "Source", 0.9)]
        }
        reasoning_plan = "Follow this plan."
        
        captured_question = {"value": None}
        
        def mock_generation(question: str, context: str) -> str:
            captured_question["value"] = question
            return "Answer"
        
        # Act
        generate_with_reasoning(
            question,
            sub_query_contexts,
            reasoning_plan,
            mock_generation
        )
        
        # Assert
        augmented_question = captured_question["value"]
        assert "Original question" in augmented_question
        assert "Follow this plan." in augmented_question
        assert "Generate a concise answer" in augmented_question
    
    def test_empty_answer_handling(self):
        """Test handling of empty answer generation."""
        # Arrange
        question = "Test question"
        sub_query_contexts = {
            "Sub-query": [Document("Doc", "Source", 0.9)]
        }
        reasoning_plan = "Plan"
        
        def mock_generation(question: str, context: str) -> str:
            return ""  # Empty answer
        
        # Act
        answer = generate_with_reasoning(
            question,
            sub_query_contexts,
            reasoning_plan,
            mock_generation
        )
        
        # Assert
        assert answer == "I do not know."
    
    def test_whitespace_only_answer_handling(self):
        """Test handling of whitespace-only answer."""
        # Arrange
        question = "Test question"
        sub_query_contexts = {
            "Sub-query": [Document("Doc", "Source", 0.9)]
        }
        reasoning_plan = "Plan"
        
        def mock_generation(question: str, context: str) -> str:
            return "   \n\t  "  # Whitespace only
        
        # Act
        answer = generate_with_reasoning(
            question,
            sub_query_contexts,
            reasoning_plan,
            mock_generation
        )
        
        # Assert
        assert answer == "I do not know."
    
    def test_llm_timeout_error_handling(self):
        """Test handling of LLM timeout/error."""
        # Arrange
        question = "Test question"
        sub_query_contexts = {
            "Sub-query": [Document("Doc", "Source", 0.9)]
        }
        reasoning_plan = "Plan"
        
        def mock_generation(question: str, context: str) -> str:
            raise TimeoutError("LLM request timed out")
        
        # Act
        answer = generate_with_reasoning(
            question,
            sub_query_contexts,
            reasoning_plan,
            mock_generation
        )
        
        # Assert
        assert answer == "I do not know."
    
    def test_llm_exception_handling(self):
        """Test handling of general LLM exceptions."""
        # Arrange
        question = "Test question"
        sub_query_contexts = {
            "Sub-query": [Document("Doc", "Source", 0.9)]
        }
        reasoning_plan = "Plan"
        
        def mock_generation(question: str, context: str) -> str:
            raise RuntimeError("LLM service unavailable")
        
        # Act
        answer = generate_with_reasoning(
            question,
            sub_query_contexts,
            reasoning_plan,
            mock_generation
        )
        
        # Assert
        assert answer == "I do not know."
    
    def test_answer_whitespace_trimming(self):
        """Test that answer whitespace is trimmed."""
        # Arrange
        question = "Test question"
        sub_query_contexts = {
            "Sub-query": [Document("Doc", "Source", 0.9)]
        }
        reasoning_plan = "Plan"
        
        def mock_generation(question: str, context: str) -> str:
            return "  Paris  \n"
        
        # Act
        answer = generate_with_reasoning(
            question,
            sub_query_contexts,
            reasoning_plan,
            mock_generation
        )
        
        # Assert
        assert answer == "Paris"
    
    def test_generation_with_three_subqueries(self):
        """Test generation with maximum three sub-queries."""
        # Arrange
        question = "Complex multi-hop question"
        sub_query_contexts = {
            "Sub-query 1": [Document("Doc 1", "Source 1", 0.95)],
            "Sub-query 2": [Document("Doc 2", "Source 2", 0.90)],
            "Sub-query 3": [Document("Doc 3", "Source 3", 0.85)]
        }
        reasoning_plan = "Combine all three pieces of information."
        
        captured_context = {"value": None}
        
        def mock_generation(question: str, context: str) -> str:
            captured_context["value"] = context
            return "Final answer"
        
        # Act
        answer = generate_with_reasoning(
            question,
            sub_query_contexts,
            reasoning_plan,
            mock_generation
        )
        
        # Assert
        assert answer == "Final answer"
        context = captured_context["value"]
        assert "Sub-query 1" in context
        assert "Sub-query 2" in context
        assert "Sub-query 3" in context
    
    def test_generation_with_multiple_documents_per_subquery(self):
        """Test generation with multiple documents per sub-query."""
        # Arrange
        question = "Test question"
        sub_query_contexts = {
            "Sub-query": [
                Document("Doc 1", "Source 1", 0.95),
                Document("Doc 2", "Source 2", 0.90),
                Document("Doc 3", "Source 3", 0.85)
            ]
        }
        reasoning_plan = "Use all documents."
        
        captured_context = {"value": None}
        
        def mock_generation(question: str, context: str) -> str:
            captured_context["value"] = context
            return "Answer"
        
        # Act
        generate_with_reasoning(
            question,
            sub_query_contexts,
            reasoning_plan,
            mock_generation
        )
        
        # Assert
        context = captured_context["value"]
        assert "Document 1" in context
        assert "Document 2" in context
        assert "Document 3" in context
        assert "Doc 1" in context
        assert "Doc 2" in context
        assert "Doc 3" in context
    
    def test_empty_subquery_contexts(self):
        """Test generation with empty sub-query contexts."""
        # Arrange
        question = "Test question"
        sub_query_contexts = {}
        reasoning_plan = "No sub-queries available."
        
        def mock_generation(question: str, context: str) -> str:
            return "Answer based on question only"
        
        # Act
        answer = generate_with_reasoning(
            question,
            sub_query_contexts,
            reasoning_plan,
            mock_generation
        )
        
        # Assert
        assert answer == "Answer based on question only"
    
    def test_generation_function_receives_correct_parameters(self):
        """Test that generation function receives correctly formatted parameters."""
        # Arrange
        question = "What is X?"
        sub_query_contexts = {
            "Sub-query": [Document("Doc", "Source", 0.9)]
        }
        reasoning_plan = "Use the information."
        
        received_params = {"question": None, "context": None}
        
        def mock_generation(question: str, context: str) -> str:
            received_params["question"] = question
            received_params["context"] = context
            return "Answer"
        
        # Act
        generate_with_reasoning(
            question,
            sub_query_contexts,
            reasoning_plan,
            mock_generation
        )
        
        # Assert
        # Question should be augmented
        assert received_params["question"] is not None
        assert "What is X?" in received_params["question"]
        assert "Use the information." in received_params["question"]
        
        # Context should be formatted
        assert received_params["context"] is not None
        assert "Reasoning Plan:" in received_params["context"]
        assert "Sub-query:" in received_params["context"]
