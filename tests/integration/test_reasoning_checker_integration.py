"""Integration tests for the Reasoning Checker component.

Tests the check_reasoning function with realistic scenarios to verify
end-to-end functionality including LLM integration, decision logic,
and error handling.
"""

import pytest
from unittest.mock import patch

from src.critic import check_reasoning
from src.critic.models import Document


class TestReasoningCheckerIntegration:
    """Integration test suite for check_reasoning function."""
    
    def test_realistic_pass_scenario(self):
        """Test a realistic scenario where validation passes."""
        # Simulate a realistic LLM response for a valid answer
        mock_llm_response = """{
  "facts_retrieved": true,
  "missing_facts": [],
  "reasoning_valid": true,
  "reasoning_explanation": "The documents contain information about Christopher Nolan directing Inception and his birthplace in London. The answer correctly combines these facts following the reasoning plan."
}"""
        
        with patch('src.critic.reasoning_checker.call_llm', return_value=mock_llm_response):
            result = check_reasoning(
                question="Where was the director of Inception born?",
                sub_queries=[
                    "Who directed Inception?",
                    "Where was Christopher Nolan born?"
                ],
                sub_query_contexts={
                    "Who directed Inception?": [
                        Document(
                            text="Inception is a 2010 science fiction action film written and directed by Christopher Nolan.",
                            source="wikipedia_inception",
                            score=0.95
                        )
                    ],
                    "Where was Christopher Nolan born?": [
                        Document(
                            text="Christopher Edward Nolan was born on 30 July 1970 in Westminster, London, England.",
                            source="wikipedia_nolan",
                            score=0.92
                        )
                    ]
                },
                reasoning_plan="First identify who directed Inception, then find where that director was born.",
                answer="Christopher Nolan, the director of Inception, was born in London, England."
            )
        
        assert result.action == "PASS"
        assert result.missing_facts == []
        assert len(result.reasoning_explanation) > 0
    
    def test_realistic_retry_retrieval_scenario(self):
        """Test a realistic scenario where retrieval needs to be retried."""
        # Simulate LLM response indicating missing facts
        mock_llm_response = """{
  "facts_retrieved": false,
  "missing_facts": ["Where was Christopher Nolan born?"],
  "reasoning_valid": false,
  "reasoning_explanation": "The documents contain information about who directed Inception (Christopher Nolan), but there is no information about where Christopher Nolan was born. Cannot complete the reasoning chain without this information."
}"""
        
        with patch('src.critic.reasoning_checker.call_llm', return_value=mock_llm_response):
            result = check_reasoning(
                question="Where was the director of Inception born?",
                sub_queries=[
                    "Who directed Inception?",
                    "Where was Christopher Nolan born?"
                ],
                sub_query_contexts={
                    "Who directed Inception?": [
                        Document(
                            text="Inception is a 2010 science fiction action film written and directed by Christopher Nolan.",
                            source="wikipedia_inception",
                            score=0.95
                        )
                    ],
                    "Where was Christopher Nolan born?": []  # No documents retrieved
                },
                reasoning_plan="First identify who directed Inception, then find where that director was born.",
                answer="I do not know."
            )
        
        assert result.action == "RETRY_RETRIEVAL"
        assert "Where was Christopher Nolan born?" in result.missing_facts
        assert "born" in result.reasoning_explanation.lower()
    
    def test_realistic_regenerate_scenario(self):
        """Test a realistic scenario where answer needs to be regenerated."""
        # Simulate LLM response indicating invalid reasoning
        mock_llm_response = """{
  "facts_retrieved": true,
  "missing_facts": [],
  "reasoning_valid": false,
  "reasoning_explanation": "All necessary facts are present in the documents, but the generated answer does not follow the reasoning plan. The answer provides information about the film's release date instead of the director's birthplace."
}"""
        
        with patch('src.critic.reasoning_checker.call_llm', return_value=mock_llm_response):
            result = check_reasoning(
                question="Where was the director of Inception born?",
                sub_queries=[
                    "Who directed Inception?",
                    "Where was Christopher Nolan born?"
                ],
                sub_query_contexts={
                    "Who directed Inception?": [
                        Document(
                            text="Inception is a 2010 science fiction action film written and directed by Christopher Nolan.",
                            source="wikipedia_inception",
                            score=0.95
                        )
                    ],
                    "Where was Christopher Nolan born?": [
                        Document(
                            text="Christopher Edward Nolan was born on 30 July 1970 in Westminster, London, England.",
                            source="wikipedia_nolan",
                            score=0.92
                        )
                    ]
                },
                reasoning_plan="First identify who directed Inception, then find where that director was born.",
                answer="Inception was released in 2010."  # Wrong answer
            )
        
        assert result.action == "REGENERATE"
        assert result.missing_facts == []
        assert "reasoning" in result.reasoning_explanation.lower()
    
    def test_multiple_sub_queries_with_partial_retrieval(self):
        """Test scenario with multiple sub-queries where some have documents and some don't."""
        mock_llm_response = """{
  "facts_retrieved": false,
  "missing_facts": ["What is the population of that city?"],
  "reasoning_valid": false,
  "reasoning_explanation": "Found the capital city (Paris) but missing population information."
}"""
        
        with patch('src.critic.reasoning_checker.call_llm', return_value=mock_llm_response):
            result = check_reasoning(
                question="What is the population of the capital of France?",
                sub_queries=[
                    "What is the capital of France?",
                    "What is the population of that city?"
                ],
                sub_query_contexts={
                    "What is the capital of France?": [
                        Document(
                            text="Paris is the capital and most populous city of France.",
                            source="wikipedia_paris",
                            score=0.98
                        )
                    ],
                    "What is the population of that city?": []  # Missing
                },
                reasoning_plan="Find the capital city, then find its population.",
                answer="I do not know."
            )
        
        assert result.action == "RETRY_RETRIEVAL"
        assert "What is the population of that city?" in result.missing_facts
    
    def test_complex_multi_hop_question(self):
        """Test a complex multi-hop question with three sub-queries."""
        mock_llm_response = """{
  "facts_retrieved": true,
  "missing_facts": [],
  "reasoning_valid": true,
  "reasoning_explanation": "All three pieces of information are present: the author (J.K. Rowling), her birthplace (England), and the capital (London). The answer correctly combines these facts."
}"""
        
        with patch('src.critic.reasoning_checker.call_llm', return_value=mock_llm_response):
            result = check_reasoning(
                question="What is the capital of the country where the author of Harry Potter was born?",
                sub_queries=[
                    "Who is the author of Harry Potter?",
                    "Where was J.K. Rowling born?",
                    "What is the capital of England?"
                ],
                sub_query_contexts={
                    "Who is the author of Harry Potter?": [
                        Document(
                            text="Harry Potter is a series of fantasy novels written by British author J.K. Rowling.",
                            source="wikipedia_hp",
                            score=0.99
                        )
                    ],
                    "Where was J.K. Rowling born?": [
                        Document(
                            text="Joanne Rowling was born on 31 July 1965 in Yate, Gloucestershire, England.",
                            source="wikipedia_jkr",
                            score=0.94
                        )
                    ],
                    "What is the capital of England?": [
                        Document(
                            text="London is the capital and largest city of England and the United Kingdom.",
                            source="wikipedia_london",
                            score=0.97
                        )
                    ]
                },
                reasoning_plan="Find the author, then their birthplace country, then that country's capital.",
                answer="London is the capital of England, where J.K. Rowling was born."
            )
        
        assert result.action == "PASS"
        assert result.missing_facts == []
