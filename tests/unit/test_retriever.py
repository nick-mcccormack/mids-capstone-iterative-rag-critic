"""Unit tests for the Sub-Query Retriever component.

Tests cover:
- Retrieval for single sub-query
- Retrieval for multiple sub-queries
- Empty retrieval results handling
- Retrieval function exceptions handling
"""

import pytest
from typing import Dict, List

from src.critic.retriever import retrieve_for_subqueries
from src.critic.models import Document


class TestRetrieveForSubqueries:
    """Test suite for retrieve_for_subqueries function."""
    
    def test_single_subquery_retrieval(self):
        """Test retrieval for a single sub-query."""
        # Arrange
        sub_queries = ["What is the capital of France?"]
        
        def mock_retrieval(query: str) -> List[Dict]:
            return [
                {
                    "text": "Paris is the capital of France.",
                    "title": "France Geography",
                    "score": 0.95
                }
            ]
        
        # Act
        results = retrieve_for_subqueries(sub_queries, mock_retrieval)
        
        # Assert
        assert len(results) == 1
        assert sub_queries[0] in results
        assert len(results[sub_queries[0]]) == 1
        
        doc = results[sub_queries[0]][0]
        assert isinstance(doc, Document)
        assert doc.text == "Paris is the capital of France."
        assert doc.source == "France Geography"
        assert doc.score == 0.95
    
    def test_multiple_subqueries_retrieval(self):
        """Test retrieval for multiple sub-queries."""
        # Arrange
        sub_queries = [
            "What is the capital of France?",
            "What is the population of Paris?",
            "When was the Eiffel Tower built?"
        ]
        
        def mock_retrieval(query: str) -> List[Dict]:
            # Return different documents based on query
            if "capital" in query:
                return [
                    {"text": "Paris is the capital.", "title": "France", "score": 0.9}
                ]
            elif "population" in query:
                return [
                    {"text": "Paris has 2.2M people.", "title": "Demographics", "score": 0.85}
                ]
            elif "Eiffel" in query:
                return [
                    {"text": "Built in 1889.", "title": "History", "score": 0.92}
                ]
            return []
        
        # Act
        results = retrieve_for_subqueries(sub_queries, mock_retrieval)
        
        # Assert
        assert len(results) == 3
        for sub_query in sub_queries:
            assert sub_query in results
            assert len(results[sub_query]) >= 1
    
    def test_empty_retrieval_results(self):
        """Test handling of empty retrieval results."""
        # Arrange
        sub_queries = ["What is X?", "What is Y?"]
        
        def mock_retrieval(query: str) -> List[Dict]:
            # Return empty list for all queries
            return []
        
        # Act
        results = retrieve_for_subqueries(sub_queries, mock_retrieval)
        
        # Assert
        assert len(results) == 2
        for sub_query in sub_queries:
            assert sub_query in results
            assert results[sub_query] == []
    
    def test_retrieval_function_exception(self):
        """Test handling of retrieval function exceptions."""
        # Arrange
        sub_queries = ["Query 1", "Query 2", "Query 3"]
        
        def mock_retrieval(query: str) -> List[Dict]:
            if "Query 2" in query:
                raise RuntimeError("Retrieval service unavailable")
            return [
                {"text": f"Result for {query}", "title": "Doc", "score": 0.8}
            ]
        
        # Act
        results = retrieve_for_subqueries(sub_queries, mock_retrieval)
        
        # Assert
        assert len(results) == 3
        # Query 1 and 3 should succeed
        assert len(results["Query 1"]) == 1
        assert len(results["Query 3"]) == 1
        # Query 2 should have empty results due to exception
        assert results["Query 2"] == []
    
    def test_missing_document_fields(self):
        """Test handling of documents with missing fields."""
        # Arrange
        sub_queries = ["Test query"]
        
        def mock_retrieval(query: str) -> List[Dict]:
            return [
                {"text": "Document with all fields", "title": "Title", "score": 0.9},
                {"text": "Document missing title"},  # Missing title and score
                {"title": "Title only", "score": 0.5},  # Missing text
                {},  # Empty document
            ]
        
        # Act
        results = retrieve_for_subqueries(sub_queries, mock_retrieval)
        
        # Assert
        assert len(results) == 1
        docs = results["Test query"]
        assert len(docs) == 4
        
        # First document has all fields
        assert docs[0].text == "Document with all fields"
        assert docs[0].source == "Title"
        assert docs[0].score == 0.9
        
        # Second document uses defaults for missing fields
        assert docs[1].text == "Document missing title"
        assert docs[1].source == ""
        assert docs[1].score == 0.0
        
        # Third document has empty text
        assert docs[2].text == ""
        assert docs[2].source == "Title only"
        assert docs[2].score == 0.5
        
        # Fourth document uses all defaults
        assert docs[3].text == ""
        assert docs[3].source == ""
        assert docs[3].score == 0.0
    
    def test_doc_id_fallback_for_source(self):
        """Test that doc_id is used as source when title is missing."""
        # Arrange
        sub_queries = ["Test query"]
        
        def mock_retrieval(query: str) -> List[Dict]:
            return [
                {"text": "Doc with doc_id", "doc_id": "doc123", "score": 0.8}
            ]
        
        # Act
        results = retrieve_for_subqueries(sub_queries, mock_retrieval)
        
        # Assert
        docs = results["Test query"]
        assert len(docs) == 1
        assert docs[0].source == "doc123"
    
    def test_multiple_documents_per_subquery(self):
        """Test retrieval returning multiple documents per sub-query."""
        # Arrange
        sub_queries = ["Multi-doc query"]
        
        def mock_retrieval(query: str) -> List[Dict]:
            return [
                {"text": "Doc 1", "title": "Title 1", "score": 0.95},
                {"text": "Doc 2", "title": "Title 2", "score": 0.90},
                {"text": "Doc 3", "title": "Title 3", "score": 0.85},
            ]
        
        # Act
        results = retrieve_for_subqueries(sub_queries, mock_retrieval)
        
        # Assert
        docs = results["Multi-doc query"]
        assert len(docs) == 3
        assert docs[0].score == 0.95
        assert docs[1].score == 0.90
        assert docs[2].score == 0.85
    
    def test_empty_subqueries_list(self):
        """Test handling of empty sub-queries list."""
        # Arrange
        sub_queries = []
        
        def mock_retrieval(query: str) -> List[Dict]:
            return [{"text": "Should not be called", "title": "N/A", "score": 1.0}]
        
        # Act
        results = retrieve_for_subqueries(sub_queries, mock_retrieval)
        
        # Assert
        assert results == {}
    
    def test_partial_failure_continues_processing(self):
        """Test that failure on one sub-query doesn't stop processing others."""
        # Arrange
        sub_queries = ["Query A", "Query B", "Query C"]
        call_count = {"count": 0}
        
        def mock_retrieval(query: str) -> List[Dict]:
            call_count["count"] += 1
            if query == "Query B":
                raise ValueError("Simulated error")
            return [{"text": f"Result for {query}", "title": "Doc", "score": 0.8}]
        
        # Act
        results = retrieve_for_subqueries(sub_queries, mock_retrieval)
        
        # Assert
        # All three queries should be attempted
        assert call_count["count"] == 3
        # Query A and C should succeed
        assert len(results["Query A"]) == 1
        assert len(results["Query C"]) == 1
        # Query B should have empty results
        assert results["Query B"] == []
