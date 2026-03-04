"""Validation script for the Sub-Query Retriever component.

This script demonstrates the retrieve_for_subqueries function with example
sub-queries and a mock retrieval function.
"""

import sys
from pathlib import Path
from typing import Dict, List

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.critic.retriever import retrieve_for_subqueries
from src.critic.models import Document


def mock_retrieval_function(query: str) -> List[Dict]:
    """Mock retrieval function that returns sample documents.
    
    Args:
        query: The query string
        
    Returns:
        List of document dictionaries with text, title, and score
    """
    # Simulate different retrieval results based on query content
    if "capital" in query.lower():
        return [
            {
                "text": "Paris is the capital and largest city of France.",
                "title": "Paris - Wikipedia",
                "score": 0.95
            },
            {
                "text": "The capital of France has been Paris since 987 AD.",
                "title": "History of Paris",
                "score": 0.88
            }
        ]
    elif "population" in query.lower():
        return [
            {
                "text": "Paris has a population of approximately 2.2 million people.",
                "title": "Paris Demographics",
                "score": 0.92
            }
        ]
    elif "eiffel" in query.lower():
        return [
            {
                "text": "The Eiffel Tower was built between 1887 and 1889.",
                "title": "Eiffel Tower History",
                "score": 0.97
            }
        ]
    else:
        # Return empty for unknown queries
        return []


def main():
    """Run validation examples."""
    print("=" * 70)
    print("Sub-Query Retriever Validation")
    print("=" * 70)
    
    # Example 1: Multiple sub-queries
    print("\n1. Testing with multiple sub-queries:")
    print("-" * 70)
    sub_queries = [
        "What is the capital of France?",
        "What is the population of Paris?",
        "When was the Eiffel Tower built?"
    ]
    
    print(f"Sub-queries: {sub_queries}")
    results = retrieve_for_subqueries(sub_queries, mock_retrieval_function)
    
    for sub_query, documents in results.items():
        print(f"\n  Sub-query: {sub_query}")
        print(f"  Retrieved {len(documents)} document(s):")
        for i, doc in enumerate(documents, 1):
            print(f"    {i}. [{doc.score:.2f}] {doc.source}")
            print(f"       {doc.text[:60]}...")
    
    # Example 2: Empty retrieval results
    print("\n\n2. Testing with query that returns no results:")
    print("-" * 70)
    empty_queries = ["What is the meaning of life?"]
    print(f"Sub-queries: {empty_queries}")
    
    results = retrieve_for_subqueries(empty_queries, mock_retrieval_function)
    for sub_query, documents in results.items():
        print(f"\n  Sub-query: {sub_query}")
        print(f"  Retrieved {len(documents)} document(s)")
    
    # Example 3: Exception handling
    print("\n\n3. Testing exception handling:")
    print("-" * 70)
    
    def failing_retrieval(query: str) -> List[Dict]:
        if "fail" in query.lower():
            raise RuntimeError("Simulated retrieval failure")
        return [{"text": "Success", "title": "Doc", "score": 0.8}]
    
    mixed_queries = ["Normal query", "This will fail", "Another normal query"]
    print(f"Sub-queries: {mixed_queries}")
    
    results = retrieve_for_subqueries(mixed_queries, failing_retrieval)
    for sub_query, documents in results.items():
        print(f"\n  Sub-query: {sub_query}")
        print(f"  Retrieved {len(documents)} document(s)")
        if documents:
            print(f"    Status: Success")
        else:
            print(f"    Status: Failed or empty")
    
    print("\n" + "=" * 70)
    print("Validation complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
