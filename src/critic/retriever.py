"""Sub-query retriever for the LLM Critic Enhancement system.

This module implements the Sub-Query Retriever component that executes retrieval
for each sub-query independently using the existing RAG pipeline retrieval function,
then aggregates results in a dictionary mapping sub-queries to documents.
"""

from typing import Callable, Dict, List

from src.critic.models import Document


def retrieve_for_subqueries(
    sub_queries: List[str],
    retrieval_function: Callable[[str], List[Dict]]
) -> Dict[str, List[Document]]:
    """Retrieve documents for each sub-query independently.
    
    This function iterates through sub-queries and calls the existing retrieval
    function for each one, aggregating results in a dictionary that maps each
    sub-query to its retrieved documents. It handles empty retrieval results
    and exceptions gracefully.
    
    Args:
        sub_queries: List of sub-questions to retrieve documents for
        retrieval_function: Existing RAG pipeline retrieval function that takes
                          a query string and returns a list of dictionaries with
                          'text', 'title', 'score', and optionally 'doc_id' keys
    
    Returns:
        Dictionary mapping each sub-query to a list of Document objects.
        If retrieval fails or returns empty results for a sub-query, that
        sub-query will map to an empty list.
    
    Example:
        >>> def mock_retrieval(query: str) -> List[Dict]:
        ...     return [{"text": "doc1", "title": "Title", "score": 0.9}]
        >>> sub_queries = ["What is X?", "Where is Y?"]
        >>> results = retrieve_for_subqueries(sub_queries, mock_retrieval)
        >>> len(results)
        2
        >>> "What is X?" in results
        True
    """
    results: Dict[str, List[Document]] = {}
    
    for sub_query in sub_queries:
        try:
            # Call the existing retrieval function
            raw_documents = retrieval_function(sub_query)
            
            # Handle empty retrieval results
            if not raw_documents:
                results[sub_query] = []
                continue
            
            # Convert raw dictionaries to Document objects
            documents = []
            for raw_doc in raw_documents:
                # Extract fields with defaults for missing keys
                text = raw_doc.get("text", "")
                # Use title as source, fallback to doc_id or empty string
                source = raw_doc.get("title", raw_doc.get("doc_id", ""))
                score = raw_doc.get("score", 0.0)
                
                documents.append(Document(
                    text=text,
                    source=source,
                    score=score
                ))
            
            results[sub_query] = documents
            
        except Exception as e:
            # Handle retrieval function exceptions gracefully
            # Log the error but continue processing other sub-queries
            print(f"Warning: Retrieval failed for sub-query '{sub_query}': {e}")
            results[sub_query] = []
    
    return results
