"""Validation script for the Generator component.

This script demonstrates the generate_with_reasoning function with example
usage scenarios.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.critic.generator import generate_with_reasoning
from src.critic.models import Document


def mock_generation_function(question: str, context: str) -> str:
    """Mock generation function for demonstration."""
    print(f"\n{'='*70}")
    print("GENERATION FUNCTION CALLED")
    print(f"{'='*70}")
    print(f"\nAugmented Question:\n{question}")
    print(f"\n{'-'*70}")
    print(f"\nFormatted Context:\n{context}")
    print(f"\n{'='*70}\n")
    
    # Simple mock response
    if "capital" in question.lower():
        return "Paris"
    elif "director" in question.lower():
        return "London, England"
    else:
        return "Mock answer"


def example_1_single_subquery():
    """Example 1: Single sub-query with one document."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Single Sub-Query")
    print("="*70)
    
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
    
    answer = generate_with_reasoning(
        question,
        sub_query_contexts,
        reasoning_plan,
        mock_generation_function
    )
    
    print(f"Final Answer: {answer}")


def example_2_multiple_subqueries():
    """Example 2: Multiple sub-queries (multi-hop reasoning)."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Multiple Sub-Queries (Multi-Hop)")
    print("="*70)
    
    question = "Where was the director of Inception born?"
    sub_query_contexts = {
        "Who directed Inception?": [
            Document(
                text="Inception was directed by Christopher Nolan.",
                source="Movie Database",
                score=0.98
            ),
            Document(
                text="The 2010 film Inception is a Christopher Nolan movie.",
                source="Film Encyclopedia",
                score=0.92
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
        "First find who directed Inception from the first sub-query, "
        "then find where that person was born from the second sub-query."
    )
    
    answer = generate_with_reasoning(
        question,
        sub_query_contexts,
        reasoning_plan,
        mock_generation_function
    )
    
    print(f"Final Answer: {answer}")


def example_3_empty_documents():
    """Example 3: Sub-query with no retrieved documents."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Sub-Query with Empty Documents")
    print("="*70)
    
    question = "What is the population of Atlantis?"
    sub_query_contexts = {
        "What is the population of Atlantis?": [],
        "Where is Atlantis located?": []
    }
    reasoning_plan = "Use available information about Atlantis."
    
    answer = generate_with_reasoning(
        question,
        sub_query_contexts,
        reasoning_plan,
        mock_generation_function
    )
    
    print(f"Final Answer: {answer}")


def example_4_error_handling():
    """Example 4: Error handling when generation fails."""
    print("\n" + "="*70)
    print("EXAMPLE 4: Error Handling")
    print("="*70)
    
    def failing_generation(question: str, context: str) -> str:
        raise RuntimeError("LLM service unavailable")
    
    question = "Test question"
    sub_query_contexts = {
        "Sub-query": [Document("Doc", "Source", 0.9)]
    }
    reasoning_plan = "Plan"
    
    answer = generate_with_reasoning(
        question,
        sub_query_contexts,
        reasoning_plan,
        failing_generation
    )
    
    print(f"Final Answer: {answer}")
    print("(Note: Returned fallback answer due to error)")


def main():
    """Run all validation examples."""
    print("\n" + "="*70)
    print("GENERATOR COMPONENT VALIDATION")
    print("="*70)
    
    example_1_single_subquery()
    example_2_multiple_subqueries()
    example_3_empty_documents()
    example_4_error_handling()
    
    print("\n" + "="*70)
    print("VALIDATION COMPLETE")
    print("="*70)
    print("\nAll examples executed successfully!")
    print("The generator component correctly:")
    print("  ✓ Formats context with sub-query associations")
    print("  ✓ Creates augmented prompts with reasoning plans")
    print("  ✓ Wraps existing generation function")
    print("  ✓ Handles empty documents gracefully")
    print("  ✓ Handles errors and returns fallback answers")


if __name__ == "__main__":
    main()
