"""Example script showing how to run HotpotQA evaluation with the Critic System.

This example demonstrates how to integrate the evaluation script with an existing
RAG pipeline and run the full 300-question HotpotQA evaluation.

Requirements: 7.1, 7.2, 7.3
"""

import json
import logging
from pathlib import Path

from src.critic.evaluate_hotpotqa import evaluate_hotpotqa


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def run_evaluation_example():
    """
    Example of running HotpotQA evaluation with the Critic System.
    
    This function demonstrates the complete evaluation workflow:
    1. Load your RAG pipeline components (retriever and llm)
    2. Optionally load baseline answers from previous evaluation
    3. Run evaluation on 300-question dataset
    4. Save results and metrics
    
    Note: You need to replace the placeholder retriever and llm with your
    actual RAG pipeline components.
    """
    
    print("=" * 80)
    print("HOTPOTQA EVALUATION EXAMPLE")
    print("=" * 80)
    print()
    
    # =========================================================================
    # STEP 1: Load your RAG pipeline components
    # =========================================================================
    print("Step 1: Loading RAG pipeline components...")
    print()
    
    # TODO: Replace these with your actual RAG pipeline components
    # Example for LangChain:
    #
    # from langchain_community.vectorstores import Chroma
    # from langchain_openai import OpenAIEmbeddings, ChatOpenAI
    # from langchain.retrievers import ContextualCompressionRetriever
    # from langchain.retrievers.document_compressors import CohereRerank
    #
    # # Load vector store
    # vectorstore = Chroma(
    #     persist_directory="path/to/vectorstore",
    #     embedding_function=OpenAIEmbeddings()
    # )
    #
    # # Create retriever with reranking
    # base_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    # compressor = CohereRerank(top_n=5)
    # retriever = ContextualCompressionRetriever(
    #     base_compressor=compressor,
    #     base_retriever=base_retriever
    # )
    #
    # # Create LLM
    # llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    
    # For this example, we'll use mock components
    print("WARNING: Using mock components for demonstration")
    print("Replace these with your actual RAG pipeline components")
    print()
    
    # Mock retriever that returns empty results
    def mock_retriever(query: str):
        return []
    
    # Mock LLM that returns a placeholder answer
    class MockLLM:
        def invoke(self, prompt: str):
            return "I do not know."
    
    retriever = mock_retriever
    llm = MockLLM()
    
    # =========================================================================
    # STEP 2: Load baseline answers (optional)
    # =========================================================================
    print("Step 2: Loading baseline answers (optional)...")
    print()
    
    # If you have baseline answers from a previous evaluation, load them here
    # This allows you to compare the critic system's performance against the baseline
    baseline_answers = None
    baseline_answers_path = Path("evaluation_results/baseline_answers.json")
    
    if baseline_answers_path.exists():
        print(f"Loading baseline answers from: {baseline_answers_path}")
        with open(baseline_answers_path, 'r') as f:
            baseline_answers = json.load(f)
        print(f"Loaded {len(baseline_answers)} baseline answers")
    else:
        print("No baseline answers found, will generate during evaluation")
    print()
    
    # =========================================================================
    # STEP 3: Run evaluation
    # =========================================================================
    print("Step 3: Running evaluation on 300-question dataset...")
    print()
    
    # Path to the stratified 300-question HotpotQA dataset
    dataset_path = "Dataset/hotpot_eval_300.json"
    
    # Output directory for results
    output_dir = "evaluation_results"
    
    # Run evaluation
    metrics = evaluate_hotpotqa(
        dataset_path=dataset_path,
        retriever=retriever,
        llm=llm,
        baseline_answers=baseline_answers,
        output_dir=output_dir,
        enable_logging=True
    )
    
    # =========================================================================
    # STEP 4: Display results
    # =========================================================================
    print()
    print("=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print()
    print(f"Total questions:      {metrics.total_questions}")
    print(f"Correct answers:      {metrics.correct_answers}")
    print(f"Accuracy:             {metrics.accuracy:.2f}%")
    print(f"Baseline accuracy:    {metrics.baseline_accuracy:.2f}%")
    print(f"Improvement:          {metrics.accuracy_improvement:+.2f}%")
    print()
    print(f"Processing time:      {metrics.processing_time:.2f} seconds")
    print(f"Avg time per question: {metrics.processing_time / metrics.total_questions:.2f} seconds")
    print()
    print(f"Results saved to: {output_dir}")
    print()
    
    # =========================================================================
    # STEP 5: Analyze results (optional)
    # =========================================================================
    print("=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    print()
    
    if metrics.accuracy > metrics.baseline_accuracy:
        print(f"✓ SUCCESS: Critic system improved accuracy by {metrics.accuracy_improvement:.2f}%")
    elif metrics.accuracy == metrics.baseline_accuracy:
        print(f"= NEUTRAL: Critic system maintained baseline accuracy")
    else:
        print(f"✗ REGRESSION: Critic system decreased accuracy by {metrics.accuracy_improvement:.2f}%")
    
    print()
    print("Next steps:")
    print("1. Review detailed results in evaluation_results/detailed_results_*.json")
    print("2. Analyze incorrect answers to identify failure patterns")
    print("3. Adjust critic system parameters if needed")
    print("4. Re-run evaluation to measure improvements")
    print()


def run_small_scale_test():
    """
    Run a small-scale test on a subset of questions for quick validation.
    
    This is useful for:
    - Testing the evaluation pipeline before running the full 300 questions
    - Quick iteration during development
    - Debugging issues with specific questions
    """
    
    print("=" * 80)
    print("SMALL-SCALE TEST (10 questions)")
    print("=" * 80)
    print()
    
    # Load full dataset
    dataset_path = Path("Dataset/hotpot_eval_300.json")
    with open(dataset_path, 'r') as f:
        full_dataset = json.load(f)
    
    # Create small test dataset (first 10 questions)
    test_dataset = full_dataset[:10]
    
    # Save test dataset
    test_dataset_path = Path("Dataset/hotpot_eval_test_10.json")
    with open(test_dataset_path, 'w') as f:
        json.dump(test_dataset, f, indent=2)
    
    print(f"Created test dataset with {len(test_dataset)} questions")
    print(f"Saved to: {test_dataset_path}")
    print()
    
    # Mock components (replace with your actual components)
    def mock_retriever(query: str):
        return []
    
    class MockLLM:
        def invoke(self, prompt: str):
            return "I do not know."
    
    retriever = mock_retriever
    llm = MockLLM()
    
    # Run evaluation on test dataset
    print("Running evaluation on test dataset...")
    print()
    
    metrics = evaluate_hotpotqa(
        dataset_path=str(test_dataset_path),
        retriever=retriever,
        llm=llm,
        output_dir="evaluation_results/test",
        enable_logging=True
    )
    
    print()
    print("=" * 80)
    print("TEST RESULTS")
    print("=" * 80)
    print()
    print(f"Accuracy: {metrics.accuracy:.2f}%")
    print(f"Correct: {metrics.correct_answers}/{metrics.total_questions}")
    print()
    print("If this looks good, run the full evaluation with run_evaluation_example()")
    print()


if __name__ == "__main__":
    # Uncomment the function you want to run:
    
    # Run small-scale test first
    # run_small_scale_test()
    
    # Run full evaluation
    # run_evaluation_example()
    
    print("=" * 80)
    print("EVALUATION EXAMPLE SCRIPT")
    print("=" * 80)
    print()
    print("This script provides examples for running HotpotQA evaluation.")
    print()
    print("To use:")
    print("1. Edit this file to add your RAG pipeline components (retriever and llm)")
    print("2. Uncomment one of the functions at the bottom:")
    print("   - run_small_scale_test() for quick testing (10 questions)")
    print("   - run_evaluation_example() for full evaluation (300 questions)")
    print("3. Run: python src/critic/evaluation_example.py")
    print()
    print("Or import and use in your own script:")
    print("    from src.critic.evaluation_example import run_evaluation_example")
    print("    run_evaluation_example()")
    print()
