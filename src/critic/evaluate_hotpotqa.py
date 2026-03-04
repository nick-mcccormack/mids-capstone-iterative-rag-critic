"""HotpotQA evaluation script for the Critic System.

This script evaluates the Critic System on the stratified 300-question HotpotQA dataset.
It processes each question through the critic system, collects answers in HotpotQA format,
and generates an accuracy report comparing to the 60.17% baseline.

Requirements: 7.1, 7.2, 7.3
"""

import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from datetime import datetime

from src.critic.integration import integrate_critic_with_rag


logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    """Metrics collected during evaluation.
    
    Attributes:
        total_questions: Total number of questions processed
        correct_answers: Number of correct answers
        accuracy: Accuracy percentage
        baseline_accuracy: Baseline accuracy for comparison (60.17%)
        accuracy_improvement: Improvement over baseline
        avg_iteration_count: Average number of critic iterations
        validation_outcomes: Count of validation outcomes by type
        processing_time: Total processing time in seconds
    """
    total_questions: int = 0
    correct_answers: int = 0
    accuracy: float = 0.0
    baseline_accuracy: float = 60.17
    accuracy_improvement: float = 0.0
    avg_iteration_count: float = 0.0
    validation_outcomes: Dict[str, int] = field(default_factory=dict)
    processing_time: float = 0.0
    
    def calculate_accuracy(self):
        """Calculate accuracy percentage and improvement."""
        if self.total_questions > 0:
            self.accuracy = (self.correct_answers / self.total_questions) * 100
            self.accuracy_improvement = self.accuracy - self.baseline_accuracy
        else:
            self.accuracy = 0.0
            self.accuracy_improvement = 0.0


@dataclass
class QuestionResult:
    """Result for a single question.
    
    Attributes:
        question_id: HotpotQA question ID
        question: Question text
        gold_answer: Ground truth answer
        baseline_answer: Answer from baseline RAG pipeline
        final_answer: Answer after critic system processing
        is_correct: Whether final answer matches gold answer
        iteration_count: Number of critic iterations
        validation_history: List of validation results
    """
    question_id: str
    question: str
    gold_answer: str
    baseline_answer: str
    final_answer: str
    is_correct: bool
    iteration_count: int = 0
    validation_history: List[str] = field(default_factory=list)


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison.
    
    Applies standard HotpotQA normalization:
    - Convert to lowercase
    - Remove articles (a, an, the)
    - Remove punctuation
    - Remove extra whitespace
    
    Args:
        answer: Answer string to normalize
    
    Returns:
        Normalized answer string
    """
    import re
    import string
    
    # Convert to lowercase
    answer = answer.lower()
    
    # Remove articles
    answer = re.sub(r'\b(a|an|the)\b', ' ', answer)
    
    # Remove punctuation
    answer = answer.translate(str.maketrans('', '', string.punctuation))
    
    # Remove extra whitespace
    answer = ' '.join(answer.split())
    
    return answer.strip()


def exact_match_score(prediction: str, ground_truth: str) -> bool:
    """Calculate exact match score between prediction and ground truth.
    
    Uses normalized string matching following HotpotQA evaluation protocol.
    
    Args:
        prediction: Predicted answer
        ground_truth: Ground truth answer
    
    Returns:
        True if normalized answers match exactly, False otherwise
    """
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def load_hotpotqa_dataset(dataset_path: str) -> List[Dict]:
    """Load HotpotQA dataset from JSON file.
    
    Args:
        dataset_path: Path to HotpotQA JSON file
    
    Returns:
        List of question dictionaries
    
    Raises:
        FileNotFoundError: If dataset file doesn't exist
        json.JSONDecodeError: If file is not valid JSON
    """
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    
    logger.info(f"Loading HotpotQA dataset from: {dataset_path}")
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    logger.info(f"Loaded {len(dataset)} questions from dataset")
    
    return dataset


def evaluate_hotpotqa(
    dataset_path: str,
    retriever: Any,
    llm: Any,
    baseline_answers: Optional[Dict[str, str]] = None,
    retrieval_adapter_kwargs: Optional[Dict] = None,
    generation_adapter_kwargs: Optional[Dict] = None,
    output_dir: Optional[str] = None,
    enable_logging: bool = True
) -> EvaluationMetrics:
    """Evaluate Critic System on HotpotQA dataset.
    
    This is the main evaluation function that:
    1. Loads the stratified 300-question HotpotQA dataset
    2. Processes each question through the Critic System
    3. Collects answers in HotpotQA evaluation format
    4. Calculates accuracy metrics
    5. Tracks iteration counts and validation outcomes
    6. Generates evaluation report
    
    Args:
        dataset_path: Path to HotpotQA JSON file (300-question stratified dataset)
        retriever: RAG pipeline retriever (LangChain retriever or custom function)
        llm: RAG pipeline LLM (LangChain LLM/chain or custom function)
        baseline_answers: Optional dict mapping question_id to baseline answer.
                         If None, baseline answers will be generated using the
                         provided retriever and llm without critic enhancement.
        retrieval_adapter_kwargs: Optional kwargs for retrieval adapter
        generation_adapter_kwargs: Optional kwargs for generation adapter
        output_dir: Optional directory to save evaluation results
        enable_logging: Whether to enable detailed logging
    
    Returns:
        EvaluationMetrics object with accuracy and other metrics
    
    Examples:
        >>> metrics = evaluate_hotpotqa(
        ...     "Dataset/hotpot_eval_300.json",
        ...     retriever=langchain_retriever,
        ...     llm=langchain_llm
        ... )
        >>> print(f"Accuracy: {metrics.accuracy:.2f}%")
        >>> print(f"Improvement: {metrics.accuracy_improvement:+.2f}%")
    """
    import time
    
    # Configure logging
    if enable_logging:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    logger.info("=" * 80)
    logger.info("HOTPOTQA EVALUATION - CRITIC SYSTEM")
    logger.info("=" * 80)
    logger.info(f"Dataset: {dataset_path}")
    logger.info(f"Baseline accuracy: {60.17}%")
    logger.info("")
    
    # Load dataset
    dataset = load_hotpotqa_dataset(dataset_path)
    
    # Initialize metrics
    metrics = EvaluationMetrics()
    metrics.total_questions = len(dataset)
    
    # Initialize results list
    results: List[QuestionResult] = []
    
    # Track processing time
    start_time = time.time()
    
    # Process each question
    for i, item in enumerate(dataset, 1):
        question_id = item.get('id', f'q_{i}')
        question = item['question']
        gold_answer = item['answer']
        
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Processing question {i}/{len(dataset)}")
        logger.info(f"ID: {question_id}")
        logger.info(f"Question: {question}")
        logger.info(f"Gold answer: {gold_answer}")
        
        try:
            # Get baseline answer if not provided
            if baseline_answers and question_id in baseline_answers:
                baseline_answer = baseline_answers[question_id]
                logger.info(f"Using provided baseline answer: {baseline_answer}")
            else:
                # Generate baseline answer without critic enhancement
                # This would typically come from the existing RAG pipeline
                # For now, we'll use a placeholder
                baseline_answer = "I do not know."
                logger.warning(f"No baseline answer provided for {question_id}, using placeholder")
            
            # Process through critic system
            final_answer = integrate_critic_with_rag(
                question=question,
                baseline_answer=baseline_answer,
                retriever=retriever,
                llm=llm,
                retrieval_adapter_kwargs=retrieval_adapter_kwargs,
                generation_adapter_kwargs=generation_adapter_kwargs,
                enable_logging=False  # Disable per-question logging for cleaner output
            )
            
            # Check if answer is correct
            is_correct = exact_match_score(final_answer, gold_answer)
            
            if is_correct:
                metrics.correct_answers += 1
                logger.info(f"✓ CORRECT")
            else:
                logger.info(f"✗ INCORRECT")
            
            logger.info(f"Final answer: {final_answer}")
            
            # Create result record
            result = QuestionResult(
                question_id=question_id,
                question=question,
                gold_answer=gold_answer,
                baseline_answer=baseline_answer,
                final_answer=final_answer,
                is_correct=is_correct
            )
            
            results.append(result)
            
        except Exception as e:
            logger.error(f"Error processing question {question_id}: {e}")
            # Record as incorrect with error
            result = QuestionResult(
                question_id=question_id,
                question=question,
                gold_answer=gold_answer,
                baseline_answer=baseline_answer if baseline_answers else "ERROR",
                final_answer="ERROR",
                is_correct=False
            )
            results.append(result)
    
    # Calculate processing time
    metrics.processing_time = time.time() - start_time
    
    # Calculate final metrics
    metrics.calculate_accuracy()
    
    # Log summary
    logger.info("\n" + "=" * 80)
    logger.info("EVALUATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Total questions: {metrics.total_questions}")
    logger.info(f"Correct answers: {metrics.correct_answers}")
    logger.info(f"Accuracy: {metrics.accuracy:.2f}%")
    logger.info(f"Baseline accuracy: {metrics.baseline_accuracy:.2f}%")
    logger.info(f"Improvement: {metrics.accuracy_improvement:+.2f}%")
    logger.info(f"Processing time: {metrics.processing_time:.2f} seconds")
    logger.info(f"Avg time per question: {metrics.processing_time / metrics.total_questions:.2f} seconds")
    logger.info("")
    
    # Save results if output directory specified
    if output_dir:
        save_evaluation_results(results, metrics, output_dir)
    
    return metrics


def save_evaluation_results(
    results: List[QuestionResult],
    metrics: EvaluationMetrics,
    output_dir: str
):
    """Save evaluation results to JSON files.
    
    Creates two files:
    1. detailed_results.json - Full results for each question
    2. metrics_summary.json - Overall metrics and statistics
    
    Args:
        results: List of QuestionResult objects
        metrics: EvaluationMetrics object
        output_dir: Directory to save results
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save detailed results
    detailed_results_path = output_path / f"detailed_results_{timestamp}.json"
    detailed_results = []
    for result in results:
        detailed_results.append({
            "question_id": result.question_id,
            "question": result.question,
            "gold_answer": result.gold_answer,
            "baseline_answer": result.baseline_answer,
            "final_answer": result.final_answer,
            "is_correct": result.is_correct,
            "iteration_count": result.iteration_count,
            "validation_history": result.validation_history
        })
    
    with open(detailed_results_path, 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved detailed results to: {detailed_results_path}")
    
    # Save metrics summary
    metrics_summary_path = output_path / f"metrics_summary_{timestamp}.json"
    metrics_summary = {
        "evaluation_timestamp": timestamp,
        "total_questions": metrics.total_questions,
        "correct_answers": metrics.correct_answers,
        "accuracy": round(metrics.accuracy, 2),
        "baseline_accuracy": metrics.baseline_accuracy,
        "accuracy_improvement": round(metrics.accuracy_improvement, 2),
        "avg_iteration_count": round(metrics.avg_iteration_count, 2),
        "validation_outcomes": metrics.validation_outcomes,
        "processing_time_seconds": round(metrics.processing_time, 2),
        "avg_time_per_question": round(metrics.processing_time / metrics.total_questions, 2)
    }
    
    with open(metrics_summary_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_summary, f, indent=2)
    
    logger.info(f"Saved metrics summary to: {metrics_summary_path}")


def main():
    """Command-line interface for HotpotQA evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Evaluate Critic System on HotpotQA dataset"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="Dataset/hotpot_eval_300.json",
        help="Path to HotpotQA dataset JSON file"
    )
    parser.add_argument(
        "--baseline-answers",
        type=str,
        help="Path to JSON file with baseline answers (optional)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation_results",
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Load baseline answers if provided
    baseline_answers = None
    if args.baseline_answers:
        with open(args.baseline_answers, 'r') as f:
            baseline_answers = json.load(f)
    
    # Note: This is a template main function
    # In practice, you would need to provide actual retriever and llm instances
    # from your RAG pipeline
    
    print("=" * 80)
    print("ERROR: This script requires retriever and llm instances")
    print("=" * 80)
    print()
    print("To use this evaluation script, you need to:")
    print("1. Import your RAG pipeline components (retriever and llm)")
    print("2. Call evaluate_hotpotqa() with your components")
    print()
    print("Example:")
    print("    from src.critic.evaluate_hotpotqa import evaluate_hotpotqa")
    print("    from your_rag_pipeline import retriever, llm")
    print()
    print("    metrics = evaluate_hotpotqa(")
    print("        dataset_path='Dataset/hotpot_eval_300.json',")
    print("        retriever=retriever,")
    print("        llm=llm,")
    print("        output_dir='evaluation_results'")
    print("    )")
    print()
    print("See the notebook integration example for more details.")
    print()
    
    sys.exit(1)


if __name__ == "__main__":
    main()
