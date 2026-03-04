"""Unit tests for the Critic System orchestrator.

Tests the main control flow loop that integrates all components:
- Answer Validator for initial check
- Query Decomposer for query breakdown
- Sub-Query Retriever for context gathering
- Generator for answer production
- Reasoning Checker for validation
- Iteration counter (max 2 iterations)
- Termination conditions
"""

import pytest
from unittest.mock import Mock, patch, call

from src.critic.critic_system import run_critic_system
from src.critic.models import (
    ValidationResult,
    DecompositionResult,
    Document,
    ReasoningCheckResult,
)


class TestCriticSystemBasicFlow:
    """Test basic control flow scenarios."""
    
    def test_initial_answer_passes_validation(self):
        """Test that system returns initial answer if it passes validation."""
        # Mock functions
        mock_retrieval = Mock()
        mock_generation = Mock()
        
        # Mock validator to pass on initial answer
        with patch('src.critic.critic_system.validate_answer') as mock_validate:
            mock_validate.return_value = ValidationResult(passed=True, triggered_rule=None)
            
            result = run_critic_system(
                question="What is the capital of France?",
                initial_answer="Paris",
                retrieval_function=mock_retrieval,
                generation_function=mock_generation
            )
            
            # Should return initial answer without calling other components
            assert result == "Paris"
            mock_validate.assert_called_once()
            mock_retrieval.assert_not_called()
            mock_generation.assert_not_called()
    
    def test_full_critic_loop_passes_on_first_iteration(self):
        """Test critic loop that passes validation on first iteration."""
        # Mock retrieval function
        def mock_retrieval(query: str):
            return [
                {"text": "Paris is the capital of France", "title": "doc1", "score": 0.9}
            ]
        
        # Mock generation function
        def mock_generation(question: str, context: str):
            return "Paris"
        
        # Mock validator: fail initial, pass after critic loop
        validation_calls = [0]
        def mock_validate(answer: str, question: str):
            validation_calls[0] += 1
            if validation_calls[0] == 1:
                # Initial answer fails
                return ValidationResult(passed=False, triggered_rule="length_exceeded")
            else:
                # After critic loop, passes
                return ValidationResult(passed=True, triggered_rule=None)
        
        # Mock decomposer
        def mock_decompose(question: str, max_retries: int = 2):
            return DecompositionResult(
                sub_queries=["What is the capital of France?"],
                reasoning_plan="Answer using the capital information"
            )
        
        # Mock reasoning checker to pass
        def mock_check_reasoning(*args, **kwargs):
            return ReasoningCheckResult(
                action="PASS",
                missing_facts=[],
                reasoning_explanation="All facts present and reasoning valid"
            )
        
        with patch('src.critic.critic_system.validate_answer', side_effect=mock_validate), \
             patch('src.critic.critic_system.decompose_query', side_effect=mock_decompose), \
             patch('src.critic.critic_system.check_reasoning', side_effect=mock_check_reasoning):
            
            result = run_critic_system(
                question="What is the capital of France?",
                initial_answer="A very long answer that exceeds ninety words and contains lots of unnecessary information about France including its history culture geography and many other details that are not relevant to the simple question about the capital city which is Paris but also mentions other cities like Lyon Marseille and Nice which are not the capital",
                retrieval_function=mock_retrieval,
                generation_function=mock_generation
            )
            
            # Should return the improved answer
            assert result == "Paris"
            # Should have called validator twice (initial + re-validation)
            assert validation_calls[0] == 2
    
    def test_full_critic_loop_passes_on_second_iteration(self):
        """Test critic loop that passes validation on second iteration."""
        # Mock retrieval function
        def mock_retrieval(query: str):
            return [
                {"text": "Paris is the capital", "title": "doc1", "score": 0.9}
            ]
        
        # Mock generation function
        generation_calls = [0]
        def mock_generation(question: str, context: str):
            generation_calls[0] += 1
            if generation_calls[0] == 1:
                return "Paris, however the context does not specify"
            else:
                return "Paris"
        
        # Mock validator: fail initial, fail first iteration, pass second iteration
        validation_calls = [0]
        def mock_validate(answer: str, question: str):
            validation_calls[0] += 1
            if validation_calls[0] <= 2:
                # Initial and first iteration fail
                return ValidationResult(passed=False, triggered_rule="hedging_detected")
            else:
                # Second iteration passes
                return ValidationResult(passed=True, triggered_rule=None)
        
        # Mock decomposer
        def mock_decompose(question: str, max_retries: int = 2):
            return DecompositionResult(
                sub_queries=["What is the capital of France?"],
                reasoning_plan="Answer using the capital information"
            )
        
        # Mock reasoning checker to pass
        def mock_check_reasoning(*args, **kwargs):
            return ReasoningCheckResult(
                action="PASS",
                missing_facts=[],
                reasoning_explanation="All facts present and reasoning valid"
            )
        
        with patch('src.critic.critic_system.validate_answer', side_effect=mock_validate), \
             patch('src.critic.critic_system.decompose_query', side_effect=mock_decompose), \
             patch('src.critic.critic_system.check_reasoning', side_effect=mock_check_reasoning):
            
            result = run_critic_system(
                question="What is the capital of France?",
                initial_answer="I do not know.",
                retrieval_function=mock_retrieval,
                generation_function=mock_generation
            )
            
            # Should return the improved answer from second iteration
            assert result == "Paris"
            # Should have called validator 3 times (initial + 2 re-validations)
            assert validation_calls[0] == 3
            # Should have generated 2 answers
            assert generation_calls[0] == 2


class TestCriticSystemIterationLimit:
    """Test iteration limit enforcement."""
    
    def test_reaches_max_iterations(self):
        """Test that system stops after max iterations even if validation fails."""
        # Mock retrieval function
        def mock_retrieval(query: str):
            return [{"text": "Some text", "title": "doc1", "score": 0.9}]
        
        # Mock generation function
        def mock_generation(question: str, context: str):
            return "Generated answer"
        
        # Mock validator to always fail
        def mock_validate(answer: str, question: str):
            return ValidationResult(passed=False, triggered_rule="length_exceeded")
        
        # Mock decomposer
        def mock_decompose(question: str, max_retries: int = 2):
            return DecompositionResult(
                sub_queries=["Sub-query 1"],
                reasoning_plan="Reasoning plan"
            )
        
        # Mock reasoning checker to always pass
        def mock_check_reasoning(*args, **kwargs):
            return ReasoningCheckResult(
                action="PASS",
                missing_facts=[],
                reasoning_explanation="Valid"
            )
        
        with patch('src.critic.critic_system.validate_answer', side_effect=mock_validate), \
             patch('src.critic.critic_system.decompose_query', side_effect=mock_decompose), \
             patch('src.critic.critic_system.check_reasoning', side_effect=mock_check_reasoning):
            
            result = run_critic_system(
                question="Test question?",
                initial_answer="Initial answer",
                retrieval_function=mock_retrieval,
                generation_function=mock_generation
            )
            
            # Should return the most recent answer after max iterations
            assert result == "Generated answer"


class TestCriticSystemReasoningActions:
    """Test handling of different reasoning checker actions."""
    
    def test_retry_retrieval_action(self):
        """Test that system retries retrieval when facts are missing."""
        # Track retrieval calls
        retrieval_calls = []
        def mock_retrieval(query: str):
            retrieval_calls.append(query)
            return [{"text": f"Text for {query}", "title": "doc1", "score": 0.9}]
        
        # Mock generation function
        def mock_generation(question: str, context: str):
            return "Generated answer"
        
        # Mock validator: fail initial, pass after retry
        validation_calls = [0]
        def mock_validate(answer: str, question: str):
            validation_calls[0] += 1
            if validation_calls[0] == 1:
                return ValidationResult(passed=False, triggered_rule="unknown_answer")
            else:
                return ValidationResult(passed=True, triggered_rule=None)
        
        # Mock decomposer
        def mock_decompose(question: str, max_retries: int = 2):
            return DecompositionResult(
                sub_queries=["Sub-query 1", "Sub-query 2"],
                reasoning_plan="Combine answers"
            )
        
        # Mock reasoning checker: first RETRY_RETRIEVAL, then PASS
        reasoning_calls = [0]
        def mock_check_reasoning(*args, **kwargs):
            reasoning_calls[0] += 1
            if reasoning_calls[0] == 1:
                return ReasoningCheckResult(
                    action="RETRY_RETRIEVAL",
                    missing_facts=["Sub-query 2"],
                    reasoning_explanation="Missing facts for sub-query 2"
                )
            else:
                return ReasoningCheckResult(
                    action="PASS",
                    missing_facts=[],
                    reasoning_explanation="All facts present"
                )
        
        with patch('src.critic.critic_system.validate_answer', side_effect=mock_validate), \
             patch('src.critic.critic_system.decompose_query', side_effect=mock_decompose), \
             patch('src.critic.critic_system.check_reasoning', side_effect=mock_check_reasoning):
            
            result = run_critic_system(
                question="Test question?",
                initial_answer="I do not know.",
                retrieval_function=mock_retrieval,
                generation_function=mock_generation
            )
            
            # Should have retried retrieval for missing sub-query
            assert "Sub-query 2" in retrieval_calls
            # Should return final answer
            assert result == "Generated answer"
    
    def test_regenerate_action(self):
        """Test that system regenerates answer when reasoning is invalid."""
        # Mock retrieval function
        def mock_retrieval(query: str):
            return [{"text": "Some text", "title": "doc1", "score": 0.9}]
        
        # Track generation calls
        generation_calls = [0]
        def mock_generation(question: str, context: str):
            generation_calls[0] += 1
            return f"Generated answer {generation_calls[0]}"
        
        # Mock validator: fail initial, pass after regeneration
        validation_calls = [0]
        def mock_validate(answer: str, question: str):
            validation_calls[0] += 1
            if validation_calls[0] == 1:
                return ValidationResult(passed=False, triggered_rule="unknown_answer")
            else:
                return ValidationResult(passed=True, triggered_rule=None)
        
        # Mock decomposer
        def mock_decompose(question: str, max_retries: int = 2):
            return DecompositionResult(
                sub_queries=["Sub-query 1"],
                reasoning_plan="Answer the question"
            )
        
        # Mock reasoning checker: first REGENERATE, then PASS
        reasoning_calls = [0]
        def mock_check_reasoning(*args, **kwargs):
            reasoning_calls[0] += 1
            if reasoning_calls[0] == 1:
                return ReasoningCheckResult(
                    action="REGENERATE",
                    missing_facts=[],
                    reasoning_explanation="Reasoning invalid"
                )
            else:
                return ReasoningCheckResult(
                    action="PASS",
                    missing_facts=[],
                    reasoning_explanation="Reasoning valid"
                )
        
        with patch('src.critic.critic_system.validate_answer', side_effect=mock_validate), \
             patch('src.critic.critic_system.decompose_query', side_effect=mock_decompose), \
             patch('src.critic.critic_system.check_reasoning', side_effect=mock_check_reasoning):
            
            result = run_critic_system(
                question="Test question?",
                initial_answer="I do not know.",
                retrieval_function=mock_retrieval,
                generation_function=mock_generation
            )
            
            # Should have generated answer twice (initial + regeneration)
            assert generation_calls[0] == 2
            # Should return the regenerated answer
            assert result == "Generated answer 2"


class TestCriticSystemErrorHandling:
    """Test error handling in the critic system."""
    
    def test_handles_decomposer_exception(self):
        """Test that system handles decomposer exceptions gracefully."""
        # Mock retrieval and generation
        mock_retrieval = Mock()
        mock_generation = Mock()
        
        # Mock validator to fail initial
        def mock_validate(answer: str, question: str):
            return ValidationResult(passed=False, triggered_rule="unknown_answer")
        
        # Mock decomposer to raise exception
        def mock_decompose(question: str, max_retries: int = 2):
            raise Exception("Decomposer error")
        
        with patch('src.critic.critic_system.validate_answer', side_effect=mock_validate), \
             patch('src.critic.critic_system.decompose_query', side_effect=mock_decompose):
            
            result = run_critic_system(
                question="Test question?",
                initial_answer="I do not know.",
                retrieval_function=mock_retrieval,
                generation_function=mock_generation
            )
            
            # Should return current answer (initial answer) on error
            assert result == "I do not know."
    
    def test_handles_retrieval_exception(self):
        """Test that system handles retrieval exceptions gracefully."""
        # Mock retrieval to raise exception
        def mock_retrieval(query: str):
            raise Exception("Retrieval error")
        
        # Mock generation to return a fallback answer
        def mock_generation(question: str, context: str):
            return "I do not know."
        
        # Mock validator: fail initial, pass after generation
        validation_calls = [0]
        def mock_validate(answer: str, question: str):
            validation_calls[0] += 1
            if validation_calls[0] == 1:
                return ValidationResult(passed=False, triggered_rule="unknown_answer")
            else:
                return ValidationResult(passed=True, triggered_rule=None)
        
        # Mock decomposer
        def mock_decompose(question: str, max_retries: int = 2):
            return DecompositionResult(
                sub_queries=["Sub-query 1"],
                reasoning_plan="Answer the question"
            )
        
        # Mock reasoning checker to pass (even with empty retrieval)
        def mock_check_reasoning(*args, **kwargs):
            return ReasoningCheckResult(
                action="PASS",
                missing_facts=[],
                reasoning_explanation="Accepting answer"
            )
        
        with patch('src.critic.critic_system.validate_answer', side_effect=mock_validate), \
             patch('src.critic.critic_system.decompose_query', side_effect=mock_decompose), \
             patch('src.critic.critic_system.check_reasoning', side_effect=mock_check_reasoning):
            
            result = run_critic_system(
                question="Test question?",
                initial_answer="I do not know.",
                retrieval_function=mock_retrieval,
                generation_function=mock_generation
            )
            
            # Should complete the loop and return the generated answer
            # The retriever handles exceptions gracefully by returning empty lists
            assert result == "I do not know."


class TestCriticSystemIntegration:
    """Test integration with baseline RAG pipeline."""
    
    def test_integration_with_baseline_pipeline(self):
        """Test that system correctly integrates with existing RAG components."""
        # Mock retrieval function (simulates existing RAG retrieval)
        retrieval_calls = []
        def mock_retrieval(query: str):
            retrieval_calls.append(query)
            return [
                {
                    "text": f"Retrieved text for: {query}",
                    "title": f"Document for {query}",
                    "score": 0.85
                }
            ]
        
        # Mock generation function (simulates existing RAG generation)
        generation_calls = []
        def mock_generation(question: str, context: str):
            generation_calls.append((question, context))
            return "Final answer"
        
        # Mock validator: fail initial, pass after critic
        validation_calls = [0]
        def mock_validate(answer: str, question: str):
            validation_calls[0] += 1
            if validation_calls[0] == 1:
                return ValidationResult(passed=False, triggered_rule="unknown_answer")
            else:
                return ValidationResult(passed=True, triggered_rule=None)
        
        # Mock decomposer
        def mock_decompose(question: str, max_retries: int = 2):
            return DecompositionResult(
                sub_queries=["What is X?", "Where is Y?"],
                reasoning_plan="Combine X and Y"
            )
        
        # Mock reasoning checker
        def mock_check_reasoning(*args, **kwargs):
            return ReasoningCheckResult(
                action="PASS",
                missing_facts=[],
                reasoning_explanation="Valid"
            )
        
        with patch('src.critic.critic_system.validate_answer', side_effect=mock_validate), \
             patch('src.critic.critic_system.decompose_query', side_effect=mock_decompose), \
             patch('src.critic.critic_system.check_reasoning', side_effect=mock_check_reasoning):
            
            result = run_critic_system(
                question="Original question?",
                initial_answer="I do not know.",
                retrieval_function=mock_retrieval,
                generation_function=mock_generation
            )
            
            # Verify retrieval was called for each sub-query
            assert "What is X?" in retrieval_calls
            assert "Where is Y?" in retrieval_calls
            
            # Verify generation was called with augmented prompt
            assert len(generation_calls) == 1
            question_arg, context_arg = generation_calls[0]
            assert "Original question?" in question_arg
            assert "Combine X and Y" in context_arg
            
            # Verify final answer returned
            assert result == "Final answer"
