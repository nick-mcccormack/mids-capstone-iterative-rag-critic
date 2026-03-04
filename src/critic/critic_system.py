"""Critic System orchestrator for the LLM Critic Enhancement system.

This module implements the main control flow loop that ties together all components
(validator, decomposer, retriever, generator, reasoning checker) into a cohesive
iterative improvement loop. It manages state, handles iteration limits, and implements
the complete critic workflow.
"""

import logging
from typing import Callable, Dict, List

from src.critic.models import CriticState, ValidationResult
from src.critic.validator import validate_answer
from src.critic.decomposer import decompose_query
from src.critic.retriever import retrieve_for_subqueries
from src.critic.generator import generate_with_reasoning
from src.critic.reasoning_checker import check_reasoning


logger = logging.getLogger(__name__)


# Maximum number of critic loop iterations
MAX_ITERATIONS = 2


def run_critic_system(
    question: str,
    initial_answer: str,
    retrieval_function: Callable[[str], List[Dict]],
    generation_function: Callable[[str, str], str]
) -> str:
    """
    Main control flow for the critic system with iteration control.
    
    This function orchestrates the complete critic workflow:
    1. Validates initial answer against quality criteria
    2. If validation fails, decomposes query into sub-queries with reasoning plan
    3. Retrieves targeted context for each sub-query
    4. Generates new answer using retrieved context and reasoning plan
    5. Validates that intermediate facts are retrieved and reasoning holds
    6. Iterates up to 2 times before returning final answer
    
    The system terminates when either:
    - The Answer Validator marks an answer as passing, OR
    - The iteration count reaches 2 (MAX_ITERATIONS)
    
    Args:
        question: Original question to answer
        initial_answer: Answer produced by baseline RAG pipeline
        retrieval_function: Existing RAG pipeline retrieval function that takes
                          a query string and returns a list of document dictionaries
        generation_function: Existing RAG pipeline generation function that takes
                           a question and context string, returns an answer string
    
    Returns:
        Final answer string (either validated answer or answer after max iterations)
    
    Examples:
        >>> answer = run_critic_system(
        ...     "Where was the director of Inception born?",
        ...     "Christopher Nolan",
        ...     mock_retrieval_function,
        ...     mock_generation_function
        ... )
        >>> isinstance(answer, str)
        True
    """
    # Initialize CriticState for tracking
    state = CriticState(
        question=question,
        current_answer=initial_answer,
        iteration_count=0
    )
    
    logger.info(f"Starting critic system for question: {question}")
    logger.info(f"Initial answer: {initial_answer}")
    
    # Step 1: Validate initial answer
    validation_result = validate_answer(state.current_answer, state.question)
    state.validation_history.append(validation_result)
    
    logger.info(f"Initial validation: passed={validation_result.passed}, "
                f"triggered_rule={validation_result.triggered_rule}")
    
    # If initial answer passes validation, return it immediately
    if validation_result.passed:
        logger.info("Initial answer passed validation, returning without critic loop")
        return state.current_answer
    
    # Main critic loop - iterate up to MAX_ITERATIONS
    while state.iteration_count < MAX_ITERATIONS:
        logger.info(f"Starting critic iteration {state.iteration_count + 1}/{MAX_ITERATIONS}")
        
        try:
            # Step 2: Decompose query into sub-queries with reasoning plan
            if state.sub_queries is None:
                logger.info("Decomposing query into sub-queries")
                decomposition = decompose_query(state.question)
                state.sub_queries = decomposition.sub_queries
                state.reasoning_plan = decomposition.reasoning_plan
                
                logger.info(f"Decomposed into {len(state.sub_queries)} sub-queries")
                logger.info(f"Reasoning plan: {state.reasoning_plan}")
            
            # Step 3: Retrieve context for each sub-query
            logger.info("Retrieving context for sub-queries")
            state.sub_query_contexts = retrieve_for_subqueries(
                state.sub_queries,
                retrieval_function
            )
            
            # Log retrieval results
            for sub_query, docs in state.sub_query_contexts.items():
                logger.info(f"Retrieved {len(docs)} documents for: {sub_query}")
            
            # Step 4: Generate new answer with context and reasoning plan
            logger.info("Generating new answer with reasoning")
            state.current_answer = generate_with_reasoning(
                state.question,
                state.sub_query_contexts,
                state.reasoning_plan,
                generation_function
            )
            
            logger.info(f"Generated answer: {state.current_answer}")
            
            # Step 5: Validate reasoning and retrieved facts
            logger.info("Checking reasoning and facts")
            reasoning_result = check_reasoning(
                state.question,
                state.sub_queries,
                state.sub_query_contexts,
                state.reasoning_plan,
                state.current_answer
            )
            
            logger.info(f"Reasoning check action: {reasoning_result.action}")
            logger.info(f"Reasoning explanation: {reasoning_result.reasoning_explanation}")
            
            # Handle reasoning check result
            if reasoning_result.action == "PASS":
                # Step 6: Re-validate answer with Answer Validator
                logger.info("Reasoning check passed, re-validating answer")
                validation_result = validate_answer(state.current_answer, state.question)
                state.validation_history.append(validation_result)
                
                logger.info(f"Re-validation: passed={validation_result.passed}, "
                           f"triggered_rule={validation_result.triggered_rule}")
                
                # Increment iteration count
                state.iteration_count += 1
                
                # If validation passes, return the answer
                if validation_result.passed:
                    logger.info(f"Answer passed validation on iteration {state.iteration_count}, "
                               f"returning final answer")
                    return state.current_answer
                
                # If validation fails but we've reached max iterations, return current answer
                if state.iteration_count >= MAX_ITERATIONS:
                    logger.info(f"Reached max iterations ({MAX_ITERATIONS}), "
                               f"returning current answer despite validation failure")
                    return state.current_answer
                
                # Otherwise, continue to next iteration
                logger.info(f"Validation failed, continuing to iteration {state.iteration_count + 1}")
                
            elif reasoning_result.action == "RETRY_RETRIEVAL":
                # Retry retrieval for missing facts
                logger.info(f"Retrying retrieval for missing facts: {reasoning_result.missing_facts}")
                
                # Re-retrieve for sub-queries with missing facts
                for missing_sub_query in reasoning_result.missing_facts:
                    if missing_sub_query in state.sub_query_contexts:
                        logger.info(f"Re-retrieving for: {missing_sub_query}")
                        try:
                            new_docs = retrieval_function(missing_sub_query)
                            # Convert to Document objects
                            from src.critic.models import Document
                            documents = []
                            for raw_doc in new_docs:
                                documents.append(Document(
                                    text=raw_doc.get("text", ""),
                                    source=raw_doc.get("title", raw_doc.get("doc_id", "")),
                                    score=raw_doc.get("score", 0.0)
                                ))
                            state.sub_query_contexts[missing_sub_query] = documents
                            logger.info(f"Re-retrieved {len(documents)} documents")
                        except Exception as e:
                            logger.warning(f"Re-retrieval failed for '{missing_sub_query}': {e}")
                
                # Continue loop to regenerate with updated context
                # Don't increment iteration count yet - only increment after validation
                
            elif reasoning_result.action == "REGENERATE":
                # Regenerate answer with same context
                logger.info("Regenerating answer with same context")
                # Continue loop to regenerate
                # Don't increment iteration count yet - only increment after validation
            
        except Exception as e:
            # Handle unexpected errors - log and return current answer
            logger.error(f"Unexpected error in critic loop: {e}")
            logger.info("Returning current answer due to error")
            return state.current_answer
    
    # If we exit the loop without returning, we've reached max iterations
    logger.info(f"Exited critic loop after {state.iteration_count} iterations, "
               f"returning final answer")
    return state.current_answer
