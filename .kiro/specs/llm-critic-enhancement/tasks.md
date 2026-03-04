# Implementation Plan: LLM Critic Enhancement

## Overview

This implementation plan adds a critic function to an existing RAG pipeline in a Jupyter notebook to improve answer accuracy on the HotpotQA dataset. The critic system validates answers, decomposes complex queries into sub-queries, retrieves targeted context, validates reasoning, and iterates up to 2 times before returning a final answer. The implementation uses Python with the Hypothesis framework for property-based testing.

## Tasks

- [x] 1. Set up project structure and data models
  - Create Python module structure in the Jupyter notebook
  - Implement ValidationResult, DecompositionResult, Document, ReasoningCheckResult, and CriticState dataclasses
  - Set up Hypothesis testing framework configuration
  - _Requirements: 6.2_

- [x] 2. Implement Answer Validator component
  - [x] 2.1 Create validate_answer function with rule-based checks
    - Implement exact match check for "I do not know."
    - Implement length check (>90 words)
    - Implement hedging phrase detection (5 phrases)
    - Implement multiple entity check with conjunction detection
    - Return ValidationResult with pass/fail status and triggered rule
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_
  
  - [ ]* 2.2 Write property test for validation rule triggering
    - **Property 1: Answer Validation Rule Triggering**
    - **Validates: Requirements 1.2, 1.3, 1.4, 1.5**
  
  - [ ]* 2.3 Write unit tests for Answer Validator
    - Test exact "I do not know." match (positive and negative)
    - Test length boundaries (89, 90, 91 words)
    - Test each hedging phrase individually
    - Test multiple entity detection with conjunctions
    - Test question type detection
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

- [x] 3. Implement Query Decomposer component
  - [x] 3.1 Create decompose_query function with LLM integration
    - Design LLM prompt template for query decomposition
    - Implement JSON parsing for sub-queries and reasoning plan
    - Validate 1-3 sub-queries constraint
    - Handle malformed JSON with retry logic (max 2 retries)
    - Implement fallback for invalid sub-query counts
    - _Requirements: 2.1, 2.2, 2.3, 2.4_
  
  - [ ]* 3.2 Write property test for decomposition cardinality
    - **Property 2: Query Decomposition Cardinality**
    - **Validates: Requirements 2.1, 2.2**
  
  - [ ]* 3.3 Write property test for single prompt decomposition
    - **Property 3: Single Prompt Decomposition**
    - **Validates: Requirements 2.3**
  
  - [ ]* 3.4 Write unit tests for Query Decomposer
    - Test decomposition with mock LLM responses (1, 2, 3 sub-queries)
    - Test malformed JSON handling
    - Test invalid sub-query counts (0, 4)
    - Test missing reasoning plan handling
    - Test LLM timeout/error handling
    - _Requirements: 2.1, 2.2, 2.3_

- [x] 4. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 5. Implement Sub-Query Retriever component
  - [x] 5.1 Create retrieve_for_subqueries function
    - Iterate through sub-queries and call existing retrieval function
    - Aggregate results in dictionary mapping sub-query to documents
    - Handle empty retrieval results
    - Handle retrieval function exceptions
    - _Requirements: 3.1_
  
  - [ ]* 5.2 Write property test for complete sub-query retrieval
    - **Property 4: Complete Sub-Query Retrieval**
    - **Validates: Requirements 3.1, 3.2, 3.3**
  
  - [ ]* 5.3 Write unit tests for Sub-Query Retriever
    - Test retrieval for single sub-query
    - Test retrieval for multiple sub-queries
    - Test empty retrieval results
    - Test retrieval function exceptions
    - _Requirements: 3.1_

- [x] 6. Implement Generator component
  - [x] 6.1 Create generate_with_reasoning function
    - Format context with sub-query associations
    - Create augmented prompt with reasoning plan
    - Wrap existing generation function
    - Handle empty answer generation
    - Handle LLM timeout/error
    - _Requirements: 3.2, 3.3_
  
  - [ ]* 6.2 Write unit tests for Generator
    - Test generation with sub-query contexts
    - Test context formatting
    - Test empty answer handling
    - Test LLM timeout/error handling
    - _Requirements: 3.2, 3.3_

- [x] 7. Implement Reasoning Checker component
  - [x] 7.1 Create check_reasoning function with LLM validation
    - Design LLM prompt template for reasoning validation
    - Implement JSON parsing for validation results
    - Implement decision logic (PASS, RETRY_RETRIEVAL, REGENERATE)
    - Handle malformed JSON with retry logic (max 2 retries)
    - Implement default fallback for missing validation fields
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6_
  
  - [ ]* 7.2 Write property test for single prompt reasoning validation
    - **Property 5: Single Prompt Reasoning Validation**
    - **Validates: Requirements 4.1, 4.2, 4.3**
  
  - [ ]* 7.3 Write property test for reasoning checker decision logic
    - **Property 6: Reasoning Checker Decision Logic**
    - **Validates: Requirements 4.4, 4.5, 4.6**
  
  - [ ]* 7.4 Write unit tests for Reasoning Checker
    - Test validation with all facts present and reasoning valid
    - Test validation with missing facts
    - Test validation with invalid reasoning
    - Test malformed JSON handling
    - Test action selection logic
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6_

- [x] 8. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 9. Implement Critic System orchestrator
  - [x] 9.1 Create run_critic_system function with iteration control
    - Implement main control flow loop
    - Initialize CriticState for tracking
    - Integrate Answer Validator for initial check
    - Integrate Query Decomposer for query breakdown
    - Integrate Sub-Query Retriever for context gathering
    - Integrate Generator for answer production
    - Integrate Reasoning Checker for validation
    - Implement iteration counter (max 2 iterations)
    - Implement termination conditions (validation pass or max iterations)
    - Handle retry retrieval logic
    - Handle regeneration logic
    - _Requirements: 5.1, 5.2, 5.3_
  
  - [ ]* 9.2 Write property test for maximum iteration limit
    - **Property 7: Maximum Iteration Limit**
    - **Validates: Requirements 5.1**
  
  - [ ]* 9.3 Write property test for iteration termination conditions
    - **Property 8: Iteration Termination Conditions**
    - **Validates: Requirements 5.2, 5.3**
  
  - [ ]* 9.4 Write property test for pipeline integration interface
    - **Property 9: Pipeline Integration Interface**
    - **Validates: Requirements 6.2**
  
  - [ ]* 9.5 Write property test for output format compatibility
    - **Property 10: Output Format Compatibility**
    - **Validates: Requirements 6.3, 6.4, 7.2**
  
  - [ ]* 9.6 Write unit tests for Critic System
    - Test full critic loop with passing validation on first iteration
    - Test full critic loop with passing validation on second iteration
    - Test full critic loop reaching max iterations
    - Test early termination when validation passes
    - Test integration with baseline RAG pipeline
    - _Requirements: 5.1, 5.2, 5.3_

- [x] 10. Integrate with existing Jupyter notebook pipeline
  - [x] 10.1 Wire Critic System to existing RAG pipeline
    - Connect to existing retrieval function
    - Connect to existing generation function
    - Accept initial answer from baseline RAG pipeline
    - Ensure output format matches baseline
    - Add logging for debugging and analysis
    - _Requirements: 6.1, 6.2, 6.3_
  
  - [ ]* 10.2 Write integration tests for end-to-end flow
    - Test complete critic loop with real LLM calls (using mock or small model)
    - Test with sample HotpotQA questions
    - Verify state transitions through full pipeline
    - Measure iteration counts and validation outcomes
    - _Requirements: 6.1, 6.2, 6.3, 6.4_

- [x] 11. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 12. Implement HotpotQA evaluation
  - [x] 12.1 Create evaluation script for 300-question dataset
    - Load stratified 300-question HotpotQA dataset
    - Process each question through Critic System
    - Collect answers in HotpotQA evaluation format
    - Integrate with HotpotQA official evaluation script
    - Generate accuracy report comparing to 60.17% baseline
    - Add metrics for iteration counts and validation outcomes
    - _Requirements: 7.1, 7.2, 7.3_
  
  - [ ]* 12.2 Write integration test for HotpotQA evaluation
    - Test on small subset of HotpotQA questions
    - Verify compatibility with official evaluation script
    - Validate output format
    - _Requirements: 7.1, 7.2_

- [x] 13. Final checkpoint - Ensure all tests pass and evaluation runs successfully
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties using Hypothesis framework (minimum 100 iterations per test)
- Unit tests validate specific examples and edge cases
- All implementation is done in Python within the existing Jupyter notebook pipeline
- The Critic System integrates with existing retrieval and generation functions
- Maximum 2 iterations per question to avoid infinite loops
- LLM calls use single prompts for decomposition and reasoning validation to minimize latency
