# Requirements Document

## Introduction

This document specifies requirements for enhancing an existing RAG pipeline with an LLM critic function to improve answer accuracy on the HotpotQA dataset. The baseline RAG system with reranking achieves 60.17% accuracy. Manual failure analysis of 100+ failed questions identified three primary failure types: over-answering with hedging text (45%), missing context from retrieval (35%), and incorrect reasoning despite relevant documents (20%). The critic enhancement will detect problematic answers, decompose complex queries, retrieve additional context, and validate reasoning chains to address these failure modes.

## Glossary

- **RAG_Pipeline**: The existing Retrieval-Augmented Generation system that retrieves documents and generates answers
- **Critic_System**: The enhancement module that evaluates answers and orchestrates iterative improvement
- **Answer_Validator**: Component that checks if an answer requires critic intervention
- **Query_Decomposer**: Component that breaks multi-hop queries into sub-queries and generates reasoning plans
- **Sub_Query_Retriever**: Component that retrieves documents for individual sub-queries
- **Reasoning_Checker**: Component that validates whether intermediate facts are retrieved and reasoning plan holds
- **Generator**: The LLM component that produces answers from retrieved context
- **Initial_Answer**: The answer produced by the baseline RAG_Pipeline before critic intervention
- **Critic_Loop**: The iterative process of query decomposition, retrieval, generation, and validation
- **Hedging_Phrase**: Text patterns indicating uncertainty such as "however", "but the context does not", "some also", "specifically", "but also"
- **Sub_Query**: An individual question extracted from a multi-hop query (maximum 3 per query)
- **Reasoning_Plan**: A structured chain showing how intermediate answers combine to produce the final answer
- **Intermediate_Fact**: A piece of information retrieved to answer a specific sub-query

## Requirements

### Requirement 1: Detect Problematic Answers

**User Story:** As a RAG system user, I want the system to identify when answers need improvement, so that low-quality responses trigger the critic enhancement process.

#### Acceptance Criteria

1. WHEN the Initial_Answer contains the exact text "I do not know.", THE Answer_Validator SHALL trigger the Critic_Loop
2. WHEN the Initial_Answer exceeds 90 words in length, THE Answer_Validator SHALL trigger the Critic_Loop
3. WHEN the Initial_Answer contains any Hedging_Phrase, THE Answer_Validator SHALL trigger the Critic_Loop
4. WHEN the Initial_Answer contains multiple candidate entities joined by "and", "or", or "but also" AND the query starts with "Where", "Who", "What year", or "When", THE Answer_Validator SHALL trigger the Critic_Loop
5. WHEN none of the triggering conditions are met, THE Answer_Validator SHALL mark the answer as passing and skip the Critic_Loop

### Requirement 2: Decompose Multi-Hop Queries

**User Story:** As a RAG system developer, I want complex queries broken into sub-queries with reasoning plans, so that the system can retrieve targeted context for each reasoning step.

#### Acceptance Criteria

1. WHEN the Critic_Loop is triggered, THE Query_Decomposer SHALL generate between 1 and 3 Sub_Queries from the original query
2. WHEN the Query_Decomposer generates Sub_Queries, THE Query_Decomposer SHALL produce a Reasoning_Plan that chains intermediate answers to a final answer
3. THE Query_Decomposer SHALL use a single LLM prompt to generate both Sub_Queries and the Reasoning_Plan
4. THE Reasoning_Plan SHALL specify how intermediate answers from Sub_Queries combine to produce the final answer

### Requirement 3: Retrieve Context for Sub-Queries

**User Story:** As a RAG system developer, I want targeted retrieval for each sub-query, so that the system surfaces relevant documents for each reasoning step.

#### Acceptance Criteria

1. WHEN Sub_Queries are generated, THE Sub_Query_Retriever SHALL retrieve documents for each Sub_Query independently
2. THE Sub_Query_Retriever SHALL provide retrieved context to the Generator along with the Reasoning_Plan
3. WHEN retrieval completes for all Sub_Queries, THE Generator SHALL produce a new answer using the retrieved context and Reasoning_Plan

### Requirement 4: Validate Reasoning and Retrieved Facts

**User Story:** As a RAG system developer, I want the system to verify that intermediate facts are retrieved and reasoning holds, so that the system can identify and correct specific failure modes.

#### Acceptance Criteria

1. WHEN the Generator produces a new answer, THE Reasoning_Checker SHALL use a single LLM prompt to validate both intermediate facts and reasoning plan
2. THE Reasoning_Checker SHALL verify that Intermediate_Fact for each Sub_Query was retrieved
3. THE Reasoning_Checker SHALL verify that the Reasoning_Plan holds in the generated answer
4. WHEN both intermediate facts and reasoning plan validations pass, THE Reasoning_Checker SHALL send the answer to the Answer_Validator for re-evaluation
5. WHEN intermediate facts validation fails, THE Reasoning_Checker SHALL trigger retrieval retry for missing facts
6. WHEN reasoning plan validation fails, THE Reasoning_Checker SHALL trigger answer regeneration

### Requirement 5: Limit Critic Iterations

**User Story:** As a RAG system operator, I want the critic process to terminate after a maximum number of attempts, so that the system avoids infinite loops and returns timely responses.

#### Acceptance Criteria

1. THE Critic_System SHALL limit the Critic_Loop to a maximum of 2 attempts
2. WHEN the Critic_Loop reaches 2 attempts without passing validation, THE Critic_System SHALL return the most recent answer as the final answer
3. WHEN the Answer_Validator marks an answer as passing within 2 attempts, THE Critic_System SHALL return that answer as the final answer

### Requirement 6: Integrate with Existing Pipeline

**User Story:** As a RAG system developer, I want the critic enhancement to work within the existing Jupyter notebook pipeline, so that I can evaluate improvements without rebuilding the infrastructure.

#### Acceptance Criteria

1. THE Critic_System SHALL accept the Initial_Answer from the existing RAG_Pipeline as input
2. THE Critic_System SHALL integrate with the existing retrieval and generation components in the Jupyter notebook
3. THE Critic_System SHALL output answers in the same format as the baseline RAG_Pipeline
4. THE Critic_System SHALL support evaluation using the HotpotQA official evaluation script

### Requirement 7: Evaluate on HotpotQA Dataset

**User Story:** As a researcher, I want to measure accuracy improvements on the stratified HotpotQA dataset, so that I can quantify the critic enhancement's impact.

#### Acceptance Criteria

1. THE Critic_System SHALL process all 300 questions from the stratified HotpotQA dataset
2. THE Critic_System SHALL support evaluation using the HotpotQA official evaluation script
3. WHEN evaluation completes, THE Critic_System SHALL report accuracy as a percentage for comparison with the 60.17% baseline
