# Design Document: LLM Critic Enhancement

## Overview

This design specifies an enhancement to an existing RAG pipeline that adds a critic function to iteratively improve answer quality on the HotpotQA dataset. The baseline system achieves 60.17% accuracy, with failure analysis revealing three primary modes: over-answering with hedging (45%), missing retrieval context (35%), and incorrect reasoning (20%).

The critic enhancement introduces a feedback loop that:
1. Validates initial answers against quality criteria
2. Decomposes complex multi-hop queries into sub-queries with reasoning plans
3. Retrieves targeted context for each sub-query
4. Validates that intermediate facts are retrieved and reasoning holds
5. Iterates up to 2 times before returning the final answer

The system integrates with the existing Jupyter notebook pipeline, maintaining compatibility with current retrieval and generation components while adding validation and decomposition layers.

## Architecture

### System Components

The critic enhancement consists of five core components that wrap the existing RAG pipeline:

```
┌─────────────────────────────────────────────────────────────────┐
│                        Critic System                             │
│                                                                   │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │   Answer     │───▶│    Query     │───▶│  Sub-Query   │      │
│  │  Validator   │    │ Decomposer   │    │  Retriever   │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│         │                                         │              │
│         │                                         ▼              │
│         │                                  ┌──────────────┐     │
│         │                                  │  Generator   │     │
│         │                                  └──────────────┘     │
│         │                                         │              │
│         │                                         ▼              │
│         │                                  ┌──────────────┐     │
│         └─────────────────────────────────│  Reasoning   │     │
│                                            │   Checker    │     │
│                                            └──────────────┘     │
│                                                   │              │
│                                            [Loop or Exit]        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                      ┌──────────────┐
                      │ Existing RAG │
                      │   Pipeline   │
                      └──────────────┘
```

### Component Responsibilities

**Answer Validator**
- Evaluates answer quality using rule-based checks
- Triggers critic loop when quality issues detected
- Re-evaluates answers after critic iterations
- Returns final answer when validation passes or max iterations reached

**Query Decomposer**
- Uses LLM to break complex queries into 1-3 sub-queries
- Generates reasoning plan showing how sub-answers combine
- Single prompt produces both sub-queries and reasoning plan
- Outputs structured format for downstream processing

**Sub-Query Retriever**
- Executes retrieval for each sub-query independently
- Leverages existing RAG pipeline retrieval mechanism
- Aggregates retrieved documents with sub-query associations
- Passes context bundle to generator

**Generator**
- Produces answers using retrieved context and reasoning plan
- Wraps existing RAG pipeline generation component
- Receives sub-query contexts and reasoning plan as input
- Outputs answer for validation

**Reasoning Checker**
- Uses LLM to validate intermediate facts and reasoning plan
- Single prompt checks both fact retrieval and reasoning validity
- Determines next action: pass to validator, retry retrieval, or regenerate
- Tracks validation failures for iteration control

### Control Flow


```
Input: Question, Initial_Answer from baseline RAG
iteration_count = 0

1. Answer_Validator checks Initial_Answer
   ├─ If passes → Return Initial_Answer
   └─ If fails → Continue to step 2

2. Query_Decomposer generates sub-queries + reasoning plan
   
3. Sub_Query_Retriever fetches context for each sub-query

4. Generator produces new answer with context + reasoning plan

5. Reasoning_Checker validates facts and reasoning
   ├─ If both pass → Go to step 6
   ├─ If facts missing → Retry retrieval (go to step 3)
   └─ If reasoning fails → Regenerate answer (go to step 4)

6. iteration_count += 1
   
7. If iteration_count >= 2:
   └─ Return current answer
   Else:
   └─ Answer_Validator checks current answer (go to step 1)
```

### Integration Points

The critic system integrates with the existing Jupyter notebook pipeline at three points:

1. **Input Interface**: Accepts question and initial answer from baseline RAG pipeline
2. **Retrieval Interface**: Calls existing retrieval function with sub-queries
3. **Generation Interface**: Calls existing generation function with augmented context

This design preserves the existing pipeline infrastructure while adding validation and decomposition layers as preprocessing and postprocessing steps.

## Components and Interfaces

### Answer Validator

**Purpose**: Rule-based quality checker that determines if an answer requires critic intervention.

**Interface**:
```python
def validate_answer(answer: str, question: str) -> ValidationResult:
    """
    Checks answer against quality criteria.
    
    Args:
        answer: The answer text to validate
        question: The original question (needed for entity check)
    
    Returns:
        ValidationResult with pass/fail status and triggered rules
    """
```

**Validation Rules**:
1. Exact match check: `answer == "I do not know."`
2. Length check: `len(answer.split()) > 90`
3. Hedging phrase check: Contains any of ["however", "but the context does not", "some also", "specifically", "but also"]
4. Multiple entity check: Question starts with ["Where", "Who", "What year", "When"] AND answer contains ["and", "or", "but also"] joining entities

**Algorithm**:
```
For each validation rule:
    If rule triggers:
        Return ValidationResult(passed=False, triggered_rule=rule_name)
Return ValidationResult(passed=True, triggered_rule=None)
```

### Query Decomposer

**Purpose**: LLM-based component that breaks multi-hop queries into sub-queries with reasoning plans.

**Interface**:
```python
def decompose_query(question: str) -> DecompositionResult:
    """
    Decomposes question into sub-queries with reasoning plan.
    
    Args:
        question: The original multi-hop question
    
    Returns:
        DecompositionResult containing 1-3 sub-queries and reasoning plan
    """
```

**LLM Prompt Template**:
```
You are analyzing a multi-hop question that requires multiple pieces of information to answer.

Question: {question}

Your task:
1. Break this question into 1-3 sub-questions that each retrieve a specific piece of information
2. Create a reasoning plan showing how the answers to sub-questions combine to produce the final answer

Output format (JSON):
{
  "sub_queries": [
    "sub-question 1",
    "sub-question 2",
    ...
  ],
  "reasoning_plan": "Step-by-step explanation of how sub-answers combine: [sub-answer 1] provides X, [sub-answer 2] provides Y, combining them gives final answer Z"
}

Requirements:
- Generate 1-3 sub-queries (no more, no less)
- Each sub-query should be independently answerable
- Reasoning plan must reference each sub-query's answer
- Use clear, specific language
```

**Output Parsing**:
- Parse JSON response from LLM
- Validate 1-3 sub-queries present
- Validate reasoning plan is non-empty string
- Handle malformed JSON with retry or fallback

### Sub-Query Retriever

**Purpose**: Executes retrieval for each sub-query and aggregates results.

**Interface**:
```python
def retrieve_for_subqueries(
    sub_queries: List[str],
    retrieval_function: Callable
) -> Dict[str, List[Document]]:
    """
    Retrieves documents for each sub-query.
    
    Args:
        sub_queries: List of sub-questions to retrieve for
        retrieval_function: Existing RAG pipeline retrieval function
    
    Returns:
        Dictionary mapping sub-query to retrieved documents
    """
```

**Algorithm**:
```
results = {}
For each sub_query in sub_queries:
    documents = retrieval_function(sub_query)
    results[sub_query] = documents
Return results
```

**Context Aggregation**:
- Maintains association between sub-query and retrieved documents
- Passes structured context to generator
- Preserves document metadata (source, relevance scores)

### Generator

**Purpose**: Produces answers using sub-query contexts and reasoning plan.

**Interface**:
```python
def generate_with_reasoning(
    question: str,
    sub_query_contexts: Dict[str, List[Document]],
    reasoning_plan: str,
    generation_function: Callable
) -> str:
    """
    Generates answer using sub-query contexts and reasoning plan.
    
    Args:
        question: Original question
        sub_query_contexts: Retrieved documents per sub-query
        reasoning_plan: How to combine sub-answers
        generation_function: Existing RAG pipeline generation function
    
    Returns:
        Generated answer string
    """
```

**Context Formatting**:
```
For the question: {question}

Reasoning Plan: {reasoning_plan}

Retrieved Information:
Sub-query 1: {sub_query_1}
Documents: {documents_1}

Sub-query 2: {sub_query_2}
Documents: {documents_2}

...

Generate a concise answer following the reasoning plan.
```

**Integration**:
- Wraps existing generation function with augmented prompt
- Maintains compatibility with existing LLM interface
- Returns plain text answer

### Reasoning Checker

**Purpose**: LLM-based validator that checks intermediate facts and reasoning plan.

**Interface**:
```python
def check_reasoning(
    question: str,
    sub_queries: List[str],
    sub_query_contexts: Dict[str, List[Document]],
    reasoning_plan: str,
    answer: str
) -> ReasoningCheckResult:
    """
    Validates that intermediate facts are retrieved and reasoning holds.
    
    Args:
        question: Original question
        sub_queries: List of sub-questions
        sub_query_contexts: Retrieved documents per sub-query
        reasoning_plan: Expected reasoning chain
        answer: Generated answer to validate
    
    Returns:
        ReasoningCheckResult with validation status and next action
    """
```

**LLM Prompt Template**:
```
You are validating whether a generated answer properly uses retrieved information and follows a reasoning plan.

Original Question: {question}

Sub-queries and Retrieved Context:
{for each sub_query and its documents}

Reasoning Plan: {reasoning_plan}

Generated Answer: {answer}

Validate two things:
1. INTERMEDIATE FACTS: For each sub-query, was the necessary information retrieved in the documents?
2. REASONING PLAN: Does the generated answer follow the reasoning plan and correctly combine the intermediate facts?

Output format (JSON):
{
  "facts_retrieved": true/false,
  "missing_facts": ["sub-query 1", ...] or [],
  "reasoning_valid": true/false,
  "reasoning_explanation": "brief explanation of reasoning validation"
}
```

**Decision Logic**:
```
If facts_retrieved == True AND reasoning_valid == True:
    Return ReasoningCheckResult(action="PASS")
Else If facts_retrieved == False:
    Return ReasoningCheckResult(action="RETRY_RETRIEVAL", missing=missing_facts)
Else If reasoning_valid == False:
    Return ReasoningCheckResult(action="REGENERATE")
```

## Data Models

### ValidationResult

```python
@dataclass
class ValidationResult:
    passed: bool
    triggered_rule: Optional[str]  # None if passed, rule name if failed
```

**Fields**:
- `passed`: Boolean indicating if answer passes quality checks
- `triggered_rule`: Name of the rule that triggered failure (e.g., "length_exceeded", "hedging_detected", "multiple_entities", "unknown_answer")

### DecompositionResult

```python
@dataclass
class DecompositionResult:
    sub_queries: List[str]  # 1-3 sub-questions
    reasoning_plan: str     # How sub-answers combine
```

**Constraints**:
- `sub_queries` must contain 1-3 elements
- Each sub-query must be a non-empty string
- `reasoning_plan` must be a non-empty string

### Document

```python
@dataclass
class Document:
    text: str
    source: str
    score: float  # Relevance score from retrieval
```

**Fields**:
- `text`: Document content
- `source`: Document identifier or URL
- `score`: Retrieval relevance score

### ReasoningCheckResult

```python
@dataclass
class ReasoningCheckResult:
    action: str  # "PASS", "RETRY_RETRIEVAL", "REGENERATE"
    missing_facts: List[str]  # Sub-queries with missing facts
    reasoning_explanation: str  # Explanation of reasoning validation
```

**Action Values**:
- `PASS`: Both facts and reasoning validated successfully
- `RETRY_RETRIEVAL`: Some intermediate facts not retrieved
- `REGENERATE`: Facts retrieved but reasoning invalid

### CriticState

```python
@dataclass
class CriticState:
    question: str
    current_answer: str
    iteration_count: int
    sub_queries: Optional[List[str]]
    reasoning_plan: Optional[str]
    sub_query_contexts: Optional[Dict[str, List[Document]]]
    validation_history: List[ValidationResult]
```

**Purpose**: Tracks state across critic loop iterations for debugging and analysis.



## Correctness Properties

A property is a characteristic or behavior that should hold true across all valid executions of a system-essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.

### Property 1: Answer Validation Rule Triggering

For any answer and question pair, the Answer_Validator should trigger the critic loop if and only if at least one of the following conditions holds: (1) answer length exceeds 90 words, (2) answer contains a hedging phrase, (3) answer contains multiple entities with conjunctions and question starts with specific interrogatives, or (4) answer is exactly "I do not know."

**Validates: Requirements 1.2, 1.3, 1.4, 1.5**

### Property 2: Query Decomposition Cardinality

For any question that triggers the critic loop, the Query_Decomposer should generate between 1 and 3 sub-queries (inclusive) along with a non-empty reasoning plan.

**Validates: Requirements 2.1, 2.2**

### Property 3: Single Prompt Decomposition

For any question decomposition, the Query_Decomposer should make exactly one LLM call to generate both sub-queries and reasoning plan.

**Validates: Requirements 2.3**

### Property 4: Complete Sub-Query Retrieval

For any set of sub-queries generated by the decomposer, the Sub_Query_Retriever should retrieve documents for each sub-query independently, and the Generator should receive both the retrieved contexts and the reasoning plan.

**Validates: Requirements 3.1, 3.2, 3.3**

### Property 5: Single Prompt Reasoning Validation

For any generated answer with sub-queries and reasoning plan, the Reasoning_Checker should make exactly one LLM call to validate both intermediate facts and reasoning plan, and should output validation results for both aspects.

**Validates: Requirements 4.1, 4.2, 4.3**

### Property 6: Reasoning Checker Decision Logic

For any reasoning validation result, the Reasoning_Checker should return action "PASS" when both facts and reasoning are valid, "RETRY_RETRIEVAL" when facts are missing, and "REGENERATE" when facts are present but reasoning is invalid.

**Validates: Requirements 4.4, 4.5, 4.6**

### Property 7: Maximum Iteration Limit

For any question processed by the Critic_System, the critic loop should execute at most 2 iterations before returning a final answer, regardless of validation results.

**Validates: Requirements 5.1**

### Property 8: Iteration Termination Conditions

For any question processed by the Critic_System, the system should terminate and return the current answer when either (1) the Answer_Validator marks an answer as passing, or (2) the iteration count reaches 2.

**Validates: Requirements 5.2, 5.3**

### Property 9: Pipeline Integration Interface

For any execution of the Critic_System, the system should call the provided retrieval function for sub-queries and the provided generation function for answer production, maintaining compatibility with the existing RAG pipeline components.

**Validates: Requirements 6.2**

### Property 10: Output Format Compatibility

For any answer produced by the Critic_System, the output should be a string in the same format as the baseline RAG pipeline, compatible with the HotpotQA official evaluation script.

**Validates: Requirements 6.3, 6.4, 7.2**

## Error Handling

### LLM Call Failures

**Query Decomposer Failures**:
- Malformed JSON response: Retry with clarified prompt (max 2 retries)
- Invalid sub-query count: If <1 or >3, use fallback of treating original question as single sub-query
- Missing reasoning plan: Generate default plan "Answer the question using retrieved information"
- LLM timeout/error: Fall back to baseline RAG pipeline (skip critic loop)

**Reasoning Checker Failures**:
- Malformed JSON response: Retry with clarified prompt (max 2 retries)
- Missing validation fields: Default to facts_retrieved=False, reasoning_valid=False
- LLM timeout/error: Default to PASS action (accept current answer)

### Retrieval Failures

**Sub-Query Retrieval Failures**:
- Empty results for sub-query: Continue with empty document list, let reasoning checker detect missing facts
- Retrieval function exception: Log error, return empty document list for that sub-query
- Timeout: Use partial results if available, otherwise empty list

**Retry Retrieval Failures**:
- If retry retrieval also fails: Proceed to regeneration step
- If retry exhausted (2 attempts): Accept current answer and increment iteration count

### Generation Failures

**Answer Generation Failures**:
- Empty answer: Treat as validation failure, trigger regeneration
- LLM timeout/error: If first generation, return baseline answer; if retry, return previous answer
- Malformed output: Attempt to extract answer text, if impossible return previous answer

### Iteration Limit Handling

**Max Iterations Reached**:
- Return most recent generated answer (even if validation failed)
- Log iteration history for debugging
- Include metadata indicating max iterations reached

### Input Validation

**Invalid Inputs**:
- Empty question: Return "I do not know."
- Empty initial answer: Trigger critic loop with question only
- Null/undefined inputs: Raise ValueError with descriptive message

## Testing Strategy

### Dual Testing Approach

The testing strategy employs both unit tests and property-based tests to ensure comprehensive coverage:

**Unit Tests** focus on:
- Specific examples of validation rules (e.g., exact "I do not know." string)
- Edge cases (empty inputs, boundary conditions like exactly 90 words)
- Error handling paths (LLM failures, malformed responses)
- Integration points (correct function calls to retrieval/generation)
- Example questions from HotpotQA dataset

**Property-Based Tests** focus on:
- Universal properties across all inputs (validation rule triggering logic)
- Invariants (iteration count never exceeds 2)
- Interface contracts (decomposer always returns 1-3 sub-queries)
- Decision logic (reasoning checker action selection)
- Round-trip properties (state consistency across iterations)

### Property-Based Testing Configuration

**Framework**: Hypothesis (Python)

**Configuration**:
- Minimum 100 iterations per property test
- Each test tagged with comment referencing design property
- Tag format: `# Feature: llm-critic-enhancement, Property {number}: {property_text}`

**Test Organization**:
```
tests/
├── unit/
│   ├── test_answer_validator.py
│   ├── test_query_decomposer.py
│   ├── test_sub_query_retriever.py
│   ├── test_generator.py
│   ├── test_reasoning_checker.py
│   └── test_critic_system.py
├── property/
│   ├── test_validation_properties.py
│   ├── test_decomposition_properties.py
│   ├── test_retrieval_properties.py
│   ├── test_reasoning_properties.py
│   └── test_iteration_properties.py
└── integration/
    ├── test_end_to_end.py
    └── test_hotpotqa_evaluation.py
```

### Unit Test Coverage

**Answer Validator**:
- Test exact "I do not know." match (positive and negative cases)
- Test length threshold at boundaries (89, 90, 91 words)
- Test each hedging phrase individually
- Test multiple entity detection with different conjunctions
- Test question type detection (Where, Who, What year, When)
- Test combination of multiple triggering conditions

**Query Decomposer**:
- Test decomposition with mock LLM responses (1, 2, 3 sub-queries)
- Test malformed JSON handling
- Test invalid sub-query counts (0, 4)
- Test missing reasoning plan handling
- Test LLM timeout/error handling

**Sub-Query Retriever**:
- Test retrieval for single sub-query
- Test retrieval for multiple sub-queries
- Test empty retrieval results
- Test retrieval function exceptions

**Generator**:
- Test generation with sub-query contexts
- Test context formatting
- Test empty answer handling
- Test LLM timeout/error handling

**Reasoning Checker**:
- Test validation with all facts present and reasoning valid
- Test validation with missing facts
- Test validation with invalid reasoning
- Test malformed JSON handling
- Test action selection logic

**Critic System**:
- Test full critic loop with passing validation on first iteration
- Test full critic loop with passing validation on second iteration
- Test full critic loop reaching max iterations
- Test early termination when validation passes
- Test integration with baseline RAG pipeline

### Property-Based Test Specifications

**Property 1: Answer Validation Rule Triggering**
```python
# Feature: llm-critic-enhancement, Property 1: Answer Validation Rule Triggering
@given(answer=st.text(), question=st.text())
@settings(max_examples=100)
def test_validation_rule_triggering(answer, question):
    result = validate_answer(answer, question)
    
    # Check if any trigger condition is met
    has_trigger = (
        answer == "I do not know." or
        len(answer.split()) > 90 or
        any(phrase in answer for phrase in HEDGING_PHRASES) or
        (question_starts_with_interrogative(question) and 
         has_multiple_entities(answer))
    )
    
    assert result.passed == (not has_trigger)
```

**Property 2: Query Decomposition Cardinality**
```python
# Feature: llm-critic-enhancement, Property 2: Query Decomposition Cardinality
@given(question=st.text(min_size=1))
@settings(max_examples=100)
def test_decomposition_cardinality(question, mock_llm):
    result = decompose_query(question)
    
    assert 1 <= len(result.sub_queries) <= 3
    assert len(result.reasoning_plan) > 0
```

**Property 3: Single Prompt Decomposition**
```python
# Feature: llm-critic-enhancement, Property 3: Single Prompt Decomposition
@given(question=st.text(min_size=1))
@settings(max_examples=100)
def test_single_prompt_decomposition(question, mock_llm):
    decompose_query(question)
    
    assert mock_llm.call_count == 1
```

**Property 4: Complete Sub-Query Retrieval**
```python
# Feature: llm-critic-enhancement, Property 4: Complete Sub-Query Retrieval
@given(sub_queries=st.lists(st.text(min_size=1), min_size=1, max_size=3))
@settings(max_examples=100)
def test_complete_subquery_retrieval(sub_queries, mock_retrieval, mock_generator):
    contexts = retrieve_for_subqueries(sub_queries, mock_retrieval)
    
    # Verify retrieval called for each sub-query
    assert len(contexts) == len(sub_queries)
    assert all(sq in contexts for sq in sub_queries)
    
    # Verify generator receives contexts and reasoning plan
    generate_with_reasoning("question", contexts, "plan", mock_generator)
    assert mock_generator.called
```

**Property 5: Single Prompt Reasoning Validation**
```python
# Feature: llm-critic-enhancement, Property 5: Single Prompt Reasoning Validation
@given(
    question=st.text(min_size=1),
    sub_queries=st.lists(st.text(min_size=1), min_size=1, max_size=3),
    answer=st.text(min_size=1)
)
@settings(max_examples=100)
def test_single_prompt_reasoning_validation(question, sub_queries, answer, mock_llm):
    result = check_reasoning(question, sub_queries, {}, "plan", answer)
    
    assert mock_llm.call_count == 1
    assert hasattr(result, 'action')
    assert hasattr(result, 'missing_facts')
```

**Property 6: Reasoning Checker Decision Logic**
```python
# Feature: llm-critic-enhancement, Property 6: Reasoning Checker Decision Logic
@given(
    facts_retrieved=st.booleans(),
    reasoning_valid=st.booleans()
)
@settings(max_examples=100)
def test_reasoning_checker_decision_logic(facts_retrieved, reasoning_valid, mock_llm):
    # Mock LLM to return specific validation results
    mock_llm.return_value = {
        "facts_retrieved": facts_retrieved,
        "reasoning_valid": reasoning_valid,
        "missing_facts": [] if facts_retrieved else ["sub1"],
        "reasoning_explanation": "test"
    }
    
    result = check_reasoning("q", ["sub1"], {}, "plan", "answer")
    
    if facts_retrieved and reasoning_valid:
        assert result.action == "PASS"
    elif not facts_retrieved:
        assert result.action == "RETRY_RETRIEVAL"
    else:  # facts_retrieved but not reasoning_valid
        assert result.action == "REGENERATE"
```

**Property 7: Maximum Iteration Limit**
```python
# Feature: llm-critic-enhancement, Property 7: Maximum Iteration Limit
@given(question=st.text(min_size=1), initial_answer=st.text())
@settings(max_examples=100)
def test_maximum_iteration_limit(question, initial_answer, mock_components):
    # Mock components to always fail validation
    mock_components.validator.return_value = ValidationResult(passed=False, triggered_rule="test")
    
    state = CriticState(question=question, current_answer=initial_answer, iteration_count=0)
    final_answer = run_critic_system(state, mock_components)
    
    assert state.iteration_count <= 2
```

**Property 8: Iteration Termination Conditions**
```python
# Feature: llm-critic-enhancement, Property 8: Iteration Termination Conditions
@given(
    question=st.text(min_size=1),
    initial_answer=st.text(),
    pass_on_iteration=st.integers(min_value=0, max_value=2)
)
@settings(max_examples=100)
def test_iteration_termination_conditions(question, initial_answer, pass_on_iteration, mock_components):
    # Mock validator to pass on specific iteration
    call_count = [0]
    def mock_validate(answer, question):
        call_count[0] += 1
        if call_count[0] == pass_on_iteration:
            return ValidationResult(passed=True, triggered_rule=None)
        return ValidationResult(passed=False, triggered_rule="test")
    
    mock_components.validator.side_effect = mock_validate
    
    state = CriticState(question=question, current_answer=initial_answer, iteration_count=0)
    final_answer = run_critic_system(state, mock_components)
    
    # Should terminate when validation passes or iteration reaches 2
    assert state.iteration_count == min(pass_on_iteration, 2)
```

**Property 9: Pipeline Integration Interface**
```python
# Feature: llm-critic-enhancement, Property 9: Pipeline Integration Interface
@given(question=st.text(min_size=1), initial_answer=st.text())
@settings(max_examples=100)
def test_pipeline_integration_interface(question, initial_answer, mock_retrieval, mock_generation):
    # Mock validation to trigger critic loop
    mock_validator = lambda a, q: ValidationResult(passed=False, triggered_rule="test")
    
    state = CriticState(question=question, current_answer=initial_answer, iteration_count=0)
    run_critic_system(state, mock_retrieval, mock_generation, mock_validator)
    
    # Verify existing pipeline functions were called
    assert mock_retrieval.called
    assert mock_generation.called
```

**Property 10: Output Format Compatibility**
```python
# Feature: llm-critic-enhancement, Property 10: Output Format Compatibility
@given(question=st.text(min_size=1), initial_answer=st.text())
@settings(max_examples=100)
def test_output_format_compatibility(question, initial_answer, mock_components):
    state = CriticState(question=question, current_answer=initial_answer, iteration_count=0)
    final_answer = run_critic_system(state, mock_components)
    
    # Output should be a string (same format as baseline)
    assert isinstance(final_answer, str)
    
    # Should be compatible with HotpotQA evaluation format
    # (evaluation script expects string answers)
    assert len(final_answer) >= 0  # Non-null string
```

### Integration Testing

**End-to-End Test**:
- Test complete critic loop with real LLM calls (using small model or mock)
- Test with sample HotpotQA questions
- Verify state transitions through full pipeline
- Measure iteration counts and validation outcomes

**HotpotQA Evaluation Test**:
- Test on stratified 300-question dataset
- Verify compatibility with official evaluation script
- Measure accuracy improvement over baseline
- Generate evaluation report with metrics

### Test Data Generation

**Hypothesis Strategies**:
```python
# Answer generation with specific properties
valid_answers = st.text(min_size=1, max_size=500).filter(
    lambda a: len(a.split()) <= 90 and 
              not any(p in a for p in HEDGING_PHRASES) and
              a != "I do not know."
)

long_answers = st.text(min_size=1).filter(lambda a: len(a.split()) > 90)

hedging_answers = st.text(min_size=1).map(
    lambda a: a + " however, the context does not specify"
)

# Question generation
interrogative_questions = st.sampled_from(["Where", "Who", "What year", "When"]).flatmap(
    lambda prefix: st.text(min_size=1).map(lambda q: f"{prefix} {q}")
)

# Sub-query lists
sub_query_lists = st.lists(st.text(min_size=1), min_size=1, max_size=3)
```

### Mocking Strategy

**LLM Mocking**:
- Mock LLM responses with valid JSON structures
- Test with various response formats (valid, malformed, edge cases)
- Simulate timeouts and errors
- Use deterministic responses for reproducibility

**Retrieval Mocking**:
- Mock document retrieval with synthetic documents
- Test with empty results, partial results, full results
- Simulate retrieval errors

**Component Mocking**:
- Mock individual components for unit testing
- Use dependency injection for testability
- Maintain clear interfaces between components

### Performance Testing

**Latency Benchmarks**:
- Measure end-to-end latency per question
- Measure component-level latency (decomposition, retrieval, generation, validation)
- Target: <10 seconds per question on average

**Throughput Testing**:
- Process full 300-question dataset
- Measure total processing time
- Target: Complete evaluation in <1 hour

**Resource Usage**:
- Monitor LLM API call counts
- Track token usage per question
- Measure memory usage during processing

