# Task 3.1 Implementation Summary

## Overview
Successfully implemented the `decompose_query` function with LLM integration for the Query Decomposer component of the LLM Critic Enhancement system.

## Files Created/Modified

### 1. `src/critic/decomposer.py` (NEW)
Main implementation file containing:

#### Core Function: `decompose_query(question: str, max_retries: int = 2) -> DecompositionResult`
- **Purpose**: Decomposes multi-hop questions into 1-3 sub-queries with reasoning plan
- **LLM Integration**: Uses `call_llm` from `src.rag.generator` module
- **Single Prompt**: Makes one LLM call per attempt (satisfies Requirement 2.3)
- **JSON Parsing**: Parses structured JSON response with sub-queries and reasoning plan
- **Validation**: Enforces 1-3 sub-queries constraint (satisfies Requirement 2.1)
- **Error Handling**: Implements comprehensive error handling as specified

#### Error Handling Features:
1. **Malformed JSON Retry Logic** (max 2 retries):
   - Catches `json.JSONDecodeError`
   - Retries with clarified prompt emphasizing JSON-only output
   - Falls back to single sub-query after max retries

2. **Invalid Sub-Query Count Fallback**:
   - Detects counts <1 or >3
   - Falls back to treating original question as single sub-query
   - Uses default reasoning plan

3. **Missing Reasoning Plan Handling**:
   - Detects empty or missing reasoning_plan field
   - Substitutes default: "Answer the question using retrieved information"

4. **Input Validation**:
   - Raises `ValueError` for empty/None/whitespace-only questions
   - Validates sub-queries are non-empty strings

#### Helper Functions:
- `_call_decomposition_llm()`: Wraps LLM call with prompt construction
- `_create_system_prompt()`: Generates system prompt (with retry clarification)
- `_create_user_prompt()`: Generates user prompt with question and JSON format
- `_create_fallback_decomposition()`: Creates fallback single sub-query result

#### LLM Prompt Design:
**System Prompt**:
- Explains task: break question into 1-3 sub-queries + reasoning plan
- Specifies requirements: 1-3 queries, independently answerable, clear language
- Adds clarification on retry attempts

**User Prompt**:
- Includes the question
- Provides JSON output format with example structure
- Emphasizes "ONLY JSON, no additional text"

### 2. `src/critic/__init__.py` (MODIFIED)
- Added import: `from src.critic.decomposer import decompose_query`
- Added to `__all__` exports

### 3. `tests/unit/test_decomposer.py` (NEW)
Comprehensive unit test suite with 17 test cases:

**Successful Decomposition Tests**:
- 1 sub-query
- 2 sub-queries
- 3 sub-queries (maximum)

**Error Handling Tests**:
- Malformed JSON with successful retry
- Malformed JSON max retries → fallback
- Invalid count: 0 sub-queries → fallback
- Invalid count: 4 sub-queries → fallback
- Missing reasoning plan → default
- Missing reasoning_plan field → default
- Empty sub-query strings → fallback
- Non-string sub-queries → fallback

**Input Validation Tests**:
- Empty question → ValueError
- None question → ValueError
- Whitespace-only question → ValueError

**Fallback Tests**:
- Fallback creates single sub-query correctly

### 4. `tests/validate_decomposer.py` (NEW)
Standalone validation script (no pytest required) with 11 test cases covering:
- All successful decomposition scenarios
- Retry logic
- Fallback scenarios
- Input validation
- Fallback function

## Requirements Satisfied

✅ **Requirement 2.1**: Generate between 1 and 3 sub-queries
- Validated in `DecompositionResult.__post_init__`
- Fallback ensures constraint is always met

✅ **Requirement 2.2**: Produce reasoning plan that chains intermediate answers
- Required field in `DecompositionResult`
- Default provided if missing from LLM response

✅ **Requirement 2.3**: Use single LLM prompt for both sub-queries and reasoning plan
- Single `call_llm()` invocation per attempt
- JSON response contains both fields

✅ **Requirement 2.4**: Reasoning plan specifies how intermediate answers combine
- Prompt explicitly requests this
- Example format provided in user prompt

## Design Compliance

✅ **LLM Prompt Template**: Implemented as specified in design document
✅ **JSON Parsing**: Robust parsing with error handling
✅ **1-3 Sub-queries Constraint**: Validated and enforced
✅ **Retry Logic**: Max 2 retries for malformed JSON
✅ **Fallback**: Invalid counts trigger single sub-query fallback
✅ **Error Handling**: All specified error cases handled

## Testing Status

**Code Quality**:
- ✅ No syntax errors (verified with getDiagnostics)
- ✅ No linting issues
- ✅ Proper type hints
- ✅ Comprehensive docstrings

**Test Coverage**:
- ✅ 17 unit tests created
- ✅ 11 validation tests created
- ⚠️ Tests not executed (dependencies not installed in environment)

**Note**: Tests cannot be executed in current environment due to missing dependencies:
- `groq` module (required by `src.rag.generator`)
- `pytest` and `hypothesis` (testing frameworks)

Tests are ready to run once dependencies are installed via:
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

## Integration Points

**Imports**:
- `src.critic.models.DecompositionResult`: Return type
- `src.rag.generator.call_llm`: LLM integration

**Exports**:
- Available via `from src.critic import decompose_query`
- Available via `from src.critic.decomposer import decompose_query`

## Next Steps

1. **Install dependencies** to run tests:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

2. **Run unit tests**:
   ```bash
   pytest tests/unit/test_decomposer.py -v
   ```

3. **Run validation script**:
   ```bash
   python3 tests/validate_decomposer.py
   ```

4. **Proceed to Task 3.2**: Write property test for decomposition cardinality

## Implementation Notes

- **Temperature**: Set to 0.2 for more deterministic decomposition
- **Logging**: Uses Python logging module for warnings and errors
- **Retry Strategy**: Clarifies prompt on retry to emphasize JSON-only output
- **Fallback Strategy**: Conservative approach - treats original question as single sub-query
- **Validation**: Leverages `DecompositionResult.__post_init__` for constraint validation
