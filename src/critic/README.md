# LLM Critic Enhancement Module

This module implements a critic system that enhances RAG pipeline answer quality through iterative validation, query decomposition, targeted retrieval, and reasoning validation.

## Module Structure

```
src/critic/
├── __init__.py          # Module exports
├── models.py            # Core data models
├── validator.py         # Answer validation component (to be implemented)
├── decomposer.py        # Query decomposition component (to be implemented)
├── retriever.py         # Sub-query retrieval component (to be implemented)
├── generator.py         # Answer generation component (to be implemented)
├── reasoning_checker.py # Reasoning validation component (to be implemented)
└── system.py            # Main critic system orchestrator (to be implemented)
```

## Data Models

### ValidationResult
Tracks whether an answer passes quality checks and which rule triggered failure.

**Fields:**
- `passed` (bool): Whether validation passed
- `triggered_rule` (Optional[str]): Name of failed rule or None

### DecompositionResult
Contains sub-queries (1-3) and reasoning plan from query decomposition.

**Fields:**
- `sub_queries` (List[str]): 1-3 sub-questions
- `reasoning_plan` (str): How sub-answers combine

**Constraints:**
- Must have 1-3 non-empty sub-queries
- Reasoning plan must be non-empty

### Document
Represents a retrieved document with metadata.

**Fields:**
- `text` (str): Document content
- `source` (str): Document identifier
- `score` (float): Relevance score

### ReasoningCheckResult
Result of reasoning validation with next action.

**Fields:**
- `action` (str): One of "PASS", "RETRY_RETRIEVAL", "REGENERATE"
- `missing_facts` (List[str]): Sub-queries with missing facts
- `reasoning_explanation` (str): Validation explanation

### CriticState
Tracks complete state across critic loop iterations.

**Fields:**
- `question` (str): Original question
- `current_answer` (str): Most recent answer
- `iteration_count` (int): Number of iterations completed
- `sub_queries` (Optional[List[str]]): Generated sub-queries
- `reasoning_plan` (Optional[str]): Reasoning plan
- `sub_query_contexts` (Optional[Dict[str, List[Document]]]): Retrieved documents
- `validation_history` (List[ValidationResult]): All validation results

## Testing

### Running Tests

With pytest installed:
```bash
pytest tests/unit/test_models.py -v
```

Without pytest (validation script):
```bash
python3 tests/validate_models.py
```

### Test Structure

```
tests/
├── conftest.py              # Hypothesis configuration
├── validate_models.py       # Standalone validation script
├── unit/                    # Unit tests
│   └── test_models.py       # Data model tests
├── property/                # Property-based tests (Hypothesis)
└── integration/             # End-to-end tests
```

## Hypothesis Configuration

Property-based tests use Hypothesis with the following settings:
- **max_examples**: 100 (minimum per design spec)
- **verbosity**: normal
- **deadline**: None (for LLM call tests)

Configuration is in `tests/conftest.py`.

## Usage Example

```python
from src.critic.models import ValidationResult, CriticState

# Create validation result
result = ValidationResult(passed=False, triggered_rule="length_exceeded")

# Initialize critic state
state = CriticState(
    question="What is the capital of France?",
    current_answer="Paris is the capital of France."
)

# Track validation
state.validation_history.append(result)
state.iteration_count += 1
```

## Requirements

See `requirements-dev.txt` for development dependencies:
- pytest >= 7.4.0
- hypothesis >= 6.82.0
- pytest-cov >= 4.1.0
