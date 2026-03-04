"""Integration module for wiring the Critic System to existing RAG pipelines.

This module provides adapter functions and wrappers to connect the critic system
to existing Jupyter notebook RAG pipelines. It handles format conversions between
the critic system's expected interfaces and the actual RAG pipeline implementations.

Requirements: 6.1, 6.2, 6.3
"""

import logging
from typing import Any, Callable, Dict, List, Optional

from src.critic.critic_system import run_critic_system
from src.critic.models import Document


logger = logging.getLogger(__name__)


def create_retrieval_adapter(
    retriever_or_function: Any,
    doc_text_field: str = "page_content",
    doc_source_field: str = "metadata",
    doc_score_field: Optional[str] = None
) -> Callable[[str], List[Dict]]:
    """
    Create an adapter function that converts various retriever formats to the
    critic system's expected format.
    
    The critic system expects a retrieval function that takes a query string
    and returns a list of dictionaries with 'text', 'source', and 'score' fields.
    
    This adapter handles:
    - LangChain retrievers (with .invoke() method)
    - Custom retrieval functions
    - Different document field naming conventions
    
    Args:
        retriever_or_function: Either a LangChain retriever object with .invoke()
                              method, or a callable that takes a query and returns
                              documents
        doc_text_field: Field name for document text content (default: "page_content")
        doc_source_field: Field name for document source/metadata (default: "metadata")
        doc_score_field: Optional field name for relevance score
    
    Returns:
        Callable that takes a query string and returns List[Dict] with standardized
        format: [{"text": str, "source": str, "score": float}, ...]
    
    Examples:
        >>> # For LangChain retriever
        >>> adapter = create_retrieval_adapter(langchain_retriever)
        >>> docs = adapter("What is RAG?")
        
        >>> # For custom function with different field names
        >>> adapter = create_retrieval_adapter(
        ...     custom_retriever,
        ...     doc_text_field="content",
        ...     doc_source_field="doc_id"
        ... )
    """
    def adapted_retrieval(query: str) -> List[Dict]:
        """Adapted retrieval function with standardized output format."""
        try:
            # Check if it's a LangChain retriever with invoke method
            if hasattr(retriever_or_function, 'invoke'):
                logger.debug(f"Using LangChain retriever.invoke() for query: {query}")
                raw_docs = retriever_or_function.invoke(query)
            # Otherwise assume it's a callable function
            elif callable(retriever_or_function):
                logger.debug(f"Using custom retrieval function for query: {query}")
                raw_docs = retriever_or_function(query)
            else:
                raise ValueError(
                    f"retriever_or_function must be callable or have .invoke() method, "
                    f"got {type(retriever_or_function)}"
                )
            
            # Convert to standardized format
            standardized_docs = []
            for doc in raw_docs:
                # Handle LangChain Document objects
                if hasattr(doc, doc_text_field):
                    text = getattr(doc, doc_text_field, "")
                    
                    # Extract source from metadata if it's a dict
                    if hasattr(doc, doc_source_field):
                        metadata = getattr(doc, doc_source_field, {})
                        if isinstance(metadata, dict):
                            source = metadata.get("source", metadata.get("title", ""))
                        else:
                            source = str(metadata)
                    else:
                        source = ""
                    
                    # Extract score if available
                    score = 0.0
                    if doc_score_field and hasattr(doc, doc_score_field):
                        score = float(getattr(doc, doc_score_field, 0.0))
                
                # Handle dictionary documents
                elif isinstance(doc, dict):
                    text = doc.get(doc_text_field, doc.get("text", ""))
                    source = doc.get(doc_source_field, doc.get("source", ""))
                    if isinstance(source, dict):
                        source = source.get("source", source.get("title", ""))
                    score = float(doc.get(doc_score_field, doc.get("score", 0.0)))
                
                else:
                    logger.warning(f"Unknown document format: {type(doc)}, skipping")
                    continue
                
                standardized_docs.append({
                    "text": str(text),
                    "source": str(source),
                    "score": score
                })
            
            logger.debug(f"Retrieved {len(standardized_docs)} documents for query: {query}")
            return standardized_docs
            
        except Exception as e:
            logger.error(f"Error in adapted retrieval for query '{query}': {e}")
            # Return empty list on error to allow critic system to continue
            return []
    
    return adapted_retrieval


def create_generation_adapter(
    llm_or_chain: Any,
    prompt_template: Optional[str] = None
) -> Callable[[str, str], str]:
    """
    Create an adapter function that converts various LLM/chain formats to the
    critic system's expected format.
    
    The critic system expects a generation function that takes a question and
    context string, and returns an answer string.
    
    This adapter handles:
    - LangChain LLMs (with .invoke() method)
    - LangChain chains
    - Custom generation functions
    - Optional prompt templates
    
    Args:
        llm_or_chain: Either a LangChain LLM/chain object with .invoke() method,
                     or a callable that takes inputs and returns an answer
        prompt_template: Optional prompt template string with {question} and
                        {context} placeholders. If None, uses a default template.
    
    Returns:
        Callable that takes (question: str, context: str) and returns answer: str
    
    Examples:
        >>> # For LangChain LLM
        >>> adapter = create_generation_adapter(langchain_llm)
        >>> answer = adapter("What is RAG?", "RAG stands for...")
        
        >>> # With custom prompt template
        >>> template = "Question: {question}\\nContext: {context}\\nAnswer:"
        >>> adapter = create_generation_adapter(llm, prompt_template=template)
    """
    # Default prompt template if none provided
    if prompt_template is None:
        prompt_template = (
            "Answer the following question using the provided context.\n\n"
            "Question: {question}\n\n"
            "Context:\n{context}\n\n"
            "Answer:"
        )
    
    def adapted_generation(question: str, context: str) -> str:
        """Adapted generation function with standardized interface."""
        try:
            # Format the prompt
            formatted_prompt = prompt_template.format(
                question=question,
                context=context
            )
            
            # Check if it's a LangChain LLM/chain with invoke method
            if hasattr(llm_or_chain, 'invoke'):
                logger.debug(f"Using LangChain LLM/chain.invoke() for question: {question[:50]}...")
                result = llm_or_chain.invoke(formatted_prompt)
                
                # Handle different return types
                if isinstance(result, str):
                    answer = result
                elif hasattr(result, 'content'):
                    answer = result.content
                elif isinstance(result, dict):
                    answer = result.get('answer', result.get('output', str(result)))
                else:
                    answer = str(result)
            
            # Otherwise assume it's a callable function
            elif callable(llm_or_chain):
                logger.debug(f"Using custom generation function for question: {question[:50]}...")
                result = llm_or_chain(formatted_prompt)
                answer = str(result)
            
            else:
                raise ValueError(
                    f"llm_or_chain must be callable or have .invoke() method, "
                    f"got {type(llm_or_chain)}"
                )
            
            logger.debug(f"Generated answer (first 100 chars): {answer[:100]}...")
            return answer
            
        except Exception as e:
            logger.error(f"Error in adapted generation for question '{question[:50]}...': {e}")
            # Return error message as answer to allow critic system to continue
            return "I do not know."
    
    return adapted_generation


def integrate_critic_with_rag(
    question: str,
    baseline_answer: str,
    retriever: Any,
    llm: Any,
    retrieval_adapter_kwargs: Optional[Dict] = None,
    generation_adapter_kwargs: Optional[Dict] = None,
    enable_logging: bool = True
) -> str:
    """
    Main integration function that wires the critic system to an existing RAG pipeline.
    
    This is the primary entry point for integrating the critic system with a Jupyter
    notebook RAG pipeline. It handles all the adapter creation and format conversion
    automatically.
    
    Args:
        question: The original question to answer
        baseline_answer: The initial answer from the baseline RAG pipeline
        retriever: The RAG pipeline's retriever (LangChain retriever or custom function)
        llm: The RAG pipeline's LLM (LangChain LLM/chain or custom function)
        retrieval_adapter_kwargs: Optional dict of kwargs for create_retrieval_adapter()
        generation_adapter_kwargs: Optional dict of kwargs for create_generation_adapter()
        enable_logging: Whether to enable detailed logging (default: True)
    
    Returns:
        Final answer string after critic system processing
    
    Examples:
        >>> # Basic usage with LangChain components
        >>> final_answer = integrate_critic_with_rag(
        ...     question="Where was the director of Inception born?",
        ...     baseline_answer="Christopher Nolan",
        ...     retriever=langchain_retriever,
        ...     llm=langchain_llm
        ... )
        
        >>> # With custom adapter settings
        >>> final_answer = integrate_critic_with_rag(
        ...     question="What is RAG?",
        ...     baseline_answer="RAG is...",
        ...     retriever=custom_retriever,
        ...     llm=custom_llm,
        ...     retrieval_adapter_kwargs={"doc_text_field": "content"},
        ...     generation_adapter_kwargs={"prompt_template": custom_template}
        ... )
    """
    # Configure logging
    if enable_logging:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    logger.info("=" * 80)
    logger.info("CRITIC SYSTEM INTEGRATION")
    logger.info("=" * 80)
    logger.info(f"Question: {question}")
    logger.info(f"Baseline answer: {baseline_answer}")
    logger.info("")
    
    # Create adapters with provided kwargs
    retrieval_adapter_kwargs = retrieval_adapter_kwargs or {}
    generation_adapter_kwargs = generation_adapter_kwargs or {}
    
    logger.info("Creating retrieval adapter...")
    retrieval_function = create_retrieval_adapter(retriever, **retrieval_adapter_kwargs)
    
    logger.info("Creating generation adapter...")
    generation_function = create_generation_adapter(llm, **generation_adapter_kwargs)
    
    # Run the critic system
    logger.info("Running critic system...")
    logger.info("")
    
    final_answer = run_critic_system(
        question=question,
        initial_answer=baseline_answer,
        retrieval_function=retrieval_function,
        generation_function=generation_function
    )
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("CRITIC SYSTEM COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Final answer: {final_answer}")
    logger.info("")
    
    return final_answer


# Convenience function for batch processing
def process_questions_with_critic(
    questions: List[str],
    baseline_answers: List[str],
    retriever: Any,
    llm: Any,
    retrieval_adapter_kwargs: Optional[Dict] = None,
    generation_adapter_kwargs: Optional[Dict] = None,
    enable_logging: bool = False
) -> List[str]:
    """
    Process multiple questions through the critic system.
    
    This is a convenience function for batch processing, useful for evaluation
    on the HotpotQA dataset.
    
    Args:
        questions: List of questions to process
        baseline_answers: List of baseline answers (must match length of questions)
        retriever: The RAG pipeline's retriever
        llm: The RAG pipeline's LLM
        retrieval_adapter_kwargs: Optional dict of kwargs for create_retrieval_adapter()
        generation_adapter_kwargs: Optional dict of kwargs for create_generation_adapter()
        enable_logging: Whether to enable detailed logging (default: False for batch)
    
    Returns:
        List of final answers after critic system processing
    
    Examples:
        >>> questions = ["Q1", "Q2", "Q3"]
        >>> baseline_answers = ["A1", "A2", "A3"]
        >>> final_answers = process_questions_with_critic(
        ...     questions, baseline_answers, retriever, llm
        ... )
    """
    if len(questions) != len(baseline_answers):
        raise ValueError(
            f"Length mismatch: {len(questions)} questions but "
            f"{len(baseline_answers)} baseline answers"
        )
    
    logger.info(f"Processing {len(questions)} questions with critic system...")
    
    final_answers = []
    for i, (question, baseline_answer) in enumerate(zip(questions, baseline_answers), 1):
        logger.info(f"Processing question {i}/{len(questions)}")
        
        final_answer = integrate_critic_with_rag(
            question=question,
            baseline_answer=baseline_answer,
            retriever=retriever,
            llm=llm,
            retrieval_adapter_kwargs=retrieval_adapter_kwargs,
            generation_adapter_kwargs=generation_adapter_kwargs,
            enable_logging=enable_logging
        )
        
        final_answers.append(final_answer)
    
    logger.info(f"Completed processing {len(questions)} questions")
    return final_answers
