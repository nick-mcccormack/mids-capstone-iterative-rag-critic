"""Integration tests for RAG pipeline integration.

Tests the integration module's ability to connect the critic system
to various RAG pipeline formats.

Requirements: 6.1, 6.2, 6.3
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from typing import List, Dict

from src.critic.integration import (
    create_retrieval_adapter,
    create_generation_adapter,
    integrate_critic_with_rag
)


class TestRetrievalAdapter:
    """Test retrieval adapter creation and format conversion."""
    
    def test_langchain_retriever_adapter(self):
        """Test adapter works with LangChain retriever format."""
        # Mock LangChain retriever
        mock_retriever = Mock()
        mock_doc = Mock()
        mock_doc.page_content = "Test document content"
        mock_doc.metadata = {"source": "test.pdf"}
        mock_retriever.invoke.return_value = [mock_doc]
        
        # Create adapter
        adapter = create_retrieval_adapter(mock_retriever)
        
        # Test retrieval
        result = adapter("test query")
        
        # Verify format
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["text"] == "Test document content"
        assert result[0]["source"] == "test.pdf"
        assert "score" in result[0]
        
        # Verify retriever was called
        mock_retriever.invoke.assert_called_once_with("test query")
    
    def test_custom_function_adapter(self):
        """Test adapter works with custom retrieval function."""
        # Mock custom retrieval function
        def mock_retrieval(query: str) -> List[Dict]:
            return [
                {"text": "Doc 1", "source": "source1", "score": 0.9},
                {"text": "Doc 2", "source": "source2", "score": 0.8}
            ]
        
        # Create adapter
        adapter = create_retrieval_adapter(mock_retrieval)
        
        # Test retrieval
        result = adapter("test query")
        
        # Verify format
        assert len(result) == 2
        assert result[0]["text"] == "Doc 1"
        assert result[0]["score"] == 0.9
    
    def test_custom_field_names(self):
        """Test adapter with custom field names."""
        # Mock retriever with custom field names
        mock_retriever = Mock()
        mock_doc = Mock()
        mock_doc.content = "Custom content"
        mock_doc.doc_id = "doc123"
        mock_doc.relevance = 0.95
        mock_retriever.invoke.return_value = [mock_doc]
        
        # Create adapter with custom field names
        adapter = create_retrieval_adapter(
            mock_retriever,
            doc_text_field="content",
            doc_source_field="doc_id",
            doc_score_field="relevance"
        )
        
        # Test retrieval
        result = adapter("test query")
        
        # Verify format conversion
        assert result[0]["text"] == "Custom content"
        assert result[0]["source"] == "doc123"
        assert result[0]["score"] == 0.95
    
    def test_empty_retrieval_results(self):
        """Test adapter handles empty retrieval results."""
        mock_retriever = Mock()
        mock_retriever.invoke.return_value = []
        
        adapter = create_retrieval_adapter(mock_retriever)
        result = adapter("test query")
        
        assert result == []
    
    def test_retrieval_error_handling(self):
        """Test adapter handles retrieval errors gracefully."""
        mock_retriever = Mock()
        mock_retriever.invoke.side_effect = Exception("Retrieval failed")
        
        adapter = create_retrieval_adapter(mock_retriever)
        result = adapter("test query")
        
        # Should return empty list on error
        assert result == []


class TestGenerationAdapter:
    """Test generation adapter creation and format conversion."""
    
    def test_langchain_llm_adapter(self):
        """Test adapter works with LangChain LLM format."""
        # Mock LangChain LLM
        mock_llm = Mock()
        mock_llm.invoke.return_value = "Generated answer"
        
        # Create adapter
        adapter = create_generation_adapter(mock_llm)
        
        # Test generation
        result = adapter("What is RAG?", "RAG stands for...")
        
        # Verify format
        assert isinstance(result, str)
        assert result == "Generated answer"
        
        # Verify LLM was called with formatted prompt
        mock_llm.invoke.assert_called_once()
        call_args = mock_llm.invoke.call_args[0][0]
        assert "What is RAG?" in call_args
        assert "RAG stands for..." in call_args
    
    def test_langchain_chat_model_adapter(self):
        """Test adapter works with LangChain chat model format."""
        # Mock LangChain chat model (returns object with .content)
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "Chat model answer"
        mock_llm.invoke.return_value = mock_response
        
        # Create adapter
        adapter = create_generation_adapter(mock_llm)
        
        # Test generation
        result = adapter("Question", "Context")
        
        # Verify format
        assert result == "Chat model answer"
    
    def test_custom_function_adapter(self):
        """Test adapter works with custom generation function."""
        # Mock custom generation function
        def mock_generation(prompt: str) -> str:
            return f"Answer to: {prompt[:20]}..."
        
        # Create adapter
        adapter = create_generation_adapter(mock_generation)
        
        # Test generation
        result = adapter("Question", "Context")
        
        # Verify format
        assert isinstance(result, str)
        assert "Answer to:" in result
    
    def test_custom_prompt_template(self):
        """Test adapter with custom prompt template."""
        mock_llm = Mock()
        mock_llm.invoke.return_value = "Answer"
        
        # Create adapter with custom template
        custom_template = "Q: {question}\nC: {context}\nA:"
        adapter = create_generation_adapter(mock_llm, prompt_template=custom_template)
        
        # Test generation
        adapter("What is RAG?", "RAG stands for...")
        
        # Verify custom template was used
        call_args = mock_llm.invoke.call_args[0][0]
        assert "Q: What is RAG?" in call_args
        assert "C: RAG stands for..." in call_args
    
    def test_generation_error_handling(self):
        """Test adapter handles generation errors gracefully."""
        mock_llm = Mock()
        mock_llm.invoke.side_effect = Exception("Generation failed")
        
        adapter = create_generation_adapter(mock_llm)
        result = adapter("Question", "Context")
        
        # Should return "I do not know." on error
        assert result == "I do not know."


class TestIntegration:
    """Test end-to-end integration with critic system."""
    
    def test_integrate_with_mock_components(self):
        """Test integration with mocked RAG components."""
        # Mock retriever
        mock_retriever = Mock()
        mock_doc = Mock()
        mock_doc.page_content = "Test content"
        mock_doc.metadata = {"source": "test"}
        mock_retriever.invoke.return_value = [mock_doc]
        
        # Mock LLM
        mock_llm = Mock()
        mock_llm.invoke.return_value = "Final answer"
        
        # Test integration
        result = integrate_critic_with_rag(
            question="Test question?",
            baseline_answer="London",  # Short answer that passes validation
            retriever=mock_retriever,
            llm=mock_llm,
            enable_logging=False
        )
        
        # Verify result is a string
        assert isinstance(result, str)
        
        # Should return baseline answer since it passes validation
        assert result == "London"
    
    def test_integrate_with_passing_baseline(self):
        """Test integration when baseline answer passes validation."""
        # Mock components
        mock_retriever = Mock()
        mock_llm = Mock()
        
        # Baseline answer that should pass validation (short, no hedging)
        baseline_answer = "London"
        
        # Test integration
        result = integrate_critic_with_rag(
            question="Where was he born?",
            baseline_answer=baseline_answer,
            retriever=mock_retriever,
            llm=mock_llm,
            enable_logging=False
        )
        
        # Should return baseline answer without modification
        assert result == baseline_answer
        
        # Retriever and LLM should not be called if validation passes
        assert not mock_retriever.invoke.called
        assert not mock_llm.invoke.called
    
    @patch('src.critic.decomposer.call_llm')
    @patch('src.critic.reasoning_checker.call_llm')
    def test_integrate_with_failing_baseline(self, mock_reasoning_llm, mock_decomposer_llm):
        """Test integration when baseline answer fails validation."""
        # Mock the internal LLM calls for decomposer
        mock_decomposer_llm.return_value = '''
        {
            "sub_queries": ["Where was Christopher Nolan born?"],
            "reasoning_plan": "Find the birthplace of Christopher Nolan"
        }
        '''
        
        # Mock the internal LLM calls for reasoning checker
        mock_reasoning_llm.return_value = '''
        {
            "facts_retrieved": true,
            "missing_facts": [],
            "reasoning_valid": true,
            "reasoning_explanation": "All facts retrieved and reasoning is valid"
        }
        '''
        
        # Mock retriever
        mock_retriever = Mock()
        mock_doc = Mock()
        mock_doc.page_content = "Christopher Nolan was born in London"
        mock_doc.metadata = {"source": "bio"}
        mock_retriever.invoke.return_value = [mock_doc]
        
        # Mock LLM
        mock_llm = Mock()
        mock_llm.invoke.return_value = "London"
        
        # Baseline answer that should fail validation (too long)
        baseline_answer = " ".join(["word"] * 100)  # 100 words
        
        # Test integration
        result = integrate_critic_with_rag(
            question="Where was Christopher Nolan born?",
            baseline_answer=baseline_answer,
            retriever=mock_retriever,
            llm=mock_llm,
            enable_logging=False
        )
        
        # Should return a different answer
        assert isinstance(result, str)
        
        # Retriever and LLM should be called
        assert mock_retriever.invoke.called
        assert mock_llm.invoke.called
    
    def test_output_format_compatibility(self):
        """Test that output format matches baseline pipeline format."""
        # Mock components
        mock_retriever = Mock()
        mock_retriever.invoke.return_value = []
        mock_llm = Mock()
        mock_llm.invoke.return_value = "Answer"
        
        # Test integration
        result = integrate_critic_with_rag(
            question="Test?",
            baseline_answer="Baseline",
            retriever=mock_retriever,
            llm=mock_llm,
            enable_logging=False
        )
        
        # Output should be a plain string (same as baseline)
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_custom_adapter_kwargs(self):
        """Test integration with custom adapter kwargs."""
        # Mock retriever with custom fields
        mock_retriever = Mock()
        mock_doc = Mock()
        mock_doc.content = "Custom content"
        mock_doc.doc_id = "doc1"
        mock_retriever.invoke.return_value = [mock_doc]
        
        # Mock LLM
        mock_llm = Mock()
        mock_llm.invoke.return_value = "Answer"
        
        # Test with custom adapter kwargs
        result = integrate_critic_with_rag(
            question="Test?",
            baseline_answer="Baseline",
            retriever=mock_retriever,
            llm=mock_llm,
            retrieval_adapter_kwargs={
                "doc_text_field": "content",
                "doc_source_field": "doc_id"
            },
            generation_adapter_kwargs={
                "prompt_template": "Q: {question}\nC: {context}\nA:"
            },
            enable_logging=False
        )
        
        # Should work with custom adapters
        assert isinstance(result, str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
