"""
Unit tests for Output Chunking and Streaming System.

Tests the ChunkedResult, ProgressiveProcessor, and ContentSummarizer
to ensure proper chunked output handling and streaming capabilities.
"""

import pytest
import time
import asyncio
from unittest.mock import Mock, AsyncMock, MagicMock
from typing import List, Dict, Any

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.pipeline.chunking import (
    ChunkedResult, ProgressiveProcessor, ContentSummarizer
)


class TestChunkedResult:
    """Test ChunkedResult functionality."""
    
    def test_init(self):
        """Test ChunkedResult initialization."""
        result = ChunkedResult(
            chunk_id="chunk_1",
            task_id="task_123",
            agent_id="agent_456",
            content_chunk="This is a test chunk.",
            chunk_index=0,
            is_final=False,
            timestamp=time.time(),
            continuation_token="token_abc"
        )
        
        assert result.chunk_id == "chunk_1"
        assert result.task_id == "task_123"
        assert result.agent_id == "agent_456"
        assert result.content_chunk == "This is a test chunk."
        assert result.chunk_index == 0
        assert not result.is_final
        assert result.continuation_token == "token_abc"
        assert isinstance(result.timestamp, float)
        assert isinstance(result.metadata, dict)
    
    def test_init_with_defaults(self):
        """Test ChunkedResult with default values."""
        result = ChunkedResult(
            chunk_id="chunk_1",
            task_id="task_123", 
            agent_id="agent_456",
            content_chunk="Test content",
            chunk_index=0,
            is_final=True,
            timestamp=time.time()
        )
        
        assert result.metadata == {}
        assert result.continuation_token is None


class TestProgressiveProcessor:
    """Test ProgressiveProcessor functionality."""
    
    def test_init(self):
        """Test ProgressiveProcessor initialization."""
        content = "This is a test content for chunking. " * 10  # 400+ chars
        
        processor = ProgressiveProcessor(
            task_id="task_123",
            agent_id="agent_456",
            full_content=content,
            chunk_size=100
        )
        
        assert processor.task_id == "task_123"
        assert processor.agent_id == "agent_456"
        assert processor.full_content == content
        assert processor.chunk_size == 100
        assert processor._current_chunk_index == 0
        assert processor.total_chunks > 1  # Should need multiple chunks
        assert len(processor._continuation_tokens) == processor.total_chunks
    
    def test_post_init_calculations(self):
        """Test post-initialization calculations."""
        content = "A" * 250  # 250 characters
        processor = ProgressiveProcessor(
            task_id="task_1",
            agent_id="agent_1",
            full_content=content,
            chunk_size=100
        )
        
        # Should need 3 chunks: 100 + 100 + 50
        assert processor.total_chunks == 3
        assert len(processor._continuation_tokens) == 3
        
        # All tokens should be unique
        tokens = list(processor._continuation_tokens.values())
        assert len(tokens) == len(set(tokens))
    
    def test_get_first_chunk(self):
        """Test getting the first chunk."""
        content = "This is chunk one. This is chunk two. This is chunk three."
        processor = ProgressiveProcessor(
            task_id="task_1",
            agent_id="agent_1",
            full_content=content,
            chunk_size=20
        )
        
        first_chunk = processor.get_next_chunk()
        
        assert first_chunk is not None
        assert first_chunk.chunk_index == 0
        assert first_chunk.content_chunk == content[:20]
        assert not first_chunk.is_final  # Should have more chunks
        assert first_chunk.continuation_token is not None
        assert first_chunk.task_id == "task_1"
        assert first_chunk.agent_id == "agent_1"
    
    def test_get_chunk_sequence(self):
        """Test getting a sequence of chunks."""
        content = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"  # 26 characters
        processor = ProgressiveProcessor(
            task_id="task_1",
            agent_id="agent_1",
            full_content=content,
            chunk_size=10
        )
        
        chunks = []
        current_token = None
        
        # Get all chunks
        while True:
            chunk = processor.get_next_chunk(current_token)
            if chunk is None:
                break
            chunks.append(chunk)
            current_token = chunk.continuation_token
            
            if chunk.is_final:
                break
        
        assert len(chunks) == 3  # 10 + 10 + 6 characters
        assert chunks[0].content_chunk == "ABCDEFGHIJ"
        assert chunks[1].content_chunk == "KLMNOPQRST"
        assert chunks[2].content_chunk == "UVWXYZ"
        
        # Only last chunk should be final
        assert not chunks[0].is_final
        assert not chunks[1].is_final
        assert chunks[2].is_final
        
        # Final chunk should have no continuation token
        assert chunks[2].continuation_token is None
    
    def test_get_chunk_with_invalid_token(self):
        """Test getting chunk with invalid continuation token."""
        content = "Test content for invalid token testing."
        processor = ProgressiveProcessor(
            task_id="task_1",
            agent_id="agent_1", 
            full_content=content,
            chunk_size=15
        )
        
        # Try with invalid token
        chunk = processor.get_next_chunk("invalid_token_12345")
        assert chunk is None
    
    def test_get_chunk_by_index(self):
        """Test getting chunk by specific index."""
        content = "Index test: " + "ABCDEFGHIJ" * 5  # 62 characters total
        processor = ProgressiveProcessor(
            task_id="task_1",
            agent_id="agent_1",
            full_content=content,
            chunk_size=20
        )
        
        # Get chunk at index 1
        chunk = processor.get_chunk_by_index(1)
        
        assert chunk is not None
        assert chunk.chunk_index == 1
        assert chunk.content_chunk == content[20:40]
        assert chunk.task_id == "task_1"
        
        # Get chunk at index 0
        first_chunk = processor.get_chunk_by_index(0)
        assert first_chunk is not None
        assert first_chunk.chunk_index == 0
        assert first_chunk.content_chunk == content[:20]
        
        # Get final chunk
        last_index = processor.total_chunks - 1
        last_chunk = processor.get_chunk_by_index(last_index)
        assert last_chunk is not None
        assert last_chunk.is_final
        assert last_chunk.continuation_token is None
    
    def test_get_chunk_by_invalid_index(self):
        """Test getting chunk with invalid index."""
        content = "Short content"
        processor = ProgressiveProcessor(
            task_id="task_1",
            agent_id="agent_1",
            full_content=content,
            chunk_size=50
        )
        
        # Index out of bounds
        assert processor.get_chunk_by_index(-1) is None
        assert processor.get_chunk_by_index(10) is None
    
    def test_single_chunk_content(self):
        """Test content that fits in a single chunk."""
        content = "Short"
        processor = ProgressiveProcessor(
            task_id="task_1",
            agent_id="agent_1",
            full_content=content,
            chunk_size=100
        )
        
        assert processor.total_chunks == 1
        
        chunk = processor.get_next_chunk()
        assert chunk is not None
        assert chunk.chunk_index == 0
        assert chunk.content_chunk == content
        assert chunk.is_final
        assert chunk.continuation_token is None
    
    def test_empty_content(self):
        """Test processing empty content."""
        processor = ProgressiveProcessor(
            task_id="task_1",
            agent_id="agent_1",
            full_content="",
            chunk_size=100
        )
        
        assert processor.total_chunks == 1  # Empty content still creates one chunk
        
        chunk = processor.get_next_chunk()
        assert chunk is not None
        assert chunk.content_chunk == ""
        assert chunk.is_final
    
    def test_chunk_metadata_consistency(self):
        """Test that chunk metadata is consistent across requests."""
        content = "Metadata consistency test content for chunking."
        processor = ProgressiveProcessor(
            task_id="task_123",
            agent_id="agent_456",
            full_content=content,
            chunk_size=15
        )
        
        # Get same chunk multiple times
        chunk1 = processor.get_chunk_by_index(0)
        chunk2 = processor.get_chunk_by_index(0)
        
        assert chunk1.task_id == chunk2.task_id
        assert chunk1.agent_id == chunk2.agent_id
        assert chunk1.chunk_index == chunk2.chunk_index
        assert chunk1.content_chunk == chunk2.content_chunk
        assert chunk1.is_final == chunk2.is_final
        # Note: chunk_id and timestamp will be different as they're generated fresh
    
    def test_large_content_performance(self):
        """Test performance with large content."""
        # Create large content (10KB)
        content = "Large content test. " * 500  # ~10,000 characters
        
        start_time = time.time()
        processor = ProgressiveProcessor(
            task_id="perf_test",
            agent_id="agent_1",
            full_content=content,
            chunk_size=1000
        )
        
        # Should handle large content quickly
        assert time.time() - start_time < 1.0  # Less than 1 second
        
        # Should create reasonable number of chunks
        assert processor.total_chunks <= 15  # 10KB / 1KB + buffer
        
        # First chunk should work quickly
        start_time = time.time()
        first_chunk = processor.get_next_chunk()
        assert time.time() - start_time < 0.1  # Very fast
        
        assert first_chunk is not None
        assert len(first_chunk.content_chunk) == 1000


class TestContentSummarizer:
    """Test ContentSummarizer functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_llm_client = Mock()
        self.summarizer = ContentSummarizer(self.mock_llm_client)
    
    def test_init(self):
        """Test ContentSummarizer initialization."""
        assert self.summarizer.llm_client == self.mock_llm_client
    
    @pytest.mark.asyncio
    async def test_summarize_content_short_content(self):
        """Test summarization of short content that doesn't need summarization."""
        short_content = "This is a short text."
        
        result = await self.summarizer.summarize_content(
            content=short_content,
            target_tokens=100,
            agent_id="agent_1",
            task_id="task_1"
        )
        
        # Short content should be returned as-is
        assert result == short_content
        self.mock_llm_client.complete_async.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_summarize_content_empty(self):
        """Test summarization of empty content."""
        result = await self.summarizer.summarize_content(
            content="",
            target_tokens=100,
            agent_id="agent_1",
            task_id="task_1"
        )
        
        assert result == ""
        self.mock_llm_client.complete_async.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_summarize_content_needs_summarization(self):
        """Test summarization of long content."""
        # Create long content that needs summarization (>100 words)
        long_content = "This is a very long piece of content that needs to be summarized. " * 50
        
        # Mock LLM response
        mock_response = Mock()
        mock_response.content = "This is a concise summary of the long content."
        self.mock_llm_client.complete_async = AsyncMock(return_value=mock_response)
        
        result = await self.summarizer.summarize_content(
            content=long_content,
            target_tokens=50,
            agent_id="agent_1",
            task_id="task_1"
        )
        
        assert result == "This is a concise summary of the long content."
        
        # Check LLM was called with correct parameters
        self.mock_llm_client.complete_async.assert_called_once()
        call_args = self.mock_llm_client.complete_async.call_args
        
        assert call_args.kwargs["agent_id"] == "agent_1"
        assert call_args.kwargs["max_tokens"] == 50
        assert call_args.kwargs["temperature"] == 0.3
        assert "Summarize the following content" in call_args.kwargs["user_prompt"]
        assert long_content in call_args.kwargs["user_prompt"]
    
    @pytest.mark.asyncio
    async def test_summarize_content_llm_error(self):
        """Test summarization fallback when LLM fails."""
        long_content = "Content that needs summarization. " * 100
        
        # Mock LLM to raise an exception
        self.mock_llm_client.complete_async = AsyncMock(side_effect=Exception("LLM Error"))
        
        result = await self.summarizer.summarize_content(
            content=long_content,
            target_tokens=50,
            agent_id="agent_1",
            task_id="task_1"
        )
        
        # Should fallback to simple truncation
        assert result.endswith("...")
        assert len(result) <= 200 + 3  # 50 tokens * 4 chars + "..."
    
    def test_simple_truncate(self):
        """Test simple truncation fallback."""
        content = "This is a test content that will be truncated."
        
        result = self.summarizer._simple_truncate(content, target_tokens=5)
        
        # Should be truncated to ~20 characters (5 tokens * 4 chars)
        assert len(result) <= 23  # 20 + "..."
        assert result.endswith("...")
    
    def test_simple_truncate_short_content(self):
        """Test simple truncation with content shorter than limit."""
        short_content = "Short"
        
        result = self.summarizer._simple_truncate(short_content, target_tokens=10)
        
        # Should return content as-is
        assert result == short_content
    
    @pytest.mark.asyncio
    async def test_system_prompt_includes_target_tokens(self):
        """Test that system prompt includes target token limit."""
        long_content = "Long content for testing system prompt. " * 20
        target_tokens = 75
        
        mock_response = Mock()
        mock_response.content = "Summary"
        self.mock_llm_client.complete_async = AsyncMock(return_value=mock_response)
        
        await self.summarizer.summarize_content(
            content=long_content,
            target_tokens=target_tokens,
            agent_id="agent_1",
            task_id="task_1"
        )
        
        call_args = self.mock_llm_client.complete_async.call_args
        system_prompt = call_args.kwargs["system_prompt"]
        
        assert str(target_tokens) in system_prompt
        assert "Content Summarizer" in system_prompt
        assert "essential information" in system_prompt


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""
    
    def test_chunked_blog_post_simulation(self):
        """Test chunking a realistic blog post."""
        # Simulate a blog post
        blog_content = """
        # The Future of Artificial Intelligence

        Artificial intelligence is rapidly transforming our world. In this comprehensive analysis, 
        we explore the key trends and implications for the future.

        ## Current State of AI

        Today's AI systems demonstrate remarkable capabilities in various domains including 
        natural language processing, computer vision, and decision-making. These systems are 
        being deployed across industries from healthcare to finance.

        ## Emerging Trends

        Several key trends are shaping the future of AI:
        1. Increased model sophistication
        2. Better human-AI collaboration
        3. Improved ethical frameworks
        4. Enhanced accessibility

        ## Challenges Ahead

        Despite progress, significant challenges remain including bias in algorithms, 
        privacy concerns, and the need for better interpretability.

        ## Conclusion

        The future of AI holds great promise, but requires careful consideration of ethical 
        implications and societal impact.
        """
        
        processor = ProgressiveProcessor(
            task_id="blog_post",
            agent_id="blog_writer",
            full_content=blog_content.strip(),
            chunk_size=300
        )
        
        chunks = []
        current_token = None
        
        while True:
            chunk = processor.get_next_chunk(current_token)
            if chunk is None:
                break
            chunks.append(chunk)
            current_token = chunk.continuation_token
            if chunk.is_final:
                break
        
        # Should create multiple chunks
        assert len(chunks) >= 3
        
        # Reconstruct content from chunks
        reconstructed = "".join(chunk.content_chunk for chunk in chunks)
        assert reconstructed == blog_content.strip()
        
        # Check chunk properties
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i
            assert chunk.task_id == "blog_post"
            assert chunk.agent_id == "blog_writer"
        
        # Only last chunk should be final
        for chunk in chunks[:-1]:
            assert not chunk.is_final
            assert chunk.continuation_token is not None
        
        assert chunks[-1].is_final
        assert chunks[-1].continuation_token is None
    
    @pytest.mark.asyncio
    async def test_summarization_fallback_scenario(self):
        """Test realistic summarization scenario with fallback."""
        # Long research content that exceeds token limit
        research_content = """
        Research Analysis: Climate Change Impact on Agricultural Systems
        
        Executive Summary: This comprehensive study examines the multifaceted impacts 
        of climate change on global agricultural systems, analyzing temperature variations,
        precipitation patterns, and extreme weather events across multiple geographic regions.
        
        Methodology: We employed a mixed-methods approach combining quantitative climate data
        analysis with qualitative assessments from agricultural stakeholders across 15 countries.
        
        Key Findings:
        1. Temperature increases of 2-3°C significantly reduce crop yields in tropical regions
        2. Changing precipitation patterns affect irrigation-dependent systems most severely  
        3. Extreme weather events cause both immediate and long-term agricultural disruption
        4. Adaptation strategies show varying effectiveness across different crop types
        
        Regional Analysis: Sub-Saharan Africa shows greatest vulnerability while Northern
        European regions may experience some agricultural benefits from moderate warming.
        
        Recommendations: Immediate implementation of climate-resilient farming practices,
        investment in drought-resistant crop varieties, and improved early warning systems.
        """ * 3  # Triple the content to ensure it needs summarization
        
        mock_llm_client = Mock()
        mock_response = Mock()
        mock_response.content = "Climate change significantly impacts global agriculture through temperature increases, changing precipitation, and extreme weather. Key recommendations include climate-resilient farming and drought-resistant crops."
        
        mock_llm_client.complete_async = AsyncMock(return_value=mock_response)
        
        summarizer = ContentSummarizer(mock_llm_client)
        
        summary = await summarizer.summarize_content(
            content=research_content,
            target_tokens=100,
            agent_id="research_agent",
            task_id="climate_research"
        )
        
        # Should get summarized version
        assert summary != research_content
        assert "climate change" in summary.lower()
        assert "agriculture" in summary.lower()
        assert len(summary) < len(research_content)
        
        # Verify LLM was called with research content
        mock_llm_client.complete_async.assert_called_once()
        call_args = mock_llm_client.complete_async.call_args
        assert "climate change" in call_args.kwargs["user_prompt"].lower()
    
    def test_progressive_synthesis_workflow(self):
        """Test a progressive synthesis workflow with multiple agents."""
        # Simulate multiple agent contributions
        contributions = [
            "Research findings on quantum computing fundamentals and current capabilities.",
            "Analysis of market trends and commercial applications in the quantum computing sector.",
            "Technical review of hardware approaches: superconducting, trapped ion, and photonic systems.",
            "Risk assessment and timeline projections for quantum computing milestones."
        ]
        
        # Each contribution gets processed into chunks
        processors = []
        all_chunks = []
        
        for i, contribution in enumerate(contributions):
            processor = ProgressiveProcessor(
                task_id=f"synthesis_task",
                agent_id=f"agent_{i+1}",
                full_content=contribution,
                chunk_size=50
            )
            processors.append(processor)
            
            # Get first chunk from each processor
            first_chunk = processor.get_next_chunk()
            if first_chunk:
                all_chunks.append(first_chunk)
        
        # Should have chunks from multiple agents
        assert len(all_chunks) == 4
        
        # Each chunk should have different agent_id but same task_id
        agent_ids = {chunk.agent_id for chunk in all_chunks}
        task_ids = {chunk.task_id for chunk in all_chunks}
        
        assert len(agent_ids) == 4  # Four different agents
        assert len(task_ids) == 1   # Same task
        assert "synthesis_task" in task_ids


if __name__ == "__main__":
    pytest.main([__file__, "-v"])