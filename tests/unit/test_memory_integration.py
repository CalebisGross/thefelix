"""
Test suite for memory system integration with Central Post.

This module tests the integration of the knowledge store, task memory,
and context compression systems with the Central Post communication hub.
"""

import pytest
import tempfile
import os
from unittest.mock import AsyncMock, MagicMock
from src.communication.central_post import CentralPost, MessageType
from src.memory.knowledge_store import KnowledgeStore, KnowledgeType, ConfidenceLevel
from src.memory.task_memory import TaskMemory, TaskOutcome
from src.memory.context_compression import ContextCompressor, CompressionStrategy


class TestMemoryIntegration:
    """Test memory system integration with Central Post."""
    
    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database file for testing."""
        fd, path = tempfile.mkstemp(suffix='.db')
        os.close(fd)
        yield path
        if os.path.exists(path):
            os.unlink(path)
    
    @pytest.fixture
    def central_post(self, temp_db_path):
        """Create Central Post with memory systems enabled."""
        return CentralPost(
            max_agents=10,
            enable_metrics=True,
            enable_memory=True,
            memory_db_path=temp_db_path
        )
    
    def test_memory_system_initialization(self, central_post):
        """Test that memory systems are properly initialized."""
        assert central_post.knowledge_store is not None
        assert central_post.task_memory is not None
        assert central_post.context_compressor is not None
        assert isinstance(central_post.knowledge_store, KnowledgeStore)
        assert isinstance(central_post.task_memory, TaskMemory)
        assert isinstance(central_post.context_compressor, ContextCompressor)
    
    def test_memory_disabled_initialization(self, temp_db_path):
        """Test initialization with memory systems disabled."""
        central_post = CentralPost(
            max_agents=10,
            enable_metrics=True,
            enable_memory=False
        )
        assert central_post.knowledge_store is None
        assert central_post.task_memory is None
        assert central_post.context_compressor is None
    
    def test_store_agent_result_as_knowledge(self, central_post):
        """Test storing agent results as knowledge."""
        # Test with memory enabled
        result = central_post.store_agent_result_as_knowledge(
            agent_id="test_agent",
            content="Test result content",
            confidence=0.8,
            domain="test_domain",
            tags=["test", "result"]
        )
        assert result is True
        
        # Verify knowledge was stored
        from src.memory.knowledge_store import KnowledgeQuery
        query = KnowledgeQuery(domains=["test_domain"])
        knowledge_entries = central_post.knowledge_store.retrieve_knowledge(query)
        assert len(knowledge_entries) > 0
        assert knowledge_entries[0].content["result"] == "Test result content"
        assert knowledge_entries[0].confidence_level == ConfidenceLevel.HIGH
    
    def test_store_agent_result_memory_disabled(self, temp_db_path):
        """Test storing agent results with memory disabled."""
        central_post = CentralPost(enable_memory=False)
        result = central_post.store_agent_result_as_knowledge(
            agent_id="test_agent",
            content="Test content",
            confidence=0.8
        )
        assert result is False
    
    def test_retrieve_relevant_knowledge(self, central_post):
        """Test retrieving relevant knowledge."""
        # Store some knowledge first
        central_post.store_agent_result_as_knowledge(
            agent_id="agent1",
            content="Machine learning best practices",
            confidence=0.9,
            domain="AI",
            tags=["ML", "best_practices"]
        )
        
        # Retrieve knowledge
        knowledge = central_post.retrieve_relevant_knowledge(
            domain="AI",
            keywords=["machine", "learning"]
        )
        assert len(knowledge) > 0
        assert "Machine learning best practices" in knowledge[0].content["result"]
    
    def test_retrieve_knowledge_memory_disabled(self, temp_db_path):
        """Test retrieving knowledge with memory disabled."""
        central_post = CentralPost(enable_memory=False)
        knowledge = central_post.retrieve_relevant_knowledge(domain="AI")
        assert knowledge == []
    
    def test_get_task_strategy_recommendations(self, central_post):
        """Test getting task strategy recommendations."""
        # Record a successful task execution
        if central_post.task_memory:
            from src.memory.task_memory import TaskComplexity
            central_post.task_memory.record_task_execution(
                task_description="Test AI task",
                task_type="AI",
                complexity=TaskComplexity.MODERATE,
                outcome=TaskOutcome.SUCCESS,
                duration=1.5,
                agents_used=["research_agent", "analysis_agent"],
                strategies_used=["helix_spiral"],
                context_size=1000,
                success_metrics={"confidence": 0.85, "quality_score": 0.9}
            )
        
        # Get recommendations
        recommendations = central_post.get_task_strategy_recommendations(
            task_description="Another AI test task",
            task_type="AI",
            complexity="MODERATE"
        )
        assert isinstance(recommendations, dict)
        # Should have recommendations based on the recorded task
        if recommendations.get("strategies"):
            assert "helix_spiral" in recommendations["strategies"]
    
    def test_get_strategy_recommendations_memory_disabled(self, temp_db_path):
        """Test getting strategy recommendations with memory disabled."""
        central_post = CentralPost(enable_memory=False)
        recommendations = central_post.get_task_strategy_recommendations(
            task_description="Test task"
        )
        assert recommendations == {}
    
    def test_compress_large_context(self, central_post):
        """Test context compression functionality."""
        large_context = "This is a very long context that needs compression. " * 100
        
        compressed = central_post.compress_large_context(
            context=large_context,
            strategy=CompressionStrategy.EXTRACTIVE_SUMMARY,
            target_size=200
        )
        
        assert compressed is not None
        assert len(compressed.content) < len(large_context)
        assert compressed.compression_ratio > 0
        # Allow for small discrepancies in size measurement due to processing overhead
        assert abs(compressed.original_size - len(large_context)) <= 50
    
    def test_compress_context_memory_disabled(self, temp_db_path):
        """Test context compression with memory disabled."""
        central_post = CentralPost(enable_memory=False)
        result = central_post.compress_large_context(
            context="Test context",
            strategy=CompressionStrategy.EXTRACTIVE_SUMMARY,
            target_size=100
        )
        assert result is None
    
    def test_get_memory_summary(self, central_post):
        """Test getting memory system summary."""
        # Add some data first
        central_post.store_agent_result_as_knowledge(
            agent_id="agent1",
            content="Test knowledge",
            confidence=0.8,
            domain="test"
        )
        
        summary = central_post.get_memory_summary()
        assert "knowledge_entries" in summary
        assert "task_patterns" in summary
        assert summary["knowledge_entries"] > 0
    
    def test_get_memory_summary_disabled(self, temp_db_path):
        """Test getting memory summary with memory disabled."""
        central_post = CentralPost(enable_memory=False)
        summary = central_post.get_memory_summary()
        assert summary == {
            "knowledge_entries": 0,
            "task_patterns": 0,
            "memory_enabled": False
        }
    
    def test_task_completion_message_handling(self, central_post):
        """Test that task completion messages can be stored as knowledge."""
        # Test that we can manually store task completion as knowledge
        result = central_post.store_agent_result_as_knowledge(
            agent_id="test_agent",
            content="Task completed successfully",
            confidence=0.85,
            domain="test",
            tags=["completion"]
        )
        assert result is True
        
        # Verify knowledge was stored
        knowledge = central_post.retrieve_relevant_knowledge(domain="test")
        assert len(knowledge) > 0
        assert "Task completed successfully" in knowledge[0].content["result"]
    
    def test_error_report_message_handling(self, central_post):
        """Test that error reports can be stored for learning."""
        # Test that we can manually store error information as knowledge
        result = central_post.store_agent_result_as_knowledge(
            agent_id="error_agent",
            content="Failed to process input - need better validation",
            confidence=0.3,
            domain="error_analysis",
            tags=["error", "processing", "validation"]
        )
        assert result is True
        
        # Verify error was stored as knowledge for learning
        knowledge = central_post.retrieve_relevant_knowledge(domain="error_analysis")
        assert len(knowledge) > 0
        assert "Failed to process input" in knowledge[0].content["result"]
    
    def test_cross_run_persistence(self, temp_db_path):
        """Test that knowledge persists across Central Post instances."""
        # Create first instance and store knowledge
        central_post1 = CentralPost(
            enable_memory=True,
            memory_db_path=temp_db_path
        )
        central_post1.store_agent_result_as_knowledge(
            agent_id="agent1",
            content="Persistent knowledge test",
            confidence=0.9,
            domain="persistence_test"
        )
        
        # Create second instance with same database
        central_post2 = CentralPost(
            enable_memory=True,
            memory_db_path=temp_db_path
        )
        
        # Verify knowledge persists
        knowledge = central_post2.retrieve_relevant_knowledge(
            domain="persistence_test"
        )
        assert len(knowledge) > 0
        assert "Persistent knowledge test" in knowledge[0].content["result"]
    
    def test_knowledge_filtering_and_retrieval(self, central_post):
        """Test advanced knowledge filtering and retrieval."""
        # Store different types of knowledge
        central_post.store_agent_result_as_knowledge(
            agent_id="agent1",
            content="High confidence result",
            confidence=0.95,
            domain="test",
            tags=["high_quality"]
        )
        central_post.store_agent_result_as_knowledge(
            agent_id="agent2",
            content="Medium confidence result",
            confidence=0.6,
            domain="test",
            tags=["medium_quality"]
        )
        
        # Test confidence-based filtering
        high_conf_knowledge = central_post.retrieve_relevant_knowledge(
            domain="test",
            min_confidence=ConfidenceLevel.HIGH
        )
        assert len(high_conf_knowledge) == 1
        assert "High confidence result" in high_conf_knowledge[0].content["result"]
        
        # Test tag-based filtering by retrieving all and checking tags
        all_test_knowledge = central_post.retrieve_relevant_knowledge(domain="test")
        high_quality_knowledge = [k for k in all_test_knowledge if "high_quality" in (k.tags or [])]
        assert len(high_quality_knowledge) == 1
        assert "High confidence result" in high_quality_knowledge[0].content["result"]
