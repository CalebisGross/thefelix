"""
Unit tests for Knowledge Store System.

Tests the KnowledgeStore, KnowledgeEntry, KnowledgeQuery classes and all 
persistence, retrieval, and management functionality.
"""

import pytest
import json
import time
import sqlite3
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
from src.memory.knowledge_store import (
    KnowledgeStore, KnowledgeEntry, KnowledgeQuery,
    KnowledgeType, ConfidenceLevel
)


class TestKnowledgeEntry:
    """Test KnowledgeEntry data class and serialization."""
    
    def test_knowledge_entry_creation(self):
        """Test basic KnowledgeEntry creation."""
        content = {"task": "blog_writing", "result": "success"}
        entry = KnowledgeEntry(
            knowledge_id="test_id_123",
            knowledge_type=KnowledgeType.TASK_RESULT,
            content=content,
            confidence_level=ConfidenceLevel.HIGH,
            source_agent="test_agent",
            domain="writing",
            tags=["blog", "success"]
        )
        
        assert entry.knowledge_id == "test_id_123"
        assert entry.knowledge_type == KnowledgeType.TASK_RESULT
        assert entry.content == content
        assert entry.confidence_level == ConfidenceLevel.HIGH
        assert entry.source_agent == "test_agent"
        assert entry.domain == "writing"
        assert entry.tags == ["blog", "success"]
        assert entry.access_count == 0
        assert entry.success_rate == 1.0
        assert isinstance(entry.created_at, float)
        assert isinstance(entry.updated_at, float)
    
    def test_knowledge_entry_defaults(self):
        """Test KnowledgeEntry with default values."""
        entry = KnowledgeEntry(
            knowledge_id="test_123",
            knowledge_type=KnowledgeType.AGENT_INSIGHT,
            content={"insight": "useful pattern"},
            confidence_level=ConfidenceLevel.MEDIUM,
            source_agent="analyzer",
            domain="analysis"
        )
        
        assert entry.tags == []
        assert entry.access_count == 0
        assert entry.success_rate == 1.0
        assert entry.related_entries == []
        assert entry.created_at > 0
    
    def test_to_dict_serialization(self):
        """Test KnowledgeEntry serialization to dictionary."""
        entry = KnowledgeEntry(
            knowledge_id="serialize_test",
            knowledge_type=KnowledgeType.OPTIMIZATION_DATA,
            content={"metric": "efficiency", "value": 0.85},
            confidence_level=ConfidenceLevel.VERIFIED,
            source_agent="optimizer",
            domain="performance",
            tags=["optimization", "metrics"],
            access_count=5,
            success_rate=0.9,
            related_entries=["related_123"]
        )
        
        entry_dict = entry.to_dict()
        
        assert entry_dict["knowledge_id"] == "serialize_test"
        assert entry_dict["knowledge_type"] == "optimization_data"
        assert entry_dict["confidence_level"] == "verified"
        assert entry_dict["content"]["metric"] == "efficiency"
        assert entry_dict["source_agent"] == "optimizer"
        assert entry_dict["domain"] == "performance"
        assert entry_dict["tags"] == ["optimization", "metrics"]
        assert entry_dict["access_count"] == 5
        assert entry_dict["success_rate"] == 0.9
        assert entry_dict["related_entries"] == ["related_123"]
    
    def test_from_dict_deserialization(self):
        """Test KnowledgeEntry deserialization from dictionary."""
        data = {
            "knowledge_id": "deserialize_test",
            "knowledge_type": "pattern_recognition",
            "content": {"pattern": "helix_convergence", "accuracy": 0.92},
            "confidence_level": "high",
            "source_agent": "pattern_detector",
            "domain": "geometry",
            "tags": ["patterns", "helix"],
            "created_at": 1640995200.0,
            "updated_at": 1640995300.0,
            "access_count": 3,
            "success_rate": 0.85,
            "related_entries": ["pattern_456"]
        }
        
        entry = KnowledgeEntry.from_dict(data)
        
        assert entry.knowledge_id == "deserialize_test"
        assert entry.knowledge_type == KnowledgeType.PATTERN_RECOGNITION
        assert entry.confidence_level == ConfidenceLevel.HIGH
        assert entry.content["pattern"] == "helix_convergence"
        assert entry.source_agent == "pattern_detector"
        assert entry.domain == "geometry"
        assert entry.tags == ["patterns", "helix"]
        assert entry.created_at == 1640995200.0
        assert entry.access_count == 3
        assert entry.success_rate == 0.85


class TestKnowledgeQuery:
    """Test KnowledgeQuery configuration."""
    
    def test_knowledge_query_defaults(self):
        """Test KnowledgeQuery with default values."""
        query = KnowledgeQuery()
        
        assert query.knowledge_types is None
        assert query.domains is None
        assert query.tags is None
        assert query.min_confidence is None
        assert query.min_success_rate is None
        assert query.content_keywords is None
        assert query.time_range is None
        assert query.limit == 10
    
    def test_knowledge_query_specific_filters(self):
        """Test KnowledgeQuery with specific filters."""
        query = KnowledgeQuery(
            knowledge_types=[KnowledgeType.TASK_RESULT, KnowledgeType.AGENT_INSIGHT],
            domains=["writing", "analysis"],
            tags=["blog", "research"],
            min_confidence=ConfidenceLevel.MEDIUM,
            min_success_rate=0.7,
            content_keywords=["optimization", "efficiency"],
            time_range=(1640995200.0, 1640995800.0),
            limit=25
        )
        
        assert len(query.knowledge_types) == 2
        assert KnowledgeType.TASK_RESULT in query.knowledge_types
        assert query.domains == ["writing", "analysis"]
        assert query.tags == ["blog", "research"]
        assert query.min_confidence == ConfidenceLevel.MEDIUM
        assert query.min_success_rate == 0.7
        assert query.content_keywords == ["optimization", "efficiency"]
        assert query.time_range == (1640995200.0, 1640995800.0)
        assert query.limit == 25


class TestKnowledgeStore:
    """Test KnowledgeStore functionality."""
    
    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database file for testing."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as temp_file:
            temp_path = temp_file.name
        yield temp_path
        # Cleanup after test
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    
    @pytest.fixture
    def knowledge_store(self, temp_db_path):
        """Create KnowledgeStore instance for testing."""
        return KnowledgeStore(storage_path=temp_db_path, enable_compression=False)
    
    @pytest.fixture
    def compressed_store(self, temp_db_path):
        """Create KnowledgeStore with compression enabled."""
        return KnowledgeStore(storage_path=temp_db_path, enable_compression=True)
    
    def test_knowledge_store_initialization(self, temp_db_path):
        """Test KnowledgeStore initialization and database creation."""
        store = KnowledgeStore(storage_path=temp_db_path)
        
        assert store.storage_path == Path(temp_db_path)
        assert store.enable_compression is True  # Default
        assert os.path.exists(temp_db_path)
        
        # Verify database schema
        with sqlite3.connect(temp_db_path) as conn:
            cursor = conn.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='knowledge_entries'
            """)
            assert cursor.fetchone() is not None
    
    def test_database_initialization_with_indexes(self, temp_db_path):
        """Test that database indexes are created correctly."""
        store = KnowledgeStore(storage_path=temp_db_path)
        
        with sqlite3.connect(temp_db_path) as conn:
            cursor = conn.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='index' AND tbl_name='knowledge_entries'
            """)
            indexes = [row[0] for row in cursor.fetchall()]
            
            # Check that required indexes exist
            expected_indexes = [
                "idx_knowledge_type", "idx_domain", 
                "idx_confidence", "idx_created_at"
            ]
            for index in expected_indexes:
                assert index in indexes
    
    def test_generate_knowledge_id(self, knowledge_store):
        """Test knowledge ID generation."""
        content1 = {"task": "writing", "result": "success"}
        content2 = {"task": "analysis", "result": "complete"}
        
        # Same content and agent should generate different IDs (due to timestamp)
        with patch('src.memory.knowledge_store.time.time', return_value=1640995200.0):
            id1 = knowledge_store._generate_knowledge_id(content1, "agent1")
        with patch('src.memory.knowledge_store.time.time', return_value=1640995300.0):
            id2 = knowledge_store._generate_knowledge_id(content1, "agent1")
        
        assert id1 != id2
        assert len(id1) == 16  # SHA256 truncated to 16 chars
        assert len(id2) == 16
        
        # Different content should generate different IDs
        id3 = knowledge_store._generate_knowledge_id(content2, "agent1")
        assert id3 != id1
    
    def test_compress_decompress_content(self, knowledge_store):
        """Test content compression and decompression."""
        content = {
            "large_data": "x" * 10000,  # Much larger content for effective compression
            "repeated_pattern": "Hello World! " * 1000,  # Repetitive data compresses well
            "nested": {"data": [1, 2, 3, 4, 5]},
            "metadata": {"timestamp": 1640995200.0}
        }
        
        compressed = knowledge_store._compress_content(content)
        assert isinstance(compressed, bytes)
        assert len(compressed) > 0  # Should produce compressed data
        
        decompressed = knowledge_store._decompress_content(compressed)
        assert decompressed == content  # Should decompress to original
    
    def test_store_knowledge_basic(self, knowledge_store):
        """Test basic knowledge storage."""
        content = {"task": "blog_writing", "outcome": "successful"}
        
        knowledge_id = knowledge_store.store_knowledge(
            knowledge_type=KnowledgeType.TASK_RESULT,
            content=content,
            confidence_level=ConfidenceLevel.HIGH,
            source_agent="writer_agent",
            domain="writing",
            tags=["blog", "success"]
        )
        
        assert isinstance(knowledge_id, str)
        assert len(knowledge_id) == 16
        
        # Verify data is stored in database
        with sqlite3.connect(knowledge_store.storage_path) as conn:
            cursor = conn.execute("""
                SELECT knowledge_id, knowledge_type, content_json, source_agent, domain 
                FROM knowledge_entries WHERE knowledge_id = ?
            """, (knowledge_id,))
            row = cursor.fetchone()
            
            assert row is not None
            assert row[0] == knowledge_id
            assert row[1] == "task_result"
            assert json.loads(row[2]) == content
            assert row[3] == "writer_agent"
            assert row[4] == "writing"
    
    def test_store_knowledge_with_compression(self, compressed_store):
        """Test knowledge storage with compression for large content."""
        # Create large content that should trigger compression
        large_content = {
            "description": "Large content " + "x" * 2000,
            "data": list(range(100)),
            "metadata": {"type": "performance_test"}
        }
        
        knowledge_id = compressed_store.store_knowledge(
            knowledge_type=KnowledgeType.OPTIMIZATION_DATA,
            content=large_content,
            confidence_level=ConfidenceLevel.MEDIUM,
            source_agent="performance_agent",
            domain="optimization"
        )
        
        # Verify compressed storage
        with sqlite3.connect(compressed_store.storage_path) as conn:
            cursor = conn.execute("""
                SELECT content_json, content_compressed 
                FROM knowledge_entries WHERE knowledge_id = ?
            """, (knowledge_id,))
            row = cursor.fetchone()
            
            assert row[0] == ""  # JSON should be empty (compressed)
            assert row[1] is not None  # Compressed data should exist
            assert isinstance(row[1], bytes)
    
    def test_store_knowledge_without_tags(self, knowledge_store):
        """Test storing knowledge without tags."""
        content = {"insight": "geometric convergence improves efficiency"}
        
        knowledge_id = knowledge_store.store_knowledge(
            knowledge_type=KnowledgeType.AGENT_INSIGHT,
            content=content,
            confidence_level=ConfidenceLevel.VERIFIED,
            source_agent="geometry_agent",
            domain="mathematics"
            # No tags parameter
        )
        
        # Verify empty tags are stored
        with sqlite3.connect(knowledge_store.storage_path) as conn:
            cursor = conn.execute("""
                SELECT tags_json FROM knowledge_entries WHERE knowledge_id = ?
            """, (knowledge_id,))
            tags_json = cursor.fetchone()[0]
            assert json.loads(tags_json) == []
    
    def test_retrieve_knowledge_basic(self, knowledge_store):
        """Test basic knowledge retrieval."""
        # Store some test knowledge
        content1 = {"task": "writing", "result": "success"}
        id1 = knowledge_store.store_knowledge(
            KnowledgeType.TASK_RESULT, content1, ConfidenceLevel.HIGH,
            "writer", "writing", ["blog"]
        )
        
        content2 = {"pattern": "helix_convergence", "accuracy": 0.9}
        id2 = knowledge_store.store_knowledge(
            KnowledgeType.PATTERN_RECOGNITION, content2, ConfidenceLevel.MEDIUM,
            "analyzer", "geometry", ["patterns"]
        )
        
        # Retrieve all knowledge
        query = KnowledgeQuery(limit=10)
        results = knowledge_store.retrieve_knowledge(query)
        
        assert len(results) == 2
        assert all(isinstance(entry, KnowledgeEntry) for entry in results)
        
        # Results should be ordered by confidence, success rate, updated_at DESC
        assert results[0].confidence_level == ConfidenceLevel.HIGH  # Higher confidence first
    
    def test_retrieve_knowledge_by_type(self, knowledge_store):
        """Test knowledge retrieval filtered by type."""
        # Store different types of knowledge
        knowledge_store.store_knowledge(
            KnowledgeType.TASK_RESULT, {"result": "success"}, ConfidenceLevel.HIGH,
            "agent1", "domain1"
        )
        knowledge_store.store_knowledge(
            KnowledgeType.AGENT_INSIGHT, {"insight": "pattern"}, ConfidenceLevel.MEDIUM,
            "agent2", "domain2"
        )
        knowledge_store.store_knowledge(
            KnowledgeType.OPTIMIZATION_DATA, {"metric": 0.8}, ConfidenceLevel.LOW,
            "agent3", "domain3"
        )
        
        # Query for specific types
        query = KnowledgeQuery(
            knowledge_types=[KnowledgeType.TASK_RESULT, KnowledgeType.AGENT_INSIGHT]
        )
        results = knowledge_store.retrieve_knowledge(query)
        
        assert len(results) == 2
        result_types = {entry.knowledge_type for entry in results}
        assert result_types == {KnowledgeType.TASK_RESULT, KnowledgeType.AGENT_INSIGHT}
    
    def test_retrieve_knowledge_by_domain(self, knowledge_store):
        """Test knowledge retrieval filtered by domain."""
        # Store knowledge in different domains
        knowledge_store.store_knowledge(
            KnowledgeType.TASK_RESULT, {"result": "success"}, ConfidenceLevel.HIGH,
            "agent1", "writing"
        )
        knowledge_store.store_knowledge(
            KnowledgeType.AGENT_INSIGHT, {"insight": "pattern"}, ConfidenceLevel.MEDIUM,
            "agent2", "analysis"
        )
        knowledge_store.store_knowledge(
            KnowledgeType.OPTIMIZATION_DATA, {"metric": 0.8}, ConfidenceLevel.HIGH,
            "agent3", "geometry"
        )
        
        # Query for specific domains
        query = KnowledgeQuery(domains=["writing", "geometry"])
        results = knowledge_store.retrieve_knowledge(query)
        
        assert len(results) == 2
        result_domains = {entry.domain for entry in results}
        assert result_domains == {"writing", "geometry"}
    
    def test_retrieve_knowledge_by_confidence(self, knowledge_store):
        """Test knowledge retrieval filtered by confidence level."""
        # Store knowledge with different confidence levels
        knowledge_store.store_knowledge(
            KnowledgeType.TASK_RESULT, {"result": "low"}, ConfidenceLevel.LOW,
            "agent1", "domain1"
        )
        knowledge_store.store_knowledge(
            KnowledgeType.TASK_RESULT, {"result": "medium"}, ConfidenceLevel.MEDIUM,
            "agent2", "domain2"
        )
        knowledge_store.store_knowledge(
            KnowledgeType.TASK_RESULT, {"result": "high"}, ConfidenceLevel.HIGH,
            "agent3", "domain3"
        )
        knowledge_store.store_knowledge(
            KnowledgeType.TASK_RESULT, {"result": "verified"}, ConfidenceLevel.VERIFIED,
            "agent4", "domain4"
        )
        
        # Query for medium confidence and above
        query = KnowledgeQuery(min_confidence=ConfidenceLevel.MEDIUM)
        results = knowledge_store.retrieve_knowledge(query)
        
        assert len(results) == 3  # MEDIUM, HIGH, VERIFIED
        confidence_levels = {entry.confidence_level for entry in results}
        assert ConfidenceLevel.LOW not in confidence_levels
        assert all(level in [ConfidenceLevel.MEDIUM, ConfidenceLevel.HIGH, ConfidenceLevel.VERIFIED] 
                  for level in confidence_levels)
    
    def test_retrieve_knowledge_by_success_rate(self, knowledge_store):
        """Test knowledge retrieval filtered by success rate."""
        # Store knowledge and update success rates
        id1 = knowledge_store.store_knowledge(
            KnowledgeType.TASK_RESULT, {"result": "poor"}, ConfidenceLevel.HIGH,
            "agent1", "domain1"
        )
        id2 = knowledge_store.store_knowledge(
            KnowledgeType.TASK_RESULT, {"result": "good"}, ConfidenceLevel.HIGH,
            "agent2", "domain2"
        )
        
        # Update success rates
        knowledge_store.update_success_rate(id1, 0.3)  # Low success
        knowledge_store.update_success_rate(id2, 0.8)  # High success
        
        # Query for high success rate only
        query = KnowledgeQuery(min_success_rate=0.7)
        results = knowledge_store.retrieve_knowledge(query)
        
        assert len(results) == 1
        assert results[0].success_rate >= 0.7
        assert results[0].content["result"] == "good"
    
    def test_retrieve_knowledge_by_time_range(self, knowledge_store):
        """Test knowledge retrieval filtered by time range."""
        # Store knowledge at different times
        with patch('src.memory.knowledge_store.time.time', return_value=1640995200.0):  # Jan 1, 2022
            id1 = knowledge_store.store_knowledge(
                KnowledgeType.TASK_RESULT, {"result": "old"}, ConfidenceLevel.HIGH,
                "agent1", "domain1"
            )
        
        with patch('src.memory.knowledge_store.time.time', return_value=1672531200.0):  # Jan 1, 2023
            id2 = knowledge_store.store_knowledge(
                KnowledgeType.TASK_RESULT, {"result": "new"}, ConfidenceLevel.HIGH,
                "agent2", "domain2"
            )
        
        # Query for entries from 2023 only
        query = KnowledgeQuery(time_range=(1672531200.0 - 1, 1672531200.0 + 1))
        results = knowledge_store.retrieve_knowledge(query)
        
        assert len(results) == 1
        assert results[0].content["result"] == "new"
    
    def test_retrieve_knowledge_by_content_keywords(self, knowledge_store):
        """Test knowledge retrieval filtered by content keywords."""
        # Store knowledge with different content
        knowledge_store.store_knowledge(
            KnowledgeType.TASK_RESULT, 
            {"description": "helix geometry optimization", "metric": 0.9}, 
            ConfidenceLevel.HIGH, "agent1", "geometry"
        )
        knowledge_store.store_knowledge(
            KnowledgeType.AGENT_INSIGHT, 
            {"description": "blog writing improvements", "efficiency": 0.8}, 
            ConfidenceLevel.MEDIUM, "agent2", "writing"
        )
        
        # Query for content containing "helix"
        query = KnowledgeQuery(content_keywords=["helix"])
        results = knowledge_store.retrieve_knowledge(query)
        
        assert len(results) == 1
        assert "helix" in results[0].content["description"]
    
    def test_retrieve_knowledge_by_tags(self, knowledge_store):
        """Test knowledge retrieval filtered by tags."""
        # Store knowledge with different tags
        knowledge_store.store_knowledge(
            KnowledgeType.TASK_RESULT, {"result": "success"}, ConfidenceLevel.HIGH,
            "agent1", "domain1", ["blog", "writing", "success"]
        )
        knowledge_store.store_knowledge(
            KnowledgeType.AGENT_INSIGHT, {"insight": "pattern"}, ConfidenceLevel.MEDIUM,
            "agent2", "domain2", ["analysis", "patterns", "geometry"]
        )
        
        # Query for entries with "blog" tag
        query = KnowledgeQuery(tags=["blog"])
        results = knowledge_store.retrieve_knowledge(query)
        
        assert len(results) == 1
        assert "blog" in results[0].tags
    
    def test_retrieve_knowledge_with_limit(self, knowledge_store):
        """Test knowledge retrieval with result limit."""
        # Store multiple knowledge entries
        for i in range(5):
            knowledge_store.store_knowledge(
                KnowledgeType.TASK_RESULT, {"index": i}, ConfidenceLevel.HIGH,
                f"agent{i}", "domain"
            )
        
        # Query with limit
        query = KnowledgeQuery(limit=3)
        results = knowledge_store.retrieve_knowledge(query)
        
        assert len(results) == 3
    
    def test_retrieve_knowledge_access_count_update(self, knowledge_store):
        """Test that access count is incremented during retrieval."""
        # Store knowledge
        knowledge_id = knowledge_store.store_knowledge(
            KnowledgeType.TASK_RESULT, {"result": "test"}, ConfidenceLevel.HIGH,
            "agent", "domain"
        )
        
        # Initial access count should be 0
        with sqlite3.connect(knowledge_store.storage_path) as conn:
            cursor = conn.execute("""
                SELECT access_count FROM knowledge_entries WHERE knowledge_id = ?
            """, (knowledge_id,))
            assert cursor.fetchone()[0] == 0
        
        # Retrieve knowledge (should increment access count)
        query = KnowledgeQuery()
        results = knowledge_store.retrieve_knowledge(query)
        
        # Access count should be incremented
        with sqlite3.connect(knowledge_store.storage_path) as conn:
            cursor = conn.execute("""
                SELECT access_count FROM knowledge_entries WHERE knowledge_id = ?
            """, (knowledge_id,))
            assert cursor.fetchone()[0] == 1
    
    def test_update_success_rate(self, knowledge_store):
        """Test updating success rate for knowledge entry."""
        # Store knowledge
        knowledge_id = knowledge_store.store_knowledge(
            KnowledgeType.OPTIMIZATION_DATA, {"metric": "efficiency"}, 
            ConfidenceLevel.HIGH, "agent", "performance"
        )
        
        # Update success rate
        result = knowledge_store.update_success_rate(knowledge_id, 0.75)
        assert result is True
        
        # Verify update in database
        with sqlite3.connect(knowledge_store.storage_path) as conn:
            cursor = conn.execute("""
                SELECT success_rate, updated_at FROM knowledge_entries 
                WHERE knowledge_id = ?
            """, (knowledge_id,))
            row = cursor.fetchone()
            
            assert row[0] == 0.75
            assert row[1] > time.time() - 5  # Recently updated
    
    def test_update_success_rate_nonexistent(self, knowledge_store):
        """Test updating success rate for non-existent entry."""
        result = knowledge_store.update_success_rate("nonexistent_id", 0.5)
        assert result is False
    
    def test_add_related_entry(self, knowledge_store):
        """Test adding relationships between knowledge entries."""
        # Store two knowledge entries
        id1 = knowledge_store.store_knowledge(
            KnowledgeType.TASK_RESULT, {"result": "primary"}, 
            ConfidenceLevel.HIGH, "agent1", "domain1"
        )
        id2 = knowledge_store.store_knowledge(
            KnowledgeType.AGENT_INSIGHT, {"insight": "related"}, 
            ConfidenceLevel.MEDIUM, "agent2", "domain2"
        )
        
        # Add relationship
        result = knowledge_store.add_related_entry(id1, id2)
        assert result is True
        
        # Verify relationship in database
        with sqlite3.connect(knowledge_store.storage_path) as conn:
            cursor = conn.execute("""
                SELECT related_entries_json FROM knowledge_entries 
                WHERE knowledge_id = ?
            """, (id1,))
            related_json = cursor.fetchone()[0]
            related_entries = json.loads(related_json)
            
            assert id2 in related_entries
    
    def test_add_related_entry_duplicate(self, knowledge_store):
        """Test adding duplicate relationship (should not duplicate)."""
        # Store knowledge entries and add relationship twice
        id1 = knowledge_store.store_knowledge(
            KnowledgeType.TASK_RESULT, {"result": "primary"}, 
            ConfidenceLevel.HIGH, "agent1", "domain1"
        )
        id2 = knowledge_store.store_knowledge(
            KnowledgeType.AGENT_INSIGHT, {"insight": "related"}, 
            ConfidenceLevel.MEDIUM, "agent2", "domain2"
        )
        
        # Add relationship twice
        knowledge_store.add_related_entry(id1, id2)
        knowledge_store.add_related_entry(id1, id2)
        
        # Verify only one relationship exists
        with sqlite3.connect(knowledge_store.storage_path) as conn:
            cursor = conn.execute("""
                SELECT related_entries_json FROM knowledge_entries 
                WHERE knowledge_id = ?
            """, (id1,))
            related_json = cursor.fetchone()[0]
            related_entries = json.loads(related_json)
            
            assert related_entries.count(id2) == 1
    
    def test_add_related_entry_nonexistent(self, knowledge_store):
        """Test adding relationship to non-existent entry."""
        result = knowledge_store.add_related_entry("nonexistent_id", "some_id")
        assert result is False
    
    def test_get_knowledge_summary_empty(self, knowledge_store):
        """Test knowledge summary for empty store."""
        summary = knowledge_store.get_knowledge_summary()
        
        assert summary["total_entries"] == 0
        assert summary["by_type"] == {}
        assert summary["by_domain"] == {}
        assert summary["by_confidence"] == {}
        assert summary["average_success_rate"] == 0.0
        assert "storage_path" in summary
    
    def test_get_knowledge_summary_populated(self, knowledge_store):
        """Test knowledge summary for populated store."""
        # Store various types of knowledge
        knowledge_store.store_knowledge(
            KnowledgeType.TASK_RESULT, {"result1": "success"}, 
            ConfidenceLevel.HIGH, "agent1", "writing"
        )
        knowledge_store.store_knowledge(
            KnowledgeType.TASK_RESULT, {"result2": "success"}, 
            ConfidenceLevel.MEDIUM, "agent2", "writing"
        )
        knowledge_store.store_knowledge(
            KnowledgeType.AGENT_INSIGHT, {"insight": "pattern"}, 
            ConfidenceLevel.HIGH, "agent3", "analysis"
        )
        
        summary = knowledge_store.get_knowledge_summary()
        
        assert summary["total_entries"] == 3
        assert summary["by_type"]["task_result"] == 2
        assert summary["by_type"]["agent_insight"] == 1
        assert summary["by_domain"]["writing"] == 2
        assert summary["by_domain"]["analysis"] == 1
        assert summary["by_confidence"]["high"] == 2
        assert summary["by_confidence"]["medium"] == 1
        assert summary["average_success_rate"] == 1.0  # All default to 1.0
    
    def test_cleanup_old_entries(self, knowledge_store):
        """Test cleanup of old and low-performing entries."""
        current_time = time.time()
        old_time = current_time - (31 * 24 * 3600)  # 31 days ago
        recent_time = current_time - (10 * 24 * 3600)  # 10 days ago
        
        # Store entries with different ages and success rates
        with patch('src.memory.knowledge_store.time.time', return_value=old_time):
            old_bad_id = knowledge_store.store_knowledge(
                KnowledgeType.TASK_RESULT, {"result": "old_bad"}, 
                ConfidenceLevel.LOW, "agent1", "domain1"
            )
            old_good_id = knowledge_store.store_knowledge(
                KnowledgeType.TASK_RESULT, {"result": "old_good"}, 
                ConfidenceLevel.HIGH, "agent2", "domain2"
            )
        
        with patch('src.memory.knowledge_store.time.time', return_value=recent_time):
            recent_id = knowledge_store.store_knowledge(
                KnowledgeType.TASK_RESULT, {"result": "recent"}, 
                ConfidenceLevel.MEDIUM, "agent3", "domain3"
            )
        
        # Update success rates
        knowledge_store.update_success_rate(old_bad_id, 0.2)  # Low success
        knowledge_store.update_success_rate(old_good_id, 0.8)  # High success
        
        # Access the good entry to increment access_count
        query = KnowledgeQuery()
        good_entries = knowledge_store.retrieve_knowledge(query)
        accessed_entry = None
        for entry in good_entries:
            if entry.knowledge_id == old_good_id:
                accessed_entry = entry
                break
        assert accessed_entry is not None  # Ensure we accessed it
        
        # Run cleanup (max_age_days=30, min_success_rate=0.3)
        deleted_count = knowledge_store.cleanup_old_entries(
            max_age_days=30, min_success_rate=0.3
        )
        
        # Should delete old entry with low success rate
        assert deleted_count == 1
        
        # Verify remaining entries
        query = KnowledgeQuery()
        remaining = knowledge_store.retrieve_knowledge(query)
        remaining_results = {entry.content.get("result") for entry in remaining}
        
        assert "old_bad" not in remaining_results  # Should be deleted
        assert "old_good" in remaining_results  # Should remain (good success rate)
        assert "recent" in remaining_results  # Should remain (recent)
    
    def test_cleanup_unused_entries(self, knowledge_store):
        """Test cleanup of unused entries (zero access count)."""
        current_time = time.time()
        old_time = current_time - (31 * 24 * 3600)  # 31 days ago
        
        # Store old entries 
        with patch('src.memory.knowledge_store.time.time', return_value=old_time):
            unused_id = knowledge_store.store_knowledge(
                KnowledgeType.TASK_RESULT, {"result": "unused"}, 
                ConfidenceLevel.HIGH, "agent1", "domain1"
            )
            accessed_id = knowledge_store.store_knowledge(
                KnowledgeType.TASK_RESULT, {"result": "accessed"}, 
                ConfidenceLevel.HIGH, "agent2", "domain2"
            )
        
        # Access only the specific entry we want to mark as accessed
        query = KnowledgeQuery(domains=["domain2"])  # Only get the accessed entry
        knowledge_store.retrieve_knowledge(query)
        
        # Run cleanup - should delete unused old entry
        deleted_count = knowledge_store.cleanup_old_entries(max_age_days=30)
        
        assert deleted_count == 1  # Unused old entry deleted


class TestIntegrationScenarios:
    """Test integration scenarios for knowledge store system."""
    
    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database file for testing."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as temp_file:
            temp_path = temp_file.name
        yield temp_path
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    
    def test_blog_writing_knowledge_lifecycle(self, temp_db_path):
        """Test complete knowledge lifecycle for blog writing scenario."""
        store = KnowledgeStore(storage_path=temp_db_path)
        
        # 1. Store initial task result
        task_result_id = store.store_knowledge(
            knowledge_type=KnowledgeType.TASK_RESULT,
            content={
                "task": "blog_writing",
                "topic": "AI ethics",
                "word_count": 1500,
                "quality_score": 0.85,
                "completion_time": 45.2,
                "agent_count": 5
            },
            confidence_level=ConfidenceLevel.HIGH,
            source_agent="blog_coordinator",
            domain="writing",
            tags=["blog", "AI", "ethics", "successful"]
        )
        
        # 2. Store agent insight about effective collaboration
        insight_id = store.store_knowledge(
            knowledge_type=KnowledgeType.AGENT_INSIGHT,
            content={
                "insight": "helix_convergence_pattern",
                "description": "Research agents spawning early improved topic coverage",
                "effectiveness": 0.92,
                "pattern_data": {
                    "research_agents": 2,
                    "analysis_agents": 2,
                    "synthesis_agents": 1,
                    "spawn_timing": [0.1, 0.3, 0.5, 0.7, 0.9]
                }
            },
            confidence_level=ConfidenceLevel.VERIFIED,
            source_agent="pattern_analyzer",
            domain="coordination",
            tags=["helix", "spawning", "collaboration", "patterns"]
        )
        
        # 3. Store optimization data
        optimization_id = store.store_knowledge(
            knowledge_type=KnowledgeType.OPTIMIZATION_DATA,
            content={
                "optimization_target": "completion_time",
                "baseline": 60.0,
                "optimized": 45.2,
                "improvement": 0.247,
                "technique": "dynamic_spawning",
                "parameters": {
                    "confidence_threshold": 0.7,
                    "max_agents": 5,
                    "spawn_interval": 0.2
                }
            },
            confidence_level=ConfidenceLevel.HIGH,
            source_agent="optimizer",
            domain="performance",
            tags=["optimization", "timing", "dynamic_spawning"]
        )
        
        # 4. Add relationships between entries
        store.add_related_entry(task_result_id, insight_id)
        store.add_related_entry(task_result_id, optimization_id)
        store.add_related_entry(insight_id, optimization_id)
        
        # 5. Query for similar tasks (blog writing)
        blog_query = KnowledgeQuery(
            domains=["writing"],
            tags=["blog"],
            min_confidence=ConfidenceLevel.MEDIUM
        )
        blog_results = store.retrieve_knowledge(blog_query)
        
        assert len(blog_results) == 1
        assert blog_results[0].content["task"] == "blog_writing"
        assert "AI ethics" in blog_results[0].content["topic"]
        
        # 6. Query for optimization insights
        optimization_query = KnowledgeQuery(
            knowledge_types=[KnowledgeType.OPTIMIZATION_DATA],
            content_keywords=["dynamic_spawning"]
        )
        opt_results = store.retrieve_knowledge(optimization_query)
        
        assert len(opt_results) == 1
        assert opt_results[0].content["improvement"] > 0.2
        
        # 7. Query for coordination patterns
        pattern_query = KnowledgeQuery(
            knowledge_types=[KnowledgeType.AGENT_INSIGHT],
            tags=["helix", "patterns"]
        )
        pattern_results = store.retrieve_knowledge(pattern_query)
        
        assert len(pattern_results) == 1
        assert "helix_convergence_pattern" in pattern_results[0].content["insight"]
        
        # 8. Update success rates based on real usage
        store.update_success_rate(task_result_id, 0.9)  # Very successful
        store.update_success_rate(insight_id, 0.95)  # Extremely valuable
        store.update_success_rate(optimization_id, 0.85)  # Good results
        
        # 9. Get knowledge summary
        summary = store.get_knowledge_summary()
        
        assert summary["total_entries"] == 3
        assert summary["by_domain"]["writing"] == 1
        assert summary["by_domain"]["coordination"] == 1
        assert summary["by_domain"]["performance"] == 1
        assert summary["average_success_rate"] > 0.85
    
    def test_cross_domain_knowledge_search(self, temp_db_path):
        """Test searching for knowledge across multiple domains."""
        store = KnowledgeStore(storage_path=temp_db_path)
        
        # Store knowledge across different domains but related concepts
        # Technical domain - helix geometry
        store.store_knowledge(
            KnowledgeType.PATTERN_RECOGNITION,
            {
                "pattern": "geometric_convergence",
                "mathematical_model": "parametric_helix",
                "concentration_ratio": 4119,
                "precision": 1e-12
            },
            ConfidenceLevel.VERIFIED,
            "geometry_agent",
            "mathematics",
            ["helix", "geometry", "convergence", "precision"]
        )
        
        # Performance domain - efficiency gains
        store.store_knowledge(
            KnowledgeType.OPTIMIZATION_DATA,
            {
                "optimization": "task_distribution",
                "architecture": "helix_spoke",
                "efficiency_gain": 0.441,
                "statistical_significance": 0.0441,
                "comparison": "vs_linear_pipeline"
            },
            ConfidenceLevel.HIGH,
            "performance_agent",
            "performance",
            ["helix", "efficiency", "task_distribution", "statistical"]
        )
        
        # Coordination domain - agent spawning
        store.store_knowledge(
            KnowledgeType.AGENT_INSIGHT,
            {
                "coordination_pattern": "temporal_spawning",
                "helix_position": "all_spawn_at_top",
                "attention_focusing": "natural_convergence",
                "spawn_timing": "different_times_same_geometry"
            },
            ConfidenceLevel.HIGH,
            "coordinator_agent",
            "coordination",
            ["helix", "spawning", "attention", "coordination"]
        )
        
        # Search across domains for helix-related knowledge
        helix_query = KnowledgeQuery(
            tags=["helix"],
            min_confidence=ConfidenceLevel.MEDIUM
        )
        helix_results = store.retrieve_knowledge(helix_query)
        
        assert len(helix_results) == 3
        domains = {entry.domain for entry in helix_results}
        assert domains == {"mathematics", "performance", "coordination"}
        
        # Search for high-precision, verified knowledge
        precision_query = KnowledgeQuery(
            content_keywords=["precision", "statistical"],
            min_confidence=ConfidenceLevel.HIGH
        )
        precision_results = store.retrieve_knowledge(precision_query)
        
        assert len(precision_results) == 2  # Mathematics and performance entries
        
        # Search for coordination-specific insights
        coordination_query = KnowledgeQuery(
            domains=["coordination"],
            knowledge_types=[KnowledgeType.AGENT_INSIGHT]
        )
        coord_results = store.retrieve_knowledge(coordination_query)
        
        assert len(coord_results) == 1
        assert "temporal_spawning" in coord_results[0].content["coordination_pattern"]
    
    def test_knowledge_evolution_over_time(self, temp_db_path):
        """Test how knowledge evolves and gets refined over time."""
        store = KnowledgeStore(storage_path=temp_db_path)
        
        # Initial hypothesis with low confidence
        hypothesis_time = time.time() - (10 * 24 * 3600)  # 10 days ago
        with patch('src.memory.knowledge_store.time.time', return_value=hypothesis_time):
            hypothesis_id = store.store_knowledge(
                KnowledgeType.DOMAIN_EXPERTISE,
                {
                    "hypothesis": "helix_better_than_mesh",
                    "initial_evidence": "theoretical_analysis",
                    "confidence_factors": ["geometric_elegance", "o_n_complexity"],
                    "uncertainty": "lacks_empirical_validation"
                },
                ConfidenceLevel.LOW,
                "theorist_agent",
                "research",
                ["hypothesis", "helix", "mesh", "theory"]
            )
        
        # Experimental results with medium confidence
        experiment_time = time.time() - (5 * 24 * 3600)  # 5 days ago
        with patch('src.memory.knowledge_store.time.time', return_value=experiment_time):
            experiment_id = store.store_knowledge(
                KnowledgeType.OPTIMIZATION_DATA,
                {
                    "experiment": "helix_vs_mesh_performance",
                    "metrics": {
                        "task_distribution_efficiency": 0.441,
                        "memory_efficiency": 0.75,
                        "communication_overhead": "inconclusive"
                    },
                    "statistical_significance": 0.0441,
                    "sample_size": 100
                },
                ConfidenceLevel.MEDIUM,
                "experimenter_agent",
                "validation",
                ["experiment", "helix", "mesh", "performance", "statistical"]
            )
        
        # Verified conclusion with high confidence
        conclusion_time = time.time() - (1 * 24 * 3600)  # 1 day ago
        with patch('src.memory.knowledge_store.time.time', return_value=conclusion_time):
            conclusion_id = store.store_knowledge(
                KnowledgeType.DOMAIN_EXPERTISE,
                {
                    "conclusion": "helix_advantages_validated",
                    "validated_benefits": [
                        "task_distribution_efficiency",
                        "memory_efficiency",
                        "natural_attention_focusing"
                    ],
                    "evidence_base": ["theoretical", "experimental", "statistical"],
                    "publication_ready": True,
                    "confidence_score": 0.95
                },
                ConfidenceLevel.VERIFIED,
                "validator_agent",
                "research",
                ["conclusion", "helix", "validated", "publication"]
            )
        
        # Link related knowledge
        store.add_related_entry(hypothesis_id, experiment_id)
        store.add_related_entry(experiment_id, conclusion_id)
        store.add_related_entry(hypothesis_id, conclusion_id)
        
        # Update success rates based on validation
        store.update_success_rate(hypothesis_id, 0.8)  # Good starting point
        store.update_success_rate(experiment_id, 0.9)  # Solid experimental work
        store.update_success_rate(conclusion_id, 0.95)  # Validated conclusion
        
        # Query for evolution of helix research
        research_evolution = KnowledgeQuery(
            domains=["research", "validation"],
            tags=["helix"],
            time_range=(hypothesis_time - 3600, conclusion_time + 3600)  # Full range
        )
        evolution_results = store.retrieve_knowledge(research_evolution)
        
        assert len(evolution_results) == 3
        
        # Results should be ordered by confidence level (desc)
        confidence_levels = [entry.confidence_level for entry in evolution_results]
        assert confidence_levels[0] == ConfidenceLevel.VERIFIED  # Most confident first
        
        # Query for high-confidence, validated knowledge only
        validated_query = KnowledgeQuery(
            min_confidence=ConfidenceLevel.HIGH,
            content_keywords=["validated", "publication"]
        )
        validated_results = store.retrieve_knowledge(validated_query)
        
        assert len(validated_results) == 1
        assert validated_results[0].content["publication_ready"] is True
        assert validated_results[0].confidence_level == ConfidenceLevel.VERIFIED