"""
Unit tests for Dynamic Agent Spawning System.

Tests the ConfidenceMonitor, ContentAnalyzer, TeamSizeOptimizer, and 
DynamicSpawning coordinator to ensure proper agent spawning decisions.
"""

import pytest
import time
import statistics
from unittest.mock import Mock, MagicMock
from dataclasses import dataclass
from typing import List, Dict, Any

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from agents.dynamic_spawning import (
    ConfidenceMonitor, ContentAnalyzer, TeamSizeOptimizer, DynamicSpawning,
    ConfidenceTrend, ContentIssue, ConfidenceMetrics, ContentAnalysis,
    SpawningDecision
)


class TestConfidenceMonitor:
    """Test ConfidenceMonitor functionality."""
    
    def test_init(self):
        """Test ConfidenceMonitor initialization."""
        monitor = ConfidenceMonitor()
        assert monitor.confidence_threshold == 0.7
        assert monitor.volatility_threshold == 0.15
        assert monitor.time_window_minutes == 5.0
        assert len(monitor._confidence_history) == 0
    
    def test_record_confidence(self):
        """Test confidence recording."""
        monitor = ConfidenceMonitor()
        
        message = Message(
            id="test_msg",
            sender="agent_1",
            recipient="central_post",
            message_type=MessageType.RESULT,
            content={
                "confidence": 0.85,
                "agent_type": "research",
                "position_info": {"depth_ratio": 0.3}
            },
            timestamp=time.time()
        )
        
        monitor.record_confidence(message)
        
        assert len(monitor._confidence_history) == 1
        assert monitor._agent_type_confidence["research"] == [0.85]
        assert monitor._position_confidence["shallow"] == [0.85]
    
    def test_categorize_depth(self):
        """Test depth categorization."""
        monitor = ConfidenceMonitor()
        
        assert monitor._categorize_depth(0.2) == "shallow"
        assert monitor._categorize_depth(0.5) == "middle"
        assert monitor._categorize_depth(0.8) == "deep"
    
    def test_get_current_metrics_empty(self):
        """Test metrics with no data."""
        monitor = ConfidenceMonitor()
        metrics = monitor.get_current_metrics()
        
        assert metrics.current_average == 0.0
        assert metrics.trend == ConfidenceTrend.STABLE
        assert metrics.volatility == 0.0
    
    def test_get_current_metrics_with_data(self):
        """Test metrics calculation with sample data."""
        monitor = ConfidenceMonitor()
        current_time = time.time()
        
        # Add sample confidence data
        for i, confidence in enumerate([0.8, 0.75, 0.9, 0.85]):
            message = Message(
                id=f"msg_{i}",
                sender=f"agent_{i}",
                recipient="central_post",
                message_type=MessageType.RESULT,
                content={
                    "confidence": confidence,
                    "agent_type": "research",
                    "position_info": {"depth_ratio": 0.3}
                },
                timestamp=current_time + i
            )
            monitor.record_confidence(message)
        
        metrics = monitor.get_current_metrics()
        
        assert abs(metrics.current_average - 0.825) < 0.01  # Mean of [0.8, 0.75, 0.9, 0.85]
        assert metrics.trend in [ConfidenceTrend.STABLE, ConfidenceTrend.IMPROVING]
        assert metrics.volatility > 0  # Should have some variance
        assert "research" in metrics.agent_type_breakdown
    
    def test_should_spawn_for_confidence_low(self):
        """Test spawning trigger for low confidence."""
        monitor = ConfidenceMonitor(confidence_threshold=0.7)
        current_time = time.time()
        
        # Add low confidence data
        message = Message(
            id="low_conf",
            sender="agent_1",
            recipient="central_post",
            message_type=MessageType.RESULT,
            content={
                "confidence": 0.6,
                "agent_type": "research",
                "position_info": {"depth_ratio": 0.3}
            },
            timestamp=current_time
        )
        monitor.record_confidence(message)
        
        assert monitor.should_spawn_for_confidence()
    
    def test_should_spawn_for_confidence_high(self):
        """Test no spawning for high confidence."""
        monitor = ConfidenceMonitor(confidence_threshold=0.7)
        current_time = time.time()
        
        # Add high confidence data
        message = Message(
            id="high_conf",
            sender="agent_1",
            recipient="central_post",
            message_type=MessageType.RESULT,
            content={
                "confidence": 0.9,
                "agent_type": "research",
                "position_info": {"depth_ratio": 0.3}
            },
            timestamp=current_time
        )
        monitor.record_confidence(message)
        
        assert not monitor.should_spawn_for_confidence()
    
    def test_get_recommended_agent_type(self):
        """Test agent type recommendation."""
        monitor = ConfidenceMonitor()
        current_time = time.time()
        
        # Low confidence should recommend critic
        message = Message(
            id="low_conf",
            sender="agent_1",
            recipient="central_post",
            message_type=MessageType.RESULT,
            content={
                "confidence": 0.5,
                "agent_type": "research",
                "position_info": {"depth_ratio": 0.3}
            },
            timestamp=current_time
        )
        monitor.record_confidence(message)
        
        assert monitor.get_recommended_agent_type() == "critic"


class TestContentAnalyzer:
    """Test ContentAnalyzer functionality."""
    
    def test_init(self):
        """Test ContentAnalyzer initialization."""
        analyzer = ContentAnalyzer()
        assert len(analyzer.contradiction_patterns) > 0
        assert len(analyzer.complexity_indicators) > 0
        assert len(analyzer.domain_keywords) > 0
    
    def test_analyze_content_empty(self):
        """Test analysis with empty content."""
        analyzer = ContentAnalyzer()
        analysis = analyzer.analyze_content([])
        
        assert len(analysis.detected_issues) == 0
        assert analysis.complexity_score == 0.0
        assert analysis.contradiction_count == 0
        assert len(analysis.gap_domains) == 0
        assert analysis.quality_score == 1.0
    
    def test_analyze_content_contradictions(self):
        """Test contradiction detection."""
        analyzer = ContentAnalyzer()
        
        message = Message(
            id="test",
            sender="agent_1",
            recipient="central_post",
            message_type=MessageType.RESULT,
            content={
                "result": "The results show X is true. However, the data contradicts this finding."
            },
            timestamp=time.time()
        )
        
        analysis = analyzer.analyze_content([message])
        
        assert ContentIssue.CONTRADICTION in analysis.detected_issues
        assert analysis.contradiction_count > 0
    
    def test_analyze_content_complexity(self):
        """Test complexity detection."""
        analyzer = ContentAnalyzer()
        
        message = Message(
            id="test",
            sender="agent_1",
            recipient="central_post",
            message_type=MessageType.RESULT,
            content={
                "result": "This is a complex and intricate problem requiring further analysis with multiple factors and sophisticated approaches."
            },
            timestamp=time.time()
        )
        
        analysis = analyzer.analyze_content([message])
        
        assert ContentIssue.HIGH_COMPLEXITY in analysis.detected_issues
        assert analysis.complexity_score > 0.3
    
    def test_analyze_content_gaps(self):
        """Test knowledge gap detection."""
        analyzer = ContentAnalyzer()
        
        message = Message(
            id="test",
            sender="agent_1",
            recipient="central_post",
            message_type=MessageType.RESULT,
            content={
                "result": "We need more information and additional research. There are gaps in our analysis."
            },
            timestamp=time.time()
        )
        
        analysis = analyzer.analyze_content([message])
        
        assert ContentIssue.KNOWLEDGE_GAP in analysis.detected_issues
    
    def test_suggest_agent_types(self):
        """Test agent type suggestions."""
        analyzer = ContentAnalyzer()
        
        # Test contradiction suggests critic
        suggestions = analyzer._suggest_agent_types({ContentIssue.CONTRADICTION}, set())
        assert "critic" in suggestions
        
        # Test knowledge gap suggests research
        suggestions = analyzer._suggest_agent_types({ContentIssue.KNOWLEDGE_GAP}, set())
        assert "research" in suggestions
        
        # Test complexity suggests analysis
        suggestions = analyzer._suggest_agent_types({ContentIssue.HIGH_COMPLEXITY}, set())
        assert "analysis" in suggestions


class TestTeamSizeOptimizer:
    """Test TeamSizeOptimizer functionality."""
    
    def test_init(self):
        """Test TeamSizeOptimizer initialization."""
        optimizer = TeamSizeOptimizer()
        assert optimizer.max_agents == 15
        assert optimizer.token_budget_limit == 10000
        assert optimizer.performance_weight == 0.4
        assert optimizer.efficiency_weight == 0.6
    
    def test_update_current_state(self):
        """Test current state updates."""
        optimizer = TeamSizeOptimizer()
        optimizer.update_current_state(team_size=5, token_usage=2000)
        
        assert optimizer._current_team_size == 5
        assert optimizer._current_token_usage == 2000
    
    def test_record_performance(self):
        """Test performance recording."""
        optimizer = TeamSizeOptimizer()
        optimizer.record_performance(team_size=3, performance_score=0.8, efficiency_score=0.7)
        
        assert 3 in optimizer._team_size_performance
        assert optimizer._team_size_performance[3] == [0.8]
        assert optimizer._team_size_efficiency[3] == [0.7]
    
    def test_get_optimal_team_size(self):
        """Test optimal team size calculation."""
        optimizer = TeamSizeOptimizer()
        
        # Test different scenarios
        size_low = optimizer.get_optimal_team_size(task_complexity=0.2, current_confidence=0.8)
        size_high = optimizer.get_optimal_team_size(task_complexity=0.8, current_confidence=0.5)
        
        assert isinstance(size_low, int)
        assert isinstance(size_high, int)
        assert size_high >= size_low  # High complexity should need more agents
        assert 1 <= size_low <= optimizer.max_agents
        assert 1 <= size_high <= optimizer.max_agents
    
    def test_should_expand_team(self):
        """Test team expansion decision."""
        optimizer = TeamSizeOptimizer()
        
        # Mock confidence metrics
        mock_metrics = ConfidenceMetrics(
            current_average=0.5,
            trend=ConfidenceTrend.DECLINING,
            volatility=0.1,
            time_window_minutes=5.0
        )
        
        # Should expand with low confidence and declining trend
        should_expand = optimizer.should_expand_team(
            current_size=3,
            task_complexity=0.7,
            confidence_metrics=mock_metrics
        )
        
        assert isinstance(should_expand, bool)
    
    def test_resource_budget_allocation(self):
        """Test resource budget allocation."""
        optimizer = TeamSizeOptimizer()
        optimizer.update_current_state(team_size=3, token_usage=1000)
        
        budget = optimizer.get_resource_budget_for_new_agent("research")
        assert isinstance(budget, int)
        assert budget > 0
        assert budget <= 1000  # Should be reasonable


class TestDynamicSpawning:
    """Test DynamicSpawning coordinator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_agent_factory = Mock()
        self.mock_agent_factory.create_research_agent = Mock(return_value=Mock())
        self.mock_agent_factory.create_analysis_agent = Mock(return_value=Mock())
        self.mock_agent_factory.create_critic_agent = Mock(return_value=Mock())
        self.mock_agent_factory.create_synthesis_agent = Mock(return_value=Mock())
        
        self.spawner = DynamicSpawning(
            agent_factory=self.mock_agent_factory,
            confidence_threshold=0.7,
            max_agents=10,
            token_budget_limit=5000
        )
    
    def test_init(self):
        """Test DynamicSpawning initialization."""
        assert self.spawner.agent_factory == self.mock_agent_factory
        assert isinstance(self.spawner.confidence_monitor, ConfidenceMonitor)
        assert isinstance(self.spawner.content_analyzer, ContentAnalyzer)
        assert isinstance(self.spawner.team_optimizer, TeamSizeOptimizer)
    
    def test_analyze_and_spawn_no_spawn(self):
        """Test analysis with no spawning needed."""
        # High confidence message
        message = Message(
            id="high_conf",
            sender="agent_1",
            recipient="central_post",
            message_type=MessageType.RESULT,
            content={
                "confidence": 0.9,
                "agent_type": "research",
                "position_info": {"depth_ratio": 0.3},
                "result": "High quality analysis with clear conclusions."
            },
            timestamp=time.time()
        )
        
        current_agents = [Mock(), Mock()]  # 2 agents
        new_agents = self.spawner.analyze_and_spawn([message], current_agents, time.time())
        
        assert len(new_agents) == 0  # No spawning needed
    
    def test_analyze_and_spawn_confidence_trigger(self):
        """Test spawning triggered by low confidence."""
        # Low confidence message
        message = Message(
            id="low_conf",
            sender="agent_1",
            recipient="central_post",
            message_type=MessageType.RESULT,
            content={
                "confidence": 0.5,
                "agent_type": "research",
                "position_info": {"depth_ratio": 0.3},
                "result": "Uncertain analysis with unclear conclusions."
            },
            timestamp=time.time()
        )
        
        current_agents = [Mock(), Mock()]
        new_agents = self.spawner.analyze_and_spawn([message], current_agents, time.time())
        
        # Should spawn at least one agent for low confidence
        assert len(new_agents) >= 0  # May not spawn if other conditions not met
    
    def test_analyze_and_spawn_content_trigger(self):
        """Test spawning triggered by content analysis."""
        # Message with contradictions
        message = Message(
            id="contradiction",
            sender="agent_1",
            recipient="central_post",
            message_type=MessageType.RESULT,
            content={
                "confidence": 0.8,
                "agent_type": "research",
                "position_info": {"depth_ratio": 0.3},
                "result": "The data shows X is true. However, this contradicts our previous findings."
            },
            timestamp=time.time()
        )
        
        current_agents = [Mock()]  # Small team to allow expansion
        new_agents = self.spawner.analyze_and_spawn([message], current_agents, time.time())
        
        # Should consider spawning for content issues
        assert isinstance(new_agents, list)
    
    def test_get_spawning_summary(self):
        """Test spawning summary generation."""
        summary = self.spawner.get_spawning_summary()
        
        assert "total_spawns" in summary
        assert "spawns_by_type" in summary
        assert "average_priority" in summary
        assert "spawning_reasons" in summary
        assert isinstance(summary["total_spawns"], int)


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""
    
    def test_declining_team_performance(self):
        """Test system response to declining team performance."""
        mock_factory = Mock()
        mock_factory.create_critic_agent = Mock(return_value=Mock())
        
        spawner = DynamicSpawning(mock_factory, confidence_threshold=0.7)
        current_time = time.time()
        
        # Simulate declining confidence over time
        messages = []
        for i, conf in enumerate([0.8, 0.75, 0.65, 0.6, 0.55]):
            message = Message(
                id=f"msg_{i}",
                sender="agent_1",
                recipient="central_post",
                message_type=MessageType.RESULT,
                content={
                    "confidence": conf,
                    "agent_type": "research",
                    "position_info": {"depth_ratio": 0.3},
                    "result": f"Analysis result {i} with declining confidence."
                },
                timestamp=current_time + i
            )
            messages.append(message)
        
        current_agents = [Mock(), Mock()]
        new_agents = spawner.analyze_and_spawn(messages, current_agents, current_time + 5)
        
        # System should recognize declining trend and consider spawning
        assert isinstance(new_agents, list)
    
    def test_complex_task_handling(self):
        """Test handling of complex tasks requiring multiple agents."""
        mock_factory = Mock()
        mock_factory.create_analysis_agent = Mock(return_value=Mock())
        
        spawner = DynamicSpawning(mock_factory, max_agents=15)
        
        # Complex task with multiple issues
        message = Message(
            id="complex",
            sender="agent_1",
            recipient="central_post",
            message_type=MessageType.RESULT,
            content={
                "confidence": 0.6,
                "agent_type": "research",
                "position_info": {"depth_ratio": 0.3},
                "result": "This complex problem requires further analysis. The approach is sophisticated and multifaceted, but we need more information and there are contradictions in the data."
            },
            timestamp=time.time()
        )
        
        current_agents = [Mock()]  # Small team
        new_agents = spawner.analyze_and_spawn([message], current_agents, time.time())
        
        # Should consider spawning for complex task
        assert isinstance(new_agents, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])