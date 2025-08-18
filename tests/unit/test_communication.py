"""
Test suite for communication system.

Following test-first development: these tests define the expected behavior
of spoke-based communication between agents and the central post BEFORE implementation.

Tests validate communication patterns matching the OpenSCAD model from thefelix.md:
- Spoke-based message passing from agents to central post
- Central post coordination and response handling
- Message queuing and delivery guarantees
- Performance metrics for communication overhead (Hypothesis H2)
"""

import pytest
from unittest.mock import Mock
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime


# Import future communication classes (to be implemented)
from src.communication.central_post import CentralPost, Message, MessageType
from src.communication.spoke import Spoke, SpokeConnection
from src.agents.agent import Agent, AgentState
from src.core.helix_geometry import HelixGeometry


# Removed TestMessage class to avoid pytest collection warning


class TestCentralPost:
    """Test central coordination system and message handling."""
    
    @pytest.fixture
    def central_post(self):
        """Create central post instance for testing."""
        return CentralPost(max_agents=133)
    
    @pytest.fixture
    def standard_helix(self):
        """Create helix with OpenSCAD model parameters."""
        return HelixGeometry(
            top_radius=33.0,
            bottom_radius=0.001,
            height=33.0,
            turns=33
        )
    
    def test_central_post_initialization(self, central_post):
        """Test central post can be initialized with configuration."""
        assert central_post.max_agents == 133
        assert central_post.active_connections == 0
        assert central_post.message_queue_size == 0
        assert central_post.is_active is True
    
    def test_agent_registration(self, central_post, standard_helix):
        """Test agents can register with central post."""
        agent = Agent(agent_id="test_001", spawn_time=0.5, helix=standard_helix)
        
        # Register agent
        connection_id = central_post.register_agent(agent)
        
        assert connection_id is not None
        assert central_post.active_connections == 1
        assert central_post.is_agent_registered(agent.agent_id) is True
    
    def test_agent_deregistration(self, central_post, standard_helix):
        """Test agents can deregister from central post."""
        agent = Agent(agent_id="test_002", spawn_time=0.3, helix=standard_helix)
        
        # Register then deregister
        connection_id = central_post.register_agent(agent)
        success = central_post.deregister_agent(agent.agent_id)
        
        assert success is True
        assert central_post.active_connections == 0
        assert central_post.is_agent_registered(agent.agent_id) is False
    
    def test_message_queuing(self, central_post, standard_helix):
        """Test central post can queue messages from agents."""
        # Register agent first
        agent = Agent("agent_001", 0.3, standard_helix)
        central_post.register_agent(agent)
        
        message = Message(
            sender_id="agent_001",
            message_type=MessageType.TASK_REQUEST,
            content={"task_type": "word_count", "data": "hello world"},
            timestamp=0.5
        )
        
        # Queue message
        message_id = central_post.queue_message(message)
        
        assert message_id is not None
        assert central_post.message_queue_size == 1
        assert central_post.has_pending_messages() is True
    
    def test_message_processing_order(self, central_post, standard_helix):
        """Test messages are processed in order (FIFO)."""
        # Register agents first
        for i in [1, 2, 3]:
            agent = Agent(f"agent_{i:03d}", 0.1, standard_helix)
            central_post.register_agent(agent)
        
        # Queue multiple messages
        msg1 = Message("agent_001", MessageType.TASK_REQUEST, {}, 0.1)
        msg2 = Message("agent_002", MessageType.STATUS_UPDATE, {}, 0.2)
        msg3 = Message("agent_003", MessageType.TASK_COMPLETE, {}, 0.3)
        
        id1 = central_post.queue_message(msg1)
        id2 = central_post.queue_message(msg2)
        id3 = central_post.queue_message(msg3)
        
        # Process messages - should come out in order
        processed1 = central_post.process_next_message()
        processed2 = central_post.process_next_message()
        processed3 = central_post.process_next_message()
        
        assert processed1.sender_id == "agent_001"
        assert processed2.sender_id == "agent_002"
        assert processed3.sender_id == "agent_003"
        assert central_post.message_queue_size == 0
    
    def test_maximum_agent_limit(self, central_post, standard_helix):
        """Test central post enforces maximum agent connections."""
        # Set lower limit for testing
        central_post.max_agents = 2
        
        agent1 = Agent("agent_001", 0.1, standard_helix)
        agent2 = Agent("agent_002", 0.2, standard_helix)
        agent3 = Agent("agent_003", 0.3, standard_helix)
        
        # First two should succeed
        conn1 = central_post.register_agent(agent1)
        conn2 = central_post.register_agent(agent2)
        
        assert conn1 is not None
        assert conn2 is not None
        assert central_post.active_connections == 2
        
        # Third should fail
        with pytest.raises(ValueError, match="Maximum agent connections exceeded"):
            central_post.register_agent(agent3)


class TestSpokeConnection:
    """Test spoke-based communication channels."""
    
    @pytest.fixture
    def standard_helix(self):
        """Create helix with OpenSCAD model parameters."""
        return HelixGeometry(
            top_radius=33.0,
            bottom_radius=0.001,
            height=33.0,
            turns=33
        )
    
    @pytest.fixture
    def central_post(self):
        """Create central post for spoke testing."""
        return CentralPost(max_agents=10)
    
    @pytest.fixture
    def test_agent(self, standard_helix):
        """Create agent for spoke testing."""
        return Agent(agent_id="spoke_test", spawn_time=0.4, helix=standard_helix)
    
    def test_spoke_creation(self, test_agent, central_post):
        """Test spoke connection can be created between agent and central post."""
        spoke = Spoke(agent=test_agent, central_post=central_post)
        
        assert spoke.agent_id == test_agent.agent_id
        assert spoke.central_post == central_post
        assert spoke.is_connected is True
        assert spoke.message_count == 0
    
    def test_spoke_message_sending(self, test_agent, central_post):
        """Test messages can be sent through spoke connection."""
        spoke = Spoke(agent=test_agent, central_post=central_post)
        
        # Send message through spoke
        message = Message(
            sender_id=test_agent.agent_id,
            message_type=MessageType.TASK_REQUEST,
            content={"request": "word_counting_task"},
            timestamp=0.6
        )
        
        message_id = spoke.send_message(message)
        
        assert message_id is not None
        assert spoke.message_count == 1
        assert central_post.message_queue_size == 1
    
    def test_spoke_bidirectional_communication(self, test_agent, central_post):
        """Test spoke allows bidirectional communication."""
        spoke = Spoke(agent=test_agent, central_post=central_post)
        
        # Agent sends message to central post
        request = Message(
            sender_id=test_agent.agent_id,
            message_type=MessageType.TASK_REQUEST,
            content={"task": "process_data"},
            timestamp=0.5
        )
        
        spoke.send_message(request)
        
        # Central post processes and responds
        processed_msg = central_post.process_next_message()
        response = Message(
            sender_id="central_post",
            message_type=MessageType.TASK_ASSIGNMENT,
            content={"task_id": "task_123", "data": "sample data"},
            timestamp=0.51
        )
        
        # Send response back through spoke
        response_id = spoke.receive_message(response)
        
        assert response_id is not None
        assert len(spoke.get_received_messages()) == 1
    
    def test_spoke_connection_lifecycle(self, test_agent, central_post):
        """Test spoke connection establishment and teardown."""
        spoke = Spoke(agent=test_agent, central_post=central_post)
        
        # Initial state
        assert spoke.is_connected is True
        assert central_post.active_connections == 1
        
        # Disconnect
        spoke.disconnect()
        
        assert spoke.is_connected is False
        assert central_post.active_connections == 0
        
        # Reconnect
        spoke.reconnect()
        
        assert spoke.is_connected is True
        assert central_post.active_connections == 1
    
    def test_spoke_message_reliability(self, test_agent, central_post):
        """Test spoke ensures reliable message delivery."""
        spoke = Spoke(agent=test_agent, central_post=central_post)
        
        # Send message with delivery confirmation
        message = Message(
            sender_id=test_agent.agent_id,
            message_type=MessageType.STATUS_UPDATE,
            content={"status": "processing", "progress": 0.5},
            timestamp=0.7
        )
        
        message_id = spoke.send_message_reliable(message)
        
        # Should get delivery confirmation
        assert message_id is not None
        assert spoke.is_message_delivered(message_id) is True
        assert spoke.get_delivery_time(message_id) is not None


class TestMessageTypes:
    """Test different message types and their handling."""
    
    def test_message_type_enumeration(self):
        """Test message types are properly defined."""
        assert MessageType.TASK_REQUEST.value == "task_request"
        assert MessageType.TASK_ASSIGNMENT.value == "task_assignment"
        assert MessageType.STATUS_UPDATE.value == "status_update"
        assert MessageType.TASK_COMPLETE.value == "task_complete"
        assert MessageType.ERROR_REPORT.value == "error_report"
    
    def test_message_creation_with_types(self):
        """Test messages can be created with different types."""
        # Task request message
        task_msg = Message(
            sender_id="agent_001",
            message_type=MessageType.TASK_REQUEST,
            content={"task_type": "word_count"},
            timestamp=0.5
        )
        
        assert task_msg.message_type == MessageType.TASK_REQUEST
        assert task_msg.sender_id == "agent_001"
        
        # Status update message
        status_msg = Message(
            sender_id="agent_002",
            message_type=MessageType.STATUS_UPDATE,
            content={"progress": 0.75, "current_position": [1.2, 3.4, 5.6]},
            timestamp=0.8
        )
        
        assert status_msg.message_type == MessageType.STATUS_UPDATE
        assert status_msg.content["progress"] == 0.75


class TestCommunicationPerformance:
    """Test communication system performance metrics for Hypothesis H2."""
    
    @pytest.fixture
    def standard_helix(self):
        """Create helix with OpenSCAD model parameters."""
        return HelixGeometry(
            top_radius=33.0,
            bottom_radius=0.001,
            height=33.0,
            turns=33
        )
    
    @pytest.fixture
    def performance_central_post(self):
        """Create central post configured for performance testing."""
        return CentralPost(max_agents=133, enable_metrics=True)
    
    def test_message_throughput_measurement(self, performance_central_post, standard_helix):
        """Test system can measure message throughput."""
        # Register agents first
        for i in range(100):
            agent = Agent(f"agent_{i:03d}", 0.1, standard_helix)
            performance_central_post.register_agent(agent)
        
        # Send batch of messages
        messages = []
        for i in range(100):
            msg = Message(
                sender_id=f"agent_{i:03d}",
                message_type=MessageType.STATUS_UPDATE,
                content={"progress": i/100},
                timestamp=i/1000
            )
            messages.append(msg)
        
        # Queue all messages and measure processing time
        start_time = performance_central_post.get_current_time()
        
        for msg in messages:
            performance_central_post.queue_message(msg)
        
        # Process all messages
        while performance_central_post.has_pending_messages():
            performance_central_post.process_next_message()
        
        end_time = performance_central_post.get_current_time()
        
        # Check performance metrics
        throughput = performance_central_post.get_message_throughput()
        processing_time = end_time - start_time
        
        assert throughput > 0
        assert processing_time > 0
        assert performance_central_post.total_messages_processed == 100
    
    def test_communication_overhead_tracking(self, performance_central_post):
        """Test system tracks communication overhead vs processing time."""
        # This validates Hypothesis H2: spoke communication reduces overhead
        
        # Simulate agent processing with communication
        agent_processing_time = 0.1  # seconds
        communication_time = performance_central_post.measure_communication_overhead(
            num_messages=10,
            processing_time=agent_processing_time
        )
        
        # Communication overhead should be measurable
        assert communication_time >= 0
        overhead_ratio = communication_time / agent_processing_time
        
        # Store metrics for hypothesis validation
        performance_central_post.record_overhead_ratio(overhead_ratio)
        
        # Should be able to retrieve metrics
        avg_overhead = performance_central_post.get_average_overhead_ratio()
        assert avg_overhead >= 0
    
    def test_scalability_with_multiple_agents(self, performance_central_post, standard_helix):
        """Test communication system scales with increasing agent count."""
        agent_counts = [10, 50, 100, 133]  # Up to OpenSCAD model size
        
        for count in agent_counts:
            # Create fresh central post for each iteration to avoid registration conflicts
            test_central_post = CentralPost(max_agents=count, enable_metrics=True)
            
            # Register agents for this test
            for i in range(count):
                agent = Agent(f"scale_agent_{i:03d}", 0.1, standard_helix)
                test_central_post.register_agent(agent)
            
            # Measure performance with different agent counts
            start_time = test_central_post.get_current_time()
            
            # Simulate concurrent messages from multiple agents
            for i in range(count):
                msg = Message(
                    sender_id=f"scale_agent_{i:03d}",
                    message_type=MessageType.STATUS_UPDATE,
                    content={"agent_count": count},
                    timestamp=start_time
                )
                test_central_post.queue_message(msg)
            
            # Process all messages
            while test_central_post.has_pending_messages():
                test_central_post.process_next_message()
            
            end_time = test_central_post.get_current_time()
            
            # Record scaling metrics
            processing_time = end_time - start_time
            test_central_post.record_scaling_metric(count, processing_time)
            
            # Store metrics in main central post for final validation
            performance_central_post.record_scaling_metric(count, processing_time)
        
        # Should have scaling data for analysis
        scaling_data = performance_central_post.get_scaling_metrics()
        assert len(scaling_data) == len(agent_counts)
        
        # Check that system handles maximum load (133 agents)
        max_load_time = scaling_data[133]  # Time for 133 agents
        assert max_load_time < 1.0  # Should process within 1 second


class TestCommunicationIntegration:
    """Test integration between agents, spokes, and central post."""
    
    @pytest.fixture
    def integration_setup(self):
        """Create full communication system for integration testing."""
        helix = HelixGeometry(33.0, 0.001, 33.0, 33)
        central_post = CentralPost(max_agents=10)
        
        # Create multiple agents
        agents = []
        spokes = []
        
        for i in range(5):
            agent = Agent(f"integration_agent_{i:03d}", i/10, helix)
            spoke = Spoke(agent=agent, central_post=central_post)
            agents.append(agent)
            spokes.append(spoke)
        
        return {
            'helix': helix,
            'central_post': central_post,
            'agents': agents,
            'spokes': spokes
        }
    
    def test_full_communication_workflow(self, integration_setup):
        """Test complete workflow: agent spawn, communication, task processing."""
        setup = integration_setup
        central_post = setup['central_post']
        agents = setup['agents']
        spokes = setup['spokes']
        
        # Simulate time progression with communication
        current_time = 0.5
        
        for i, (agent, spoke) in enumerate(zip(agents, spokes)):
            if agent.can_spawn(current_time):
                # Agent spawns and requests task
                mock_task = Mock()
                mock_task.id = f"integration_task_{i}"
                agent.spawn(current_time, mock_task)
                
                # Send task request through spoke
                request = Message(
                    sender_id=agent.agent_id,
                    message_type=MessageType.TASK_REQUEST,
                    content={"request_type": "word_count"},
                    timestamp=current_time
                )
                spoke.send_message(request)
        
        # Central post processes all requests
        task_assignments = []
        while central_post.has_pending_messages():
            msg = central_post.process_next_message()
            if msg.message_type == MessageType.TASK_REQUEST:
                # Generate task assignment
                assignment = Message(
                    sender_id="central_post",
                    message_type=MessageType.TASK_ASSIGNMENT,
                    content={
                        "task_id": f"task_{len(task_assignments)}",
                        "data": "sample text for processing"
                    },
                    timestamp=current_time + 0.01
                )
                task_assignments.append(assignment)
        
        # Should have generated assignments for spawned agents
        spawned_count = sum(1 for agent in agents if agent.state != AgentState.WAITING)
        assert len(task_assignments) == spawned_count
        assert central_post.message_queue_size == 0
        
        # Validate communication system is operational
        assert central_post.active_connections == len(spokes)
        assert all(spoke.is_connected for spoke in spokes)