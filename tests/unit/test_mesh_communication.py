#!/usr/bin/env python3
"""
Test suite for mesh communication architecture - Felix Framework Phase 5.

This module provides comprehensive testing for the mesh communication implementation
that serves as a comparison baseline against the spoke-based architecture.

Mathematical Foundation:
- Mesh topology: each agent communicates with all other agents (O(N²) connections)
- Message complexity: M_mesh = Σᵢ Σⱼ≠ᵢ m_ij = O(N²)
- Distance-based latency: L_ij = α + β·d_ij + ε_ij
- Memory overhead: O(N²) connection tracking vs O(N) for spoke architecture

Key Features:
- All-to-all connectivity between agents
- Distance-based message routing and latency
- O(N²) scaling characteristics for comparison with O(N) spoke system
- Performance metrics collection for Hypothesis H2 validation

This supports Hypothesis H2 testing by providing measurable communication
overhead characteristics that demonstrate the complexity advantage of 
spoke-based architecture.

Mathematical reference: docs/hypothesis_mathematics.md, Section H2
"""

import pytest
import time
import math
import random
from unittest.mock import Mock, patch
from typing import List, Dict, Set, Tuple, Any

from src.communication.mesh import MeshCommunication, MeshConnection, MeshMessage
from src.agents.agent import Agent
from src.core.helix_geometry import HelixGeometry


class TestMeshMessage:
    """Test mesh message functionality."""
    
    def test_message_creation(self):
        """Test mesh message creation with required parameters."""
        message = MeshMessage(
            sender_id="agent_001",
            recipient_id="agent_002", 
            message_type="TASK_REQUEST",
            content={"task": "process_data"},
            timestamp=12345.67
        )
        
        assert message.sender_id == "agent_001"
        assert message.recipient_id == "agent_002"
        assert message.message_type == "TASK_REQUEST"
        assert message.content == {"task": "process_data"}
        assert message.timestamp == 12345.67
        assert message.message_id is not None
        assert len(message.message_id) > 0
    
    def test_message_validation(self):
        """Test message parameter validation."""
        # Valid message types
        valid_types = ["TASK_REQUEST", "TASK_RESPONSE", "STATUS_UPDATE", "COORDINATION", "ERROR"]
        
        for msg_type in valid_types:
            message = MeshMessage("sender", "recipient", msg_type, {}, 0.0)
            assert message.message_type == msg_type
        
        # Invalid message type
        with pytest.raises(ValueError, match="Invalid message type"):
            MeshMessage("sender", "recipient", "INVALID_TYPE", {}, 0.0)
        
        # Empty sender/recipient
        with pytest.raises(ValueError, match="sender_id cannot be empty"):
            MeshMessage("", "recipient", "STATUS_UPDATE", {}, 0.0)
        
        with pytest.raises(ValueError, match="recipient_id cannot be empty"):
            MeshMessage("sender", "", "STATUS_UPDATE", {}, 0.0)
    
    def test_message_serialization(self):
        """Test message serialization for transmission."""
        message = MeshMessage(
            sender_id="agent_alpha",
            recipient_id="agent_beta",
            message_type="COORDINATION",
            content={"action": "synchronize", "data": [1, 2, 3]},
            timestamp=98765.43
        )
        
        serialized = message.serialize()
        
        assert isinstance(serialized, dict)
        assert serialized["sender_id"] == "agent_alpha"
        assert serialized["recipient_id"] == "agent_beta"
        assert serialized["message_type"] == "COORDINATION"
        assert serialized["content"] == {"action": "synchronize", "data": [1, 2, 3]}
        assert serialized["timestamp"] == 98765.43
        assert "message_id" in serialized
    
    def test_message_deserialization(self):
        """Test message deserialization from transmission data."""
        serialized_data = {
            "sender_id": "agent_gamma",
            "recipient_id": "agent_delta", 
            "message_type": "TASK_RESPONSE",
            "content": {"result": "success", "value": 42},
            "timestamp": 11111.22,
            "message_id": "msg_12345"
        }
        
        message = MeshMessage.deserialize(serialized_data)
        
        assert message.sender_id == "agent_gamma"
        assert message.recipient_id == "agent_delta"
        assert message.message_type == "TASK_RESPONSE"
        assert message.content == {"result": "success", "value": 42}
        assert message.timestamp == 11111.22
        assert message.message_id == "msg_12345"


class TestMeshConnection:
    """Test mesh connection functionality between agent pairs."""
    
    def test_connection_establishment(self):
        """Test establishing connection between two agents."""
        agent_a = "agent_001"
        agent_b = "agent_002"
        distance = 15.75
        
        connection = MeshConnection(agent_a, agent_b, distance)
        
        assert connection.agent_a == agent_a
        assert connection.agent_b == agent_b
        assert connection.distance == distance
        assert connection.message_count == 0
        assert connection.total_latency == 0.0
        assert len(connection.message_queue) == 0
    
    def test_connection_validation(self):
        """Test connection parameter validation."""
        # Valid distance
        connection = MeshConnection("a", "b", 10.0)
        assert connection.distance == 10.0
        
        # Invalid distance (negative)
        with pytest.raises(ValueError, match="Distance must be non-negative"):
            MeshConnection("a", "b", -5.0)
        
        # Self-connection (same agent)
        with pytest.raises(ValueError, match="Cannot create connection to self"):
            MeshConnection("agent_001", "agent_001", 0.0)
    
    def test_message_queueing(self):
        """Test message queueing in connections."""
        connection = MeshConnection("sender", "receiver", 20.0)
        
        message1 = MeshMessage("sender", "receiver", "TASK_REQUEST", {"task": "A"}, 1.0)
        message2 = MeshMessage("sender", "receiver", "STATUS_UPDATE", {"status": "OK"}, 2.0)
        
        # Queue messages
        connection.queue_message(message1)
        assert len(connection.message_queue) == 1
        assert connection.message_count == 1
        
        connection.queue_message(message2)
        assert len(connection.message_queue) == 2
        assert connection.message_count == 2
        
        # Messages should be queued in order
        assert connection.message_queue[0] == message1
        assert connection.message_queue[1] == message2
    
    def test_message_transmission(self):
        """Test message transmission with latency calculation."""
        connection = MeshConnection("sender", "receiver", 10.0)
        message = MeshMessage("sender", "receiver", "COORDINATION", {"sync": True}, 5.0)
        
        # Configure latency model (α=1.0, β=0.1)
        base_latency = 1.0
        distance_coefficient = 0.1
        
        connection.queue_message(message)
        delivered_messages = connection.process_messages(base_latency, distance_coefficient)
        
        assert len(delivered_messages) == 1
        assert delivered_messages[0] == message
        assert len(connection.message_queue) == 0  # Message removed from queue
        
        # Check latency calculation: L = α + β·d = 1.0 + 0.1·10.0 = 2.0
        expected_latency = base_latency + distance_coefficient * connection.distance
        assert abs(connection.total_latency - expected_latency) < 1e-6
    
    def test_connection_metrics(self):
        """Test connection performance metrics."""
        connection = MeshConnection("a", "b", 25.0)
        
        # Send several messages
        for i in range(5):
            message = MeshMessage("a", "b", "STATUS_UPDATE", {"seq": i}, float(i))
            connection.queue_message(message)
        
        # Process with known latency parameters
        connection.process_messages(base_latency=0.5, distance_coefficient=0.05)
        
        metrics = connection.get_metrics()
        
        assert metrics["message_count"] == 5
        assert metrics["distance"] == 25.0
        assert metrics["average_latency"] > 0.0
        assert "total_latency" in metrics
        assert "throughput" in metrics
    
    def test_bidirectional_communication(self):
        """Test communication in both directions."""
        connection = MeshConnection("alpha", "beta", 12.0)
        
        # Message from alpha to beta
        msg_a_to_b = MeshMessage("alpha", "beta", "TASK_REQUEST", {"from": "alpha"}, 1.0)
        connection.queue_message(msg_a_to_b)
        
        # Message from beta to alpha
        msg_b_to_a = MeshMessage("beta", "alpha", "TASK_RESPONSE", {"from": "beta"}, 2.0)
        connection.queue_message(msg_b_to_a)
        
        # Both messages should be handled by the same connection
        delivered = connection.process_messages(0.1, 0.02)
        assert len(delivered) == 2
        
        # Check both directions are supported
        sender_ids = {msg.sender_id for msg in delivered}
        assert sender_ids == {"alpha", "beta"}


class TestMeshCommunication:
    """Test mesh communication system integration."""
    
    def test_mesh_initialization(self):
        """Test mesh communication system creation."""
        mesh = MeshCommunication(max_agents=10)
        
        assert mesh.max_agents == 10
        assert mesh.registered_agents == {}
        assert mesh.connections == {}
        assert mesh.message_count == 0
        assert mesh.total_latency == 0.0
    
    def test_mesh_parameter_validation(self):
        """Test mesh system parameter validation."""
        # Valid parameters
        mesh = MeshCommunication(max_agents=100)
        assert mesh.max_agents == 100
        
        # Invalid parameters
        with pytest.raises(ValueError, match="max_agents must be positive"):
            MeshCommunication(max_agents=0)
        
        with pytest.raises(ValueError, match="max_agents must be positive"):
            MeshCommunication(max_agents=-5)
    
    def test_agent_registration(self):
        """Test agent registration in mesh network."""
        mesh = MeshCommunication(max_agents=5)
        helix = HelixGeometry(33.0, 0.001, 33.0, 33)
        
        # Register first agent
        agent1 = Agent("agent_001", 0.3, helix)
        connection_id1 = mesh.register_agent(agent1)
        
        assert connection_id1 is not None
        assert agent1.agent_id in mesh.registered_agents
        assert mesh.registered_agents[agent1.agent_id]["agent"] == agent1
        
        # Register second agent (should create connection to first)
        agent2 = Agent("agent_002", 0.5, helix)
        connection_id2 = mesh.register_agent(agent2)
        
        assert connection_id2 is not None
        assert agent2.agent_id in mesh.registered_agents
        
        # Check connection created between agents
        connection_key = mesh._get_connection_key("agent_001", "agent_002")
        assert connection_key in mesh.connections
    
    def test_connection_creation_scaling(self):
        """Test O(N²) connection scaling behavior."""
        mesh = MeshCommunication(max_agents=10)
        helix = HelixGeometry(33.0, 0.001, 33.0, 33)
        
        # Register N agents
        N = 5
        agents = []
        for i in range(N):
            agent = Agent(f"agent_{i:03d}", i/10.0, helix)
            agents.append(agent)
            mesh.register_agent(agent)
        
        # Should have N*(N-1)/2 connections (undirected)
        expected_connections = N * (N - 1) // 2
        assert len(mesh.connections) == expected_connections
        
        # Verify all pair combinations exist
        for i in range(N):
            for j in range(i + 1, N):
                agent_a = f"agent_{i:03d}"
                agent_b = f"agent_{j:03d}"
                connection_key = mesh._get_connection_key(agent_a, agent_b)
                assert connection_key in mesh.connections
    
    def test_agent_registration_capacity_limit(self):
        """Test agent registration capacity enforcement."""
        mesh = MeshCommunication(max_agents=2)
        helix = HelixGeometry(33.0, 0.001, 33.0, 33)
        
        # Fill to capacity
        agent1 = Agent("agent_001", 0.1, helix)
        agent2 = Agent("agent_002", 0.2, helix)
        mesh.register_agent(agent1)
        mesh.register_agent(agent2)
        
        # Attempt to exceed capacity
        agent3 = Agent("agent_003", 0.3, helix)
        with pytest.raises(ValueError, match="Maximum agent connections exceeded"):
            mesh.register_agent(agent3)
    
    def test_message_sending_mesh_topology(self):
        """Test message sending in mesh topology."""
        mesh = MeshCommunication(max_agents=4)
        helix = HelixGeometry(33.0, 0.001, 33.0, 33)
        
        # Register 3 agents
        agents = []
        for i in range(3):
            agent = Agent(f"mesh_agent_{i}", i/10.0, helix)
            agent.spawn(0.5, Mock())  # Spawn all agents
            agents.append(agent)
            mesh.register_agent(agent)
        
        # Send message from agent 0 to agent 1
        message = MeshMessage(
            "mesh_agent_0", "mesh_agent_1", 
            "TASK_REQUEST", {"task": "mesh_test"}, 
            time.time()
        )
        
        success = mesh.send_message(message)
        assert success
        
        # Check message queued in connection
        connection_key = mesh._get_connection_key("mesh_agent_0", "mesh_agent_1")
        connection = mesh.connections[connection_key]
        assert len(connection.message_queue) == 1
        assert connection.message_queue[0] == message
    
    def test_message_processing_and_delivery(self):
        """Test message processing and delivery in mesh network."""
        mesh = MeshCommunication(max_agents=3, enable_metrics=True)
        helix = HelixGeometry(33.0, 0.001, 33.0, 33)
        
        # Register and spawn agents at different positions
        agents = []
        for i in range(3):
            agent = Agent(f"delivery_agent_{i}", i/10.0, helix)
            agent.spawn(0.5, Mock())
            # Set different positions for distance calculation
            agent.update_position(0.5 + i * 0.1)
            agents.append(agent)
            mesh.register_agent(agent)
        
        # Send multiple messages
        messages = []
        for i in range(3):
            for j in range(3):
                if i != j:  # Don't send to self
                    message = MeshMessage(
                        f"delivery_agent_{i}", f"delivery_agent_{j}",
                        "COORDINATION", {"from": i, "to": j},
                        time.time()
                    )
                    messages.append(message)
                    mesh.send_message(message)
        
        # Process all messages
        processed_count = mesh.process_all_messages()
        
        # Should process all sent messages
        assert processed_count == len(messages)
        
        # Check metrics updated
        metrics = mesh.get_performance_metrics()
        assert metrics["total_messages"] >= len(messages)
        assert metrics["average_latency"] > 0.0
    
    def test_mesh_performance_metrics(self):
        """Test mesh performance metrics collection."""
        mesh = MeshCommunication(max_agents=4, enable_metrics=True)
        helix = HelixGeometry(33.0, 0.001, 33.0, 33)
        
        # Register agents
        agent_count = 4
        agents = []
        for i in range(agent_count):
            agent = Agent(f"perf_agent_{i}", i/10.0, helix)
            agent.spawn(0.5, Mock())
            agents.append(agent)
            mesh.register_agent(agent)
        
        # Send messages and process
        message_count = 10
        for i in range(message_count):
            sender_idx = i % agent_count
            recipient_idx = (i + 1) % agent_count
            
            message = MeshMessage(
                f"perf_agent_{sender_idx}", f"perf_agent_{recipient_idx}",
                "STATUS_UPDATE", {"seq": i}, time.time()
            )
            mesh.send_message(message)
        
        mesh.process_all_messages()
        
        # Collect and validate metrics
        metrics = mesh.get_performance_metrics()
        
        required_metrics = [
            "total_messages", "connection_count", "average_latency", 
            "throughput", "agent_count", "message_density"
        ]
        
        for metric in required_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], (int, float))
        
        # Validate specific values
        expected_connections = agent_count * (agent_count - 1) // 2
        assert metrics["connection_count"] == expected_connections
        assert metrics["agent_count"] == agent_count
        assert metrics["total_messages"] >= message_count
    
    def test_mesh_distance_calculation(self):
        """Test distance calculation between agents."""
        mesh = MeshCommunication(max_agents=5)
        helix = HelixGeometry(33.0, 0.001, 33.0, 33)
        
        # Create two agents at known positions
        agent1 = Agent("distance_test_1", 0.2, helix)
        agent2 = Agent("distance_test_2", 0.3, helix)  # Spawn time after first agent
        
        # Spawn and position agents
        agent1.spawn(0.5, Mock())
        agent2.spawn(0.5, Mock())
        agent1.update_position(0.6)  # Position along helix 
        agent2.update_position(0.8)  # Position along helix with different progress
        
        # Register agents (this should calculate distance)
        mesh.register_agent(agent1)
        mesh.register_agent(agent2)
        
        # Check connection distance
        connection_key = mesh._get_connection_key("distance_test_1", "distance_test_2")
        connection = mesh.connections[connection_key]
        
        # Distance should be positive and reasonable for helix geometry
        assert connection.distance > 0
        assert connection.distance < 1000  # Reasonable upper bound
    
    def test_mesh_complexity_scaling(self):
        """Test O(N²) scaling characteristics."""
        # Test with different agent counts
        test_sizes = [2, 3, 5, 8]
        connection_counts = []
        
        for N in test_sizes:
            mesh = MeshCommunication(max_agents=N)
            helix = HelixGeometry(33.0, 0.001, 33.0, 33)
            
            # Register N agents
            for i in range(N):
                spawn_time = min(i/10.0, 0.5)  # Ensure spawn time <= 0.5
                agent = Agent(f"scale_test_{i}", spawn_time, helix)
                agent.spawn(0.5, Mock())
                mesh.register_agent(agent)
            
            connection_counts.append(len(mesh.connections))
        
        # Verify O(N²) scaling: connections = N*(N-1)/2
        for i, N in enumerate(test_sizes):
            expected = N * (N - 1) // 2
            assert connection_counts[i] == expected
        
        # Check scaling ratio increases quadratically
        for i in range(1, len(test_sizes)):
            n_prev = test_sizes[i-1]
            n_curr = test_sizes[i]
            ratio_connections = connection_counts[i] / connection_counts[i-1]
            
            # Connection ratio should be greater than linear scaling (n_curr/n_prev)
            linear_ratio = n_curr / n_prev
            assert ratio_connections >= linear_ratio  # Should scale at least linearly
            # Allow some flexibility for discrete scaling effects
            max_expected_ratio = linear_ratio * linear_ratio * 1.5  # Add 50% buffer
            assert ratio_connections <= max_expected_ratio


class TestMeshCommunicationComparison:
    """Test mesh communication features needed for architecture comparison."""
    
    def test_mesh_vs_spoke_connection_count(self):
        """Test connection count comparison between mesh and spoke."""
        agent_counts = [5, 10, 20]
        
        for N in agent_counts:
            # Mesh: N*(N-1)/2 connections
            mesh_connections = N * (N - 1) // 2
            
            # Spoke: N connections (each agent to central post)
            spoke_connections = N
            
            # Mesh should have more connections for N > 2
            if N > 2:
                assert mesh_connections > spoke_connections
                
                # Calculate scaling advantage
                scaling_advantage = mesh_connections / spoke_connections
                expected_advantage = (N - 1) / 2
                assert abs(scaling_advantage - expected_advantage) < 0.001
    
    def test_mesh_latency_distribution(self):
        """Test latency distribution for statistical comparison."""
        mesh = MeshCommunication(max_agents=6, enable_metrics=True)
        helix = HelixGeometry(33.0, 0.001, 33.0, 33)
        
        # Register agents at different positions (varying distances)
        agents = []
        spawn_times = [0.1, 0.15, 0.2, 0.25, 0.3]  # Different spawn times
        for i, spawn_time in enumerate(spawn_times):
            agent = Agent(f"latency_agent_{i}", spawn_time, helix)
            agent.spawn(0.5, Mock())
            # Update to different positions to create distance variation
            update_time = 0.5 + i * 0.1  # Different update times for different progress
            agent.update_position(update_time)
            agents.append(agent)
            mesh.register_agent(agent)
        
        # Send messages between all pairs
        message_count = 0
        for i in range(len(agents)):
            for j in range(len(agents)):
                if i != j:
                    message = MeshMessage(
                        f"latency_agent_{i}", f"latency_agent_{j}",
                        "STATUS_UPDATE", {"test": "latency"}, 
                        time.time()
                    )
                    mesh.send_message(message)
                    message_count += 1
        
        # Process messages
        processed = mesh.process_all_messages()
        assert processed == message_count
        
        # Collect latency data
        latencies = []
        for connection in mesh.connections.values():
            if connection.message_count > 0:
                avg_latency = connection.total_latency / connection.message_count
                latencies.append(avg_latency)
        
        # Should have latency data for all connections
        assert len(latencies) > 0
        assert all(lat > 0 for lat in latencies)
        
        # Latencies should vary based on distance
        assert max(latencies) > min(latencies)
    
    def test_mesh_memory_overhead_tracking(self):
        """Test memory overhead tracking for comparison."""
        mesh = MeshCommunication(max_agents=8, enable_metrics=True)
        helix = HelixGeometry(33.0, 0.001, 33.0, 33)
        
        # Register agents
        agent_count = 6
        for i in range(agent_count):
            agent = Agent(f"memory_agent_{i}", i/10.0, helix)
            agent.spawn(0.5, Mock())
            mesh.register_agent(agent)
        
        # Get memory metrics
        metrics = mesh.get_comparison_metrics()
        
        assert "memory_overhead" in metrics
        assert "connection_memory" in metrics
        assert "message_queue_size" in metrics
        
        # Memory should scale with O(N²) connections
        expected_connections = agent_count * (agent_count - 1) // 2
        assert metrics["connection_memory"] >= expected_connections
        
        # Should be measurably different from O(N) spoke system
        spoke_memory_equivalent = agent_count  # One connection per agent in spoke
        assert metrics["connection_memory"] > spoke_memory_equivalent
    
    def test_mesh_throughput_measurement(self):
        """Test throughput measurement for performance comparison."""
        mesh = MeshCommunication(max_agents=7, enable_metrics=True)
        helix = HelixGeometry(33.0, 0.001, 33.0, 33)
        
        # Register agents
        agent_count = 5
        for i in range(agent_count):
            agent = Agent(f"throughput_agent_{i}", i/10.0, helix)
            agent.spawn(0.5, Mock())
            mesh.register_agent(agent)
        
        # Time message sending and processing
        start_time = time.perf_counter()
        
        # Send batch of messages
        message_count = 20
        for i in range(message_count):
            sender_idx = i % agent_count
            recipient_idx = (i + 1) % agent_count
            
            message = MeshMessage(
                f"throughput_agent_{sender_idx}", f"throughput_agent_{recipient_idx}",
                "TASK_REQUEST", {"batch": i}, time.perf_counter()
            )
            mesh.send_message(message)
        
        # Process all messages
        processed = mesh.process_all_messages()
        end_time = time.perf_counter()
        
        # Calculate throughput
        processing_time = end_time - start_time
        throughput = processed / processing_time if processing_time > 0 else 0
        
        metrics = mesh.get_comparison_metrics()
        assert "throughput" in metrics
        assert metrics["throughput"] > 0
        
        # Should process all sent messages
        assert processed == message_count
    
    def test_mesh_hypothesis_h2_metrics(self):
        """Test metrics specifically needed for Hypothesis H2 validation."""
        mesh = MeshCommunication(max_agents=10, enable_metrics=True)
        helix = HelixGeometry(33.0, 0.001, 33.0, 33)
        
        # Create test scenario matching hypothesis testing requirements
        agent_count = 8
        for i in range(agent_count):
            spawn_time = min(i/10.0, 0.4)  # Ensure spawn time < 0.5
            agent = Agent(f"h2_agent_{i}", spawn_time, helix)
            agent.spawn(0.5, Mock())
            # Distribute positions within valid range
            position_time = 0.5 + min(i * 0.03, 0.4)  # Keep within bounds
            agent.update_position(position_time)
            mesh.register_agent(agent)
        
        # Generate workload for H2 testing
        test_messages = 50
        for i in range(test_messages):
            sender = f"h2_agent_{i % agent_count}"
            recipient = f"h2_agent_{(i + 1) % agent_count}"
            
            message = MeshMessage(
                sender, recipient, "COORDINATION",
                {"h2_test": True, "sequence": i}, time.perf_counter()
            )
            mesh.send_message(message)
        
        # Process and collect H2 metrics
        mesh.process_all_messages()
        h2_metrics = mesh.get_hypothesis_h2_metrics()
        
        # Check required H2 metrics
        h2_required_metrics = [
            "message_complexity",  # Should be O(N²)
            "average_latency",     # For t-test comparison
            "latency_variance",    # For statistical testing
            "connection_overhead", # Memory overhead
            "max_distance",        # Communication distance bounds
            "throughput_msgs_per_sec"  # Performance comparison
        ]
        
        for metric in h2_required_metrics:
            assert metric in h2_metrics
            assert isinstance(h2_metrics[metric], (int, float))
        
        # Validate O(N²) message complexity
        expected_connections = agent_count * (agent_count - 1) // 2
        assert h2_metrics["message_complexity"] >= expected_connections
        
        # Latency should be measurable and positive
        assert h2_metrics["average_latency"] > 0
        assert h2_metrics["latency_variance"] >= 0


# Edge cases and error conditions
class TestMeshCommunicationEdgeCases:
    """Test edge cases and error conditions in mesh communication."""
    
    def test_single_agent_mesh(self):
        """Test mesh with only one agent."""
        mesh = MeshCommunication(max_agents=5)
        helix = HelixGeometry(33.0, 0.001, 33.0, 33)
        
        agent = Agent("solo_mesh_agent", 0.5, helix)
        agent.spawn(0.7, Mock())
        mesh.register_agent(agent)
        
        # Should have no connections with single agent
        assert len(mesh.connections) == 0
        
        # Metrics should handle single agent gracefully
        metrics = mesh.get_performance_metrics()
        assert metrics["agent_count"] == 1
        assert metrics["connection_count"] == 0
    
    def test_message_to_unregistered_agent(self):
        """Test sending message to unregistered agent."""
        mesh = MeshCommunication(max_agents=3)
        helix = HelixGeometry(33.0, 0.001, 33.0, 33)
        
        # Register only one agent
        agent = Agent("registered_agent", 0.3, helix)
        agent.spawn(0.5, Mock())
        mesh.register_agent(agent)
        
        # Attempt to send message to unregistered agent
        message = MeshMessage(
            "registered_agent", "unregistered_agent",
            "ERROR", {"error": "test"}, time.time()
        )
        
        success = mesh.send_message(message)
        assert not success  # Should fail gracefully
    
    def test_mesh_agent_position_edge_cases(self):
        """Test mesh with agents at extreme positions."""
        mesh = MeshCommunication(max_agents=4)
        helix = HelixGeometry(33.0, 0.001, 33.0, 33)
        
        # Create agents at extreme positions
        positions = [0.0, 1.0]  # Bottom and top of helix
        agents = []
        
        for i, pos in enumerate(positions):
            agent = Agent(f"extreme_agent_{i}", 0.1, helix)
            agent.spawn(0.5, Mock())
            # Update to create extreme positions within valid bounds
            # Use pos as fraction of max progress (0.4) to stay within bounds
            target_time = 0.5 + (pos * 0.4)  # spawn_time + fraction of max progress
            agent.update_position(target_time)
            agents.append(agent)
            mesh.register_agent(agent)
        
        # Should create connection between extreme positions
        assert len(mesh.connections) == 1
        
        # Distance should be calculable and reasonable
        connection = list(mesh.connections.values())[0]
        assert connection.distance > 0
        assert connection.distance < 1000  # Reasonable upper bound
    
    def test_mesh_large_message_batch(self):
        """Test mesh with large number of messages."""
        mesh = MeshCommunication(max_agents=5, enable_metrics=True)
        helix = HelixGeometry(33.0, 0.001, 33.0, 33)
        
        # Register small number of agents
        agent_count = 3
        for i in range(agent_count):
            agent = Agent(f"batch_agent_{i}", i/10.0, helix)
            agent.spawn(0.5, Mock())
            mesh.register_agent(agent)
        
        # Send large batch of messages
        large_batch_size = 100
        for i in range(large_batch_size):
            sender_idx = i % agent_count
            recipient_idx = (i + 1) % agent_count
            
            message = MeshMessage(
                f"batch_agent_{sender_idx}", f"batch_agent_{recipient_idx}",
                "STATUS_UPDATE", {"batch_seq": i}, time.time()
            )
            mesh.send_message(message)
        
        # Should handle large batch without errors
        processed = mesh.process_all_messages()
        assert processed == large_batch_size
        
        # Metrics should reflect large batch processing
        metrics = mesh.get_performance_metrics()
        assert metrics["total_messages"] >= large_batch_size
    
    def test_mesh_zero_distance_agents(self):
        """Test agents at same position (zero distance)."""
        mesh = MeshCommunication(max_agents=3)
        helix = HelixGeometry(33.0, 0.001, 33.0, 33)
        
        # Create two agents at same position
        agent1 = Agent("same_pos_1", 0.4, helix)
        agent2 = Agent("same_pos_2", 0.4, helix)
        
        agent1.spawn(0.5, Mock())
        agent2.spawn(0.5, Mock())
        
        # Same position (should result in zero or minimal distance)
        agent1.update_position(0.6)
        agent2.update_position(0.6)
        
        mesh.register_agent(agent1)
        mesh.register_agent(agent2)
        
        # Should handle zero distance gracefully
        assert len(mesh.connections) == 1
        connection = list(mesh.connections.values())[0]
        assert connection.distance >= 0  # Should be non-negative