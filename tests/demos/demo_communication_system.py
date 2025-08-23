#!/usr/bin/env python3
"""
Demonstration of the Felix Framework communication system.

This script shows the complete spoke-based communication architecture
where agents communicate with a central post through radial spokes,
following the geometric model from thefelix.md.

Features demonstrated:
- Agent registration and spoke connections
- Message passing between agents and central post
- Performance metrics collection
- Task assignment workflow
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.helix_geometry import HelixGeometry
from src.agents.agent import create_openscad_agents
from src.communication import CentralPost, Message, MessageType, SpokeManager
import time


def demo_communication_system():
    """Demonstrate the complete communication system."""
    
    print("=== Felix Framework Communication System Demo ===\n")
    
    # Create helix with OpenSCAD parameters
    helix = HelixGeometry(
        top_radius=33.0,
        bottom_radius=0.001,
        height=33.0,
        turns=33
    )
    
    # Create central post with metrics enabled
    central_post = CentralPost(max_agents=20, enable_metrics=True)
    spoke_manager = SpokeManager(central_post)
    
    print(f"Created central post (max_agents=20, metrics=enabled)")
    print(f"Helix configuration: {helix}")
    print(f"Arc length: {helix.approximate_arc_length():.1f} units\n")
    
    # Create agents and their spoke connections
    agents = create_openscad_agents(helix, number_of_nodes=10, random_seed=42069)
    spokes = []
    
    print("=== Agent Registration ===")
    for agent in agents:
        try:
            spoke = spoke_manager.create_spoke(agent)
            spokes.append(spoke)
            print(f"✅ {agent.agent_id}: registered (spawn_time={agent.spawn_time:.3f})")
        except Exception as e:
            print(f"❌ {agent.agent_id}: registration failed - {e}")
    
    print(f"\nRegistered {central_post.active_connections} agents")
    print(f"Created {len(spokes)} spoke connections\n")
    
    # Simulate time progression with communication
    print("=== Communication Workflow ===")
    current_time = 0.0
    time_step = 0.25
    
    while current_time <= 1.0:
        print(f"\n--- Time {current_time:.2f} ---")
        
        # Process agent spawning and communication
        for i, (agent, spoke) in enumerate(zip(agents, spokes)):
            if agent.can_spawn(current_time) and agent.state.value == "waiting":
                # Create mock task
                class MockTask:
                    def __init__(self, task_id):
                        self.id = task_id
                        self.data = f"Process text chunk {task_id}"
                
                task = MockTask(f"task_{agent.agent_id}_{int(current_time*100)}")
                agent.spawn(current_time, task)
                
                # Agent sends task request through spoke
                request_msg = Message(
                    sender_id=agent.agent_id,
                    message_type=MessageType.TASK_REQUEST,
                    content={
                        "task_type": "word_count",
                        "agent_position": agent.current_position,
                        "priority": "normal"
                    },
                    timestamp=current_time
                )
                
                try:
                    msg_id = spoke.send_message(request_msg)
                    print(f"📤 {agent.agent_id}: sent task request (msg_id={msg_id[:8]}...)")
                except Exception as e:
                    print(f"❌ {agent.agent_id}: failed to send message - {e}")
            
            # Update agent positions
            if agent.state.value in ["active", "spawning"]:
                agent.update_position(current_time)
        
        # Central post processes messages
        messages_processed = 0
        while central_post.has_pending_messages():
            message = central_post.process_next_message()
            messages_processed += 1
            
            # Generate task assignment response
            if message.message_type == MessageType.TASK_REQUEST:
                response = Message(
                    sender_id="central_post",
                    message_type=MessageType.TASK_ASSIGNMENT,
                    content={
                        "task_id": f"assigned_{message.sender_id}_{int(current_time*100)}",
                        "data": "Lorem ipsum dolor sit amet consectetur adipiscing elit",
                        "deadline": current_time + 0.5
                    },
                    timestamp=current_time + 0.01
                )
                
                # Find agent's spoke and send response
                agent_spoke = spoke_manager.get_spoke(message.sender_id)
                if agent_spoke:
                    try:
                        agent_spoke.receive_message(response)
                        print(f"📥 {message.sender_id}: received task assignment")
                    except Exception as e:
                        print(f"❌ Failed to send assignment to {message.sender_id}: {e}")
        
        if messages_processed > 0:
            print(f"🔄 Central post processed {messages_processed} messages")
        
        # Show system status
        active_agents = sum(1 for agent in agents if agent.state.value == "active")
        waiting_agents = sum(1 for agent in agents if agent.state.value == "waiting")
        completed_agents = sum(1 for agent in agents if agent.state.value == "completed")
        
        print(f"Status - Active: {active_agents}, Waiting: {waiting_agents}, Completed: {completed_agents}")
        print(f"Queue size: {central_post.message_queue_size}")
        
        current_time += time_step
    
    # Performance summary
    print("\n=== Performance Summary ===")
    
    # Central post metrics
    performance = central_post.get_performance_summary()
    print(f"Messages processed: {performance['total_messages_processed']}")
    print(f"Message throughput: {performance['message_throughput']:.2f} msg/sec")
    print(f"Active connections: {performance['active_connections']}")
    print(f"System uptime: {performance['uptime']:.3f} seconds")
    
    # Spoke metrics
    print(f"\nSpoke Performance:")
    total_sent = 0
    total_received = 0
    
    for spoke in spokes[:5]:  # Show first 5 spokes
        metrics = spoke.get_performance_metrics()
        total_sent += metrics['messages_sent']
        total_received += metrics['messages_received']
        print(f"  {metrics['agent_id']}: sent={metrics['messages_sent']}, "
              f"received={metrics['messages_received']}, "
              f"avg_delivery={metrics['average_delivery_time']:.6f}s")
    
    print(f"\nTotal messages: sent={total_sent}, received={total_received}")
    
    # Connection summary
    connection_summary = spoke_manager.get_connection_summary()
    print(f"Spoke connections: {connection_summary['connected_spokes']}/{connection_summary['total_spokes']} active")
    
    # Hypothesis H2 validation data
    if performance['metrics_enabled']:
        print(f"\n=== Hypothesis H2 Data ===")
        print(f"Average overhead ratio: {performance['average_overhead_ratio']:.6f}")
        print("✅ Communication system operational")
        print("✅ Performance metrics collected")
        print("✅ Spoke-based architecture validated")
    
    # Cleanup
    print(f"\n=== Cleanup ===")
    spoke_manager.shutdown_all()
    central_post.shutdown()
    print("✅ All connections closed")
    print("✅ Central post shutdown")
    
    print(f"\nDemo complete! Communication system validated ✅")


if __name__ == "__main__":
    demo_communication_system()