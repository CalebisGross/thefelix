"""
Central coordination system for the Felix Framework.

The central post manages communication and coordination between agents,
implementing the hub of the spoke-based communication model from thefelix.md.

Mathematical Foundation:
- Spoke communication: O(N) message complexity vs O(N²) mesh topology
- Maximum communication distance: R_top (helix outer radius)
- Performance metrics for Hypothesis H2 validation and statistical analysis

Key Features:
- Agent registration and connection management
- FIFO message queuing with guaranteed ordering
- Performance metrics collection (throughput, latency, overhead ratios)
- Scalability up to 133 agents (matching OpenSCAD model parameters)

Mathematical references:
- docs/mathematical_model.md, Section 5: Spoke geometry and communication complexity
- docs/hypothesis_mathematics.md, Section H2: Communication overhead analysis and proofs
- Theoretical proof of O(N) vs O(N²) scaling advantage in hypothesis documentation

Implementation supports rigorous testing of Hypothesis H2 communication efficiency claims.
"""

import time
import uuid
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import deque
from queue import Queue, Empty


class MessageType(Enum):
    """Types of messages in the communication system."""
    TASK_REQUEST = "task_request"
    TASK_ASSIGNMENT = "task_assignment"
    STATUS_UPDATE = "status_update"
    TASK_COMPLETE = "task_complete"
    ERROR_REPORT = "error_report"


@dataclass
class Message:
    """Message structure for communication between agents and central post."""
    sender_id: str
    message_type: MessageType
    content: Dict[str, Any]
    timestamp: float
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))


class CentralPost:
    """
    Central coordination system managing all agent communication.
    
    The central post acts as the hub in the spoke-based communication model,
    processing messages from agents and coordinating task assignments.
    """
    
    def __init__(self, max_agents: int = 133, enable_metrics: bool = False):
        """
        Initialize central post with configuration parameters.
        
        Args:
            max_agents: Maximum number of concurrent agent connections
            enable_metrics: Whether to collect performance metrics
        """
        self.max_agents = max_agents
        self.enable_metrics = enable_metrics
        
        # Connection management
        self._registered_agents: Dict[str, str] = {}  # agent_id -> connection_id
        self._connection_times: Dict[str, float] = {}  # agent_id -> registration_time
        
        # Message processing
        self._message_queue: Queue = Queue()
        self._processed_messages: List[Message] = []
        
        # Performance metrics (for Hypothesis H2)
        self._metrics_enabled = enable_metrics
        self._start_time = time.time()
        self._total_messages_processed = 0
        self._processing_times: List[float] = []
        self._overhead_ratios: List[float] = []
        self._scaling_metrics: Dict[int, float] = {}
        
        # System state
        self._is_active = True
    
    @property
    def active_connections(self) -> int:
        """Get number of currently registered agents."""
        return len(self._registered_agents)
    
    @property
    def message_queue_size(self) -> int:
        """Get number of pending messages in queue."""
        return self._message_queue.qsize()
    
    @property
    def is_active(self) -> bool:
        """Check if central post is active and accepting connections."""
        return self._is_active
    
    @property
    def total_messages_processed(self) -> int:
        """Get total number of messages processed."""
        return self._total_messages_processed
    
    def register_agent(self, agent) -> str:
        """
        Register an agent with the central post.
        
        Args:
            agent: Agent instance to register
            
        Returns:
            Connection ID for the registered agent
            
        Raises:
            ValueError: If maximum connections exceeded or agent already registered
        """
        if self.active_connections >= self.max_agents:
            raise ValueError("Maximum agent connections exceeded")
        
        if agent.agent_id in self._registered_agents:
            raise ValueError(f"Agent {agent.agent_id} already registered")
        
        # Create unique connection ID
        connection_id = str(uuid.uuid4())
        
        # Register agent
        self._registered_agents[agent.agent_id] = connection_id
        self._connection_times[agent.agent_id] = time.time()
        
        return connection_id
    
    def deregister_agent(self, agent_id: str) -> bool:
        """
        Deregister an agent from the central post.
        
        Args:
            agent_id: ID of agent to deregister
            
        Returns:
            True if successfully deregistered, False if not found
        """
        if agent_id not in self._registered_agents:
            return False
        
        # Remove agent registration
        del self._registered_agents[agent_id]
        del self._connection_times[agent_id]
        
        return True
    
    def is_agent_registered(self, agent_id: str) -> bool:
        """
        Check if an agent is currently registered.
        
        Args:
            agent_id: ID of agent to check
            
        Returns:
            True if agent is registered, False otherwise
        """
        return agent_id in self._registered_agents
    
    def queue_message(self, message: Message) -> str:
        """
        Queue a message for processing.
        
        Args:
            message: Message to queue
            
        Returns:
            Message ID for tracking
        """
        if not self._is_active:
            raise RuntimeError("Central post is not active")
        
        # Validate sender is registered
        if message.sender_id != "central_post" and message.sender_id not in self._registered_agents:
            raise ValueError(f"Message from unregistered agent: {message.sender_id}")
        
        # Queue message
        self._message_queue.put(message)
        
        return message.message_id
    
    def has_pending_messages(self) -> bool:
        """
        Check if there are messages waiting to be processed.
        
        Returns:
            True if messages are pending, False otherwise
        """
        return not self._message_queue.empty()
    
    def process_next_message(self) -> Optional[Message]:
        """
        Process the next message in the queue (FIFO order).
        
        Returns:
            Processed message, or None if queue is empty
        """
        try:
            # Get next message
            start_time = time.time() if self._metrics_enabled else None
            message = self._message_queue.get_nowait()
            
            # Process message (placeholder - actual processing depends on message type)
            self._handle_message(message)
            
            # Record metrics
            if self._metrics_enabled and start_time:
                processing_time = time.time() - start_time
                self._processing_times.append(processing_time)
            
            # Track processed message
            self._processed_messages.append(message)
            self._total_messages_processed += 1
            
            return message
            
        except Empty:
            return None
    
    def _handle_message(self, message: Message) -> None:
        """
        Handle specific message types (internal processing).
        
        Args:
            message: Message to handle
        """
        # Message type-specific handling
        if message.message_type == MessageType.TASK_REQUEST:
            self._handle_task_request(message)
        elif message.message_type == MessageType.STATUS_UPDATE:
            self._handle_status_update(message)
        elif message.message_type == MessageType.TASK_COMPLETE:
            self._handle_task_completion(message)
        elif message.message_type == MessageType.ERROR_REPORT:
            self._handle_error_report(message)
        # Add more handlers as needed
    
    def _handle_task_request(self, message: Message) -> None:
        """Handle task request from agent."""
        # Placeholder for task assignment logic
        pass
    
    def _handle_status_update(self, message: Message) -> None:
        """Handle status update from agent."""
        # Placeholder for status tracking logic
        pass
    
    def _handle_task_completion(self, message: Message) -> None:
        """Handle task completion notification."""
        # Placeholder for completion processing logic
        pass
    
    def _handle_error_report(self, message: Message) -> None:
        """Handle error report from agent."""
        # Placeholder for error handling logic
        pass
    
    # Performance metrics methods (for Hypothesis H2)
    
    def get_current_time(self) -> float:
        """Get current timestamp for performance measurements."""
        return time.time()
    
    def get_message_throughput(self) -> float:
        """
        Calculate message processing throughput.
        
        Returns:
            Messages processed per second
        """
        if not self._metrics_enabled or self._total_messages_processed == 0:
            return 0.0
        
        elapsed_time = time.time() - self._start_time
        if elapsed_time == 0:
            return 0.0
        
        return self._total_messages_processed / elapsed_time
    
    def measure_communication_overhead(self, num_messages: int, processing_time: float) -> float:
        """
        Measure communication overhead vs processing time.
        
        Args:
            num_messages: Number of messages in the measurement
            processing_time: Actual processing time for comparison
            
        Returns:
            Communication overhead time
        """
        if not self._metrics_enabled:
            return 0.0
        
        # Simulate communication overhead calculation
        if self._processing_times:
            avg_msg_time = sum(self._processing_times) / len(self._processing_times)
            communication_overhead = avg_msg_time * num_messages
            return communication_overhead
        
        return 0.0
    
    def record_overhead_ratio(self, overhead_ratio: float) -> None:
        """
        Record overhead ratio for hypothesis validation.
        
        Args:
            overhead_ratio: Communication overhead / processing time ratio
        """
        if self._metrics_enabled:
            self._overhead_ratios.append(overhead_ratio)
    
    def get_average_overhead_ratio(self) -> float:
        """
        Get average overhead ratio across all measurements.
        
        Returns:
            Average overhead ratio
        """
        if not self._overhead_ratios:
            return 0.0
        
        return sum(self._overhead_ratios) / len(self._overhead_ratios)
    
    def record_scaling_metric(self, agent_count: int, processing_time: float) -> None:
        """
        Record scaling performance metric.
        
        Args:
            agent_count: Number of agents in the test
            processing_time: Time to process messages from all agents
        """
        if self._metrics_enabled:
            self._scaling_metrics[agent_count] = processing_time
    
    def get_scaling_metrics(self) -> Dict[int, float]:
        """
        Get scaling performance metrics.
        
        Returns:
            Dictionary mapping agent count to processing time
        """
        return self._scaling_metrics.copy()
    
    def shutdown(self) -> None:
        """Shutdown the central post and disconnect all agents."""
        self._is_active = False
        
        # Clear all connections
        self._registered_agents.clear()
        self._connection_times.clear()
        
        # Clear message queue
        while not self._message_queue.empty():
            try:
                self._message_queue.get_nowait()
            except Empty:
                break
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive performance summary for analysis.
        
        Returns:
            Dictionary containing all performance metrics
        """
        if not self._metrics_enabled:
            return {"metrics_enabled": False}
        
        return {
            "metrics_enabled": True,
            "total_messages_processed": self._total_messages_processed,
            "message_throughput": self.get_message_throughput(),
            "average_overhead_ratio": self.get_average_overhead_ratio(),
            "scaling_metrics": self.get_scaling_metrics(),
            "active_connections": self.active_connections,
            "uptime": time.time() - self._start_time
        }
    
    def accept_high_confidence_result(self, message: Message, min_confidence: float = 0.7) -> bool:
        """
        Accept agent results that meet minimum confidence threshold.
        
        This implements the natural selection aspect of the helix model -
        only high-quality results from the narrow bottom of the helix
        are accepted into the central coordination system.
        
        Args:
            message: Message containing agent result
            min_confidence: Minimum confidence threshold (0.0 to 1.0)
            
        Returns:
            True if result was accepted, False if rejected
        """
        if message.message_type != MessageType.STATUS_UPDATE:
            return False
        
        content = message.content
        confidence = content.get("confidence", 0.0)
        depth_ratio = content.get("position_info", {}).get("depth_ratio", 0.0)
        
        # Only accept results from agents deep in the helix with high confidence
        if depth_ratio >= 0.8 and confidence >= min_confidence:
            # Accept the result - add to processed messages
            self._processed_messages.append(message)
            self._total_messages_processed += 1
            return True
        else:
            # Reject the result
            return False