"""
LLM-powered agent for the Felix Framework.

This module extends the base Agent class with language model capabilities,
enabling agents to process tasks using local LLM inference via LM Studio.

Key Features:
- Integration with LM Studio for local LLM inference
- Position-aware prompt engineering based on helix location
- Adaptive behavior based on geometric constraints
- Communication via spoke system with central coordination
- Built-in task processing and result sharing

The agent's behavior adapts based on its position on the helix:
- Top (wide): Broad exploration, high creativity
- Middle: Focused analysis, balanced processing
- Bottom (narrow): Precise synthesis, low temperature
"""

import time
import asyncio
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

from agents.agent import Agent, AgentState
from core.helix_geometry import HelixGeometry
from llm.lm_studio_client import LMStudioClient, LLMResponse, RequestPriority
from llm.token_budget import TokenBudgetManager, TokenAllocation
from communication.central_post import Message, MessageType

logger = logging.getLogger(__name__)


@dataclass
class LLMTask:
    """Task for LLM agent processing."""
    task_id: str
    description: str
    context: str = ""
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class LLMResult:
    """Result from LLM agent processing."""
    agent_id: str
    task_id: str
    content: str
    position_info: Dict[str, float]
    llm_response: LLMResponse
    processing_time: float
    timestamp: float
    confidence: float = 0.0  # Confidence score (0.0 to 1.0)
    processing_stage: int = 1  # Stage number in helix descent


class LLMAgent(Agent):
    """
    LLM-powered agent that processes tasks using language models.
    
    Extends the base Agent class with LLM capabilities, providing
    position-aware prompt engineering and adaptive behavior based
    on the agent's location on the helix.
    """
    
    def __init__(self, agent_id: str, spawn_time: float, helix: HelixGeometry,
                 llm_client: LMStudioClient, agent_type: str = "general",
                 temperature_range: tuple = (0.1, 0.9), max_tokens: int = 500,
                 token_budget_manager: Optional[TokenBudgetManager] = None):
        """
        Initialize LLM agent.
        
        Args:
            agent_id: Unique identifier for the agent
            spawn_time: Time when agent becomes active (0.0 to 1.0)
            helix: Helix geometry for path calculation
            llm_client: LM Studio client for LLM inference
            agent_type: Agent specialization (research, analysis, synthesis, critic)
            temperature_range: (min, max) temperature based on helix position
            token_budget_manager: Optional budget manager for adaptive token allocation
        """
        super().__init__(agent_id, spawn_time, helix)
        
        self.llm_client = llm_client
        self.agent_type = agent_type
        self.temperature_range = temperature_range
        self.max_tokens = max_tokens
        self.token_budget_manager = token_budget_manager
        
        # Initialize token budget if manager provided
        if self.token_budget_manager:
            self.token_budget_manager.initialize_agent_budget(agent_id, agent_type, max_tokens)
        
        # LLM-specific state
        self.processing_results: List[LLMResult] = []
        self.total_tokens_used = 0
        self.total_processing_time = 0.0
        self.processing_stage = 0  # Current processing stage in helix descent
        
        # Communication state
        self.shared_context: Dict[str, Any] = {}
        self.received_messages: List[Dict[str, Any]] = []
        
        # Emergent behavior tracking
        self.influenced_by: List[str] = []  # Agent IDs that influenced this agent
        self.influence_strength: Dict[str, float] = {}  # How much each agent influenced this one
        self.collaboration_history: List[Dict[str, Any]] = []  # History of collaborations
    
    def get_position_info(self, current_time: float) -> Dict[str, float]:
        """
        Get detailed position information for the agent.
        
        Args:
            current_time: Current simulation time
            
        Returns:
            Dictionary with position details
        """
        position = self.get_position(current_time)
        if position is None:
            return {}
        
        x, y, z = position
        radius = self.helix.get_radius(z)
        depth_ratio = z / self.helix.height
        
        return {
            "x": x,
            "y": y, 
            "z": z,
            "radius": radius,
            "depth_ratio": depth_ratio,
            "progress": self._progress
        }
    
    def get_adaptive_temperature(self, current_time: float) -> float:
        """
        Calculate temperature based on helix position.
        
        Higher temperature (more creative) at top of helix,
        lower temperature (more focused) at bottom.
        
        Args:
            current_time: Current simulation time
            
        Returns:
            Temperature value for LLM
        """
        position_info = self.get_position_info(current_time)
        depth_ratio = position_info.get("depth_ratio", 0.0)
        
        # Invert depth ratio: top (0.0) = high temp, bottom (1.0) = low temp
        inverted_ratio = 1.0 - depth_ratio
        
        min_temp, max_temp = self.temperature_range
        temperature = min_temp + (max_temp - min_temp) * inverted_ratio
        
        return max(min_temp, min(max_temp, temperature))
    
    def calculate_confidence(self, current_time: float, content: str, stage: int) -> float:
        """
        Calculate confidence score based on agent type, helix position, and content quality.
        
        Agent types have different confidence ranges to ensure proper workflow:
        - Research agents: 0.3-0.6 (gather info, don't make final decisions)
        - Analysis agents: 0.4-0.7 (process info, prepare for synthesis)
        - Synthesis agents: 0.6-0.95 (create final output)
        - Critic agents: 0.5-0.8 (provide feedback)
        
        Args:
            current_time: Current simulation time
            content: Generated content to evaluate
            stage: Processing stage number
            
        Returns:
            Confidence score (0.0 to 1.0)
        """
        position_info = self.get_position_info(current_time)
        depth_ratio = position_info.get("depth_ratio", 0.0)
        
        # Base confidence range based on agent type
        if self.agent_type == "research":
            # Research agents max out at 0.6 - they gather info, don't make final decisions
            base_confidence = 0.3 + (depth_ratio * 0.3)  # 0.3-0.6 range
            max_confidence = 0.6
        elif self.agent_type == "analysis":
            # Analysis agents: 0.4-0.7 - process info but don't synthesize
            base_confidence = 0.4 + (depth_ratio * 0.3)  # 0.4-0.7 range
            max_confidence = 0.7
        elif self.agent_type == "synthesis":
            # Synthesis agents: 0.6-0.95 - create final comprehensive output
            base_confidence = 0.6 + (depth_ratio * 0.35)  # 0.6-0.95 range
            max_confidence = 0.95
        elif self.agent_type == "critic":
            # Critic agents: 0.5-0.8 - provide feedback and validation
            base_confidence = 0.5 + (depth_ratio * 0.3)  # 0.5-0.8 range
            max_confidence = 0.8
        else:
            # Default fallback
            base_confidence = 0.3 + (depth_ratio * 0.4)
            max_confidence = 0.7
        
        # Content quality bonus (up to 0.1 additional)
        content_quality = self._analyze_content_quality(content)
        content_bonus = content_quality * 0.1
        
        # Processing stage bonus (up to 0.05 additional)
        stage_bonus = min(stage * 0.005, 0.05)
        
        # Historical consistency bonus (up to 0.05 additional)
        consistency_bonus = self._calculate_consistency_bonus() * 0.05
        
        total_confidence = base_confidence + content_bonus + stage_bonus + consistency_bonus
        return min(max(total_confidence, 0.0), max_confidence)
    
    def _analyze_content_quality(self, content: str) -> float:
        """
        Analyze content quality using multiple heuristics.
        
        Args:
            content: Content to analyze
            
        Returns:
            Quality score (0.0 to 1.0)
        """
        if not content or len(content.strip()) == 0:
            return 0.0
        
        content_lower = content.lower()
        quality_score = 0.0
        
        # Length appropriateness (0.25 weight)
        content_length = len(content)
        if 100 <= content_length <= 2000:
            length_score = 1.0
        elif content_length < 100:
            length_score = content_length / 100.0
        else:  # Very long content
            length_score = max(0.3, 2000.0 / content_length)
        quality_score += length_score * 0.25
        
        # Structure indicators (0.25 weight)
        structure_indicators = [
            '\n\n' in content,  # Paragraphs
            '.' in content,     # Sentences
            any(word in content_lower for word in ['analysis', 'research', 'conclusion', 'summary']),
            content.count('.') >= 3,  # Multiple sentences
        ]
        structure_score = sum(structure_indicators) / len(structure_indicators)
        quality_score += structure_score * 0.25
        
        # Content depth indicators (0.25 weight)
        depth_indicators = [
            any(word in content_lower for word in ['because', 'therefore', 'however', 'moreover', 'furthermore']),
            any(word in content_lower for word in ['data', 'evidence', 'study', 'research', 'analysis']),
            any(word in content_lower for word in ['consider', 'suggest', 'indicate', 'demonstrate']),
            len(content.split()) > 50,  # Substantial word count
        ]
        depth_score = sum(depth_indicators) / len(depth_indicators)
        quality_score += depth_score * 0.25
        
        # Specificity indicators (0.25 weight)
        specificity_indicators = [
            any(char.isdigit() for char in content),  # Contains numbers/data
            content.count(',') > 2,  # Complex sentences with details
            any(word in content_lower for word in ['specific', 'particular', 'detail', 'example']),
            '"' in content or "'" in content,  # Quotes or citations
        ]
        specificity_score = sum(specificity_indicators) / len(specificity_indicators)
        quality_score += specificity_score * 0.25
        
        return min(quality_score, 1.0)
    
    def _calculate_consistency_bonus(self) -> float:
        """
        Calculate consistency bonus based on confidence history stability.
        
        Returns:
            Consistency bonus (0.0 to 1.0)
        """
        if len(self._confidence_history) < 3:
            return 0.5  # Neutral for insufficient data
        
        # Calculate confidence variance (lower variance = more consistent)
        recent_confidences = self._confidence_history[-3:]
        avg_confidence = sum(recent_confidences) / len(recent_confidences)
        variance = sum((c - avg_confidence) ** 2 for c in recent_confidences) / len(recent_confidences)
        
        # Convert variance to consistency bonus (lower variance = higher bonus)
        consistency_bonus = max(0.0, 1.0 - (variance * 10))  # Scale variance appropriately
        
        return min(consistency_bonus, 1.0)
    
    def create_position_aware_prompt(self, task: LLMTask, current_time: float) -> tuple[str, int]:
        """
        Create system prompt that adapts to agent's helix position with token budget.
        
        Args:
            task: Task to process
            current_time: Current simulation time
            
        Returns:
            Tuple of (position-aware system prompt, token budget for this stage)
        """
        position_info = self.get_position_info(current_time)
        depth_ratio = position_info.get("depth_ratio", 0.0)
        
        # Get token allocation if budget manager is available
        token_allocation = None
        stage_token_budget = self.max_tokens  # Default fallback
        
        if self.token_budget_manager:
            token_allocation = self.token_budget_manager.calculate_stage_allocation(
                self.agent_id, depth_ratio, self.processing_stage + 1
            )
            stage_token_budget = token_allocation.stage_budget
        
        # Add shared context from other agents
        context_summary = ""
        if self.shared_context:
            context_summary = "\n\nShared Context from Other Agents:\n"
            for key, value in self.shared_context.items():
                context_summary += f"- {key}: {value}\n"
        
        # Create base system prompt
        base_prompt = self.llm_client.create_agent_system_prompt(
            agent_type=self.agent_type,
            position_info=position_info,
            task_context=f"{task.context}{context_summary}"
        )
        
        # Add token budget guidance if available
        if token_allocation:
            budget_guidance = f"\n\nToken Budget Guidance:\n{token_allocation.style_guidance}"
            if token_allocation.compression_ratio > 0.5:
                budget_guidance += f"\nCompress previous insights by ~{token_allocation.compression_ratio:.0%} while preserving key points."
            
            enhanced_prompt = base_prompt + budget_guidance
        else:
            enhanced_prompt = base_prompt
        
        return enhanced_prompt, stage_token_budget
    
    async def process_task_with_llm_async(self, task: LLMTask, current_time: float, 
                                         priority: RequestPriority = RequestPriority.NORMAL) -> LLMResult:
        """
        Asynchronously process task using LLM with position-aware prompting.
        
        Args:
            task: Task to process
            current_time: Current simulation time
            priority: Request priority for LLM processing
            
        Returns:
            LLM processing result
        """
        start_time = time.perf_counter()
        
        # Get position-aware prompts, token budget, and temperature
        system_prompt, stage_token_budget = self.create_position_aware_prompt(task, current_time)
        temperature = self.get_adaptive_temperature(current_time)
        position_info = self.get_position_info(current_time)
        
        # Ensure stage budget doesn't exceed agent's max_tokens
        effective_token_budget = min(stage_token_budget, self.max_tokens)
        
        # Process with LLM using coordinated token budget (ASYNC)
        llm_response = await self.llm_client.complete_async(
            agent_id=self.agent_id,
            system_prompt=system_prompt,
            user_prompt=task.description,
            temperature=temperature,
            max_tokens=effective_token_budget,
            priority=priority
        )
        
        end_time = time.perf_counter()
        processing_time = end_time - start_time
        
        # Increment processing stage
        self.processing_stage += 1
        
        # Calculate confidence based on position and content
        confidence = self.calculate_confidence(current_time, llm_response.content, self.processing_stage)
        
        # Record confidence for adaptive progression
        self.record_confidence(confidence)
        
        # Create result
        result = LLMResult(
            agent_id=self.agent_id,
            task_id=task.task_id,
            content=llm_response.content,
            position_info=position_info,
            llm_response=llm_response,
            processing_time=processing_time,
            timestamp=time.time(),
            confidence=confidence,
            processing_stage=self.processing_stage
        )
        
        # Record token usage with budget manager
        if self.token_budget_manager:
            self.token_budget_manager.record_usage(self.agent_id, llm_response.tokens_used)
        
        # Update statistics
        self.processing_results.append(result)
        self.total_tokens_used += llm_response.tokens_used
        self.total_processing_time += processing_time
        
        logger.info(f"Agent {self.agent_id} processed task {task.task_id} "
                   f"at depth {position_info.get('depth_ratio', 0):.2f} "
                   f"in {processing_time:.2f}s (async)")
        
        return result
        
    def process_task_with_llm(self, task: LLMTask, current_time: float) -> LLMResult:
        """
        Process task using LLM with position-aware prompting (sync wrapper).
        
        Args:
            task: Task to process
            current_time: Current simulation time
            
        Returns:
            LLM processing result
        """
        start_time = time.perf_counter()
        
        # Get position-aware prompts, token budget, and temperature
        system_prompt, stage_token_budget = self.create_position_aware_prompt(task, current_time)
        temperature = self.get_adaptive_temperature(current_time)
        position_info = self.get_position_info(current_time)
        
        # Ensure stage budget doesn't exceed agent's max_tokens
        effective_token_budget = min(stage_token_budget, self.max_tokens)
        
        # Process with LLM using coordinated token budget (SYNC)
        llm_response = self.llm_client.complete(
            agent_id=self.agent_id,
            system_prompt=system_prompt,
            user_prompt=task.description,
            temperature=temperature,
            max_tokens=effective_token_budget
        )
        
        end_time = time.perf_counter()
        processing_time = end_time - start_time
        
        # Increment processing stage
        self.processing_stage += 1
        
        # Calculate confidence based on position and content
        confidence = self.calculate_confidence(current_time, llm_response.content, self.processing_stage)
        
        # Record confidence for adaptive progression
        self.record_confidence(confidence)
        
        # Create result
        result = LLMResult(
            agent_id=self.agent_id,
            task_id=task.task_id,
            content=llm_response.content,
            position_info=position_info,
            llm_response=llm_response,
            processing_time=processing_time,
            timestamp=time.time(),
            confidence=confidence,
            processing_stage=self.processing_stage
        )
        
        # Record token usage with budget manager
        if self.token_budget_manager:
            self.token_budget_manager.record_usage(self.agent_id, llm_response.tokens_used)
        
        # Update statistics
        self.processing_results.append(result)
        self.total_tokens_used += llm_response.tokens_used
        self.total_processing_time += processing_time
        
        logger.info(f"Agent {self.agent_id} processed task {task.task_id} "
                   f"at depth {position_info.get('depth_ratio', 0):.2f} "
                   f"in {processing_time:.2f}s")
        
        return result
    
    # Legacy method alias for backward compatibility
    async def process_task_async(self, task: LLMTask, current_time: float) -> LLMResult:
        """
        Asynchronously process task using LLM (legacy method).
        
        Args:
            task: Task to process
            current_time: Current simulation time
            
        Returns:
            LLM processing result
        """
        return await self.process_task_with_llm_async(task, current_time, RequestPriority.NORMAL)
    
    def share_result_to_central(self, result: LLMResult) -> Message:
        """
        Create message to share result with central post.
        
        Args:
            result: Processing result to share
            
        Returns:
            Message for central post communication
        """
        return Message(
            sender_id=self.agent_id,
            message_type=MessageType.STATUS_UPDATE,
            content={
                "type": "AGENT_RESULT",
                "agent_id": self.agent_id,
                "agent_type": self.agent_type,
                "task_id": result.task_id,
                "content": result.content,
                "position_info": result.position_info,
                "tokens_used": result.llm_response.tokens_used,
                "processing_time": result.processing_time,
                "confidence": result.confidence,
                "processing_stage": result.processing_stage,
                "summary": self._create_result_summary(result)
            },
            timestamp=result.timestamp
        )
    
    def _create_result_summary(self, result: LLMResult) -> str:
        """Create concise summary of processing result."""
        content_preview = result.content[:100] + "..." if len(result.content) > 100 else result.content
        depth = result.position_info.get("depth_ratio", 0.0)
        
        return f"[{self.agent_type.upper()} @ depth {depth:.2f}] {content_preview}"
    
    def receive_shared_context(self, message: Dict[str, Any]) -> None:
        """
        Receive and store shared context from other agents.
        
        Args:
            message: Shared context message
        """
        self.received_messages.append(message)
        
        # Extract relevant context
        if message.get("type") == "AGENT_RESULT":
            key = f"{message.get('agent_type', 'unknown')}_{message.get('agent_id', '')}"
            self.shared_context[key] = message.get("summary", "")
    
    def influence_agent_behavior(self, other_agent: "LLMAgent", influence_type: str, strength: float) -> None:
        """
        Influence another agent's behavior based on collaboration.
        
        Args:
            other_agent: Agent to influence
            influence_type: Type of influence ('accelerate', 'slow', 'pause', 'redirect')
            strength: Influence strength (0.0 to 1.0)
        """
        if strength <= 0.0 or other_agent.agent_id == self.agent_id:
            return  # No influence or self-influence
        
        # Record the influence relationship
        if other_agent.agent_id not in self.influence_strength:
            self.influence_strength[other_agent.agent_id] = 0.0
        self.influence_strength[other_agent.agent_id] += strength * 0.1  # Cumulative influence
        
        if self.agent_id not in other_agent.influenced_by:
            other_agent.influenced_by.append(self.agent_id)
        
        # Apply influence based on type and agent compatibility
        compatibility = self._calculate_agent_compatibility(other_agent)
        effective_strength = strength * compatibility
        
        if influence_type == "accelerate" and effective_strength > 0.3:
            # Speed up the other agent if they're compatible
            current_velocity = other_agent.velocity
            other_agent.set_velocity_multiplier(min(current_velocity * 1.2, 2.0))
        
        elif influence_type == "slow" and effective_strength > 0.4:
            # Slow down if there's strong incompatibility
            current_velocity = other_agent.velocity
            other_agent.set_velocity_multiplier(max(current_velocity * 0.8, 0.3))
        
        elif influence_type == "pause" and effective_strength > 0.6:
            # Pause for consideration of conflicting approaches
            other_agent.pause_for_duration(0.1 * effective_strength, 0.0)  # Brief pause
        
        # Record collaboration
        self.collaboration_history.append({
            "timestamp": time.time(),
            "other_agent": other_agent.agent_id,
            "influence_type": influence_type,
            "strength": strength,
            "effective_strength": effective_strength,
            "compatibility": compatibility
        })
    
    def _calculate_agent_compatibility(self, other_agent: "LLMAgent") -> float:
        """
        Calculate compatibility between this agent and another.
        
        Args:
            other_agent: Other agent to assess compatibility with
            
        Returns:
            Compatibility score (0.0 to 1.0)
        """
        # Type compatibility matrix
        type_compatibility = {
            ("research", "research"): 0.8,     # Research agents collaborate well
            ("research", "analysis"): 0.9,    # Research feeds analysis
            ("research", "synthesis"): 0.7,   # Research provides raw material
            ("research", "critic"): 0.6,      # Some tension but productive
            
            ("analysis", "analysis"): 0.7,    # Analysis agents can complement
            ("analysis", "synthesis"): 0.9,   # Analysis feeds synthesis
            ("analysis", "critic"): 0.8,      # Analysis benefits from critique
            
            ("synthesis", "synthesis"): 0.5,  # May compete for final output
            ("synthesis", "critic"): 0.8,     # Synthesis benefits from review
            
            ("critic", "critic"): 0.6,        # Critics can disagree
        }
        
        # Get base compatibility from types
        type_pair = (self.agent_type, other_agent.agent_type)
        reverse_type_pair = (other_agent.agent_type, self.agent_type)
        
        base_compatibility = type_compatibility.get(
            type_pair, type_compatibility.get(reverse_type_pair, 0.5)
        )
        
        # Modify based on confidence histories
        if (len(self._confidence_history) > 2 and 
            len(other_agent._confidence_history) > 2):
            
            my_trend = self._confidence_history[-1] - self._confidence_history[-2]
            their_trend = other_agent._confidence_history[-1] - other_agent._confidence_history[-2]
            
            # Agents with similar confidence trends are more compatible
            trend_similarity = 1.0 - abs(my_trend - their_trend)
            base_compatibility = (base_compatibility + trend_similarity) / 2
        
        return max(0.0, min(base_compatibility, 1.0))
    
    def assess_collaboration_opportunities(self, available_agents: List["LLMAgent"], 
                                         current_time: float) -> List[Dict[str, Any]]:
        """
        Assess opportunities for collaboration with other agents.
        
        Args:
            available_agents: List of other agents available for collaboration
            current_time: Current simulation time
            
        Returns:
            List of collaboration opportunities with recommendations
        """
        opportunities = []
        
        for other_agent in available_agents:
            if other_agent.agent_id == self.agent_id or other_agent.state != AgentState.ACTIVE:
                continue
            
            compatibility = self._calculate_agent_compatibility(other_agent)
            
            # Skip if compatibility is too low
            if compatibility < 0.3:
                continue
            
            # Assess potential collaboration based on current states
            opportunity = {
                "agent_id": other_agent.agent_id,
                "agent_type": other_agent.agent_type,
                "compatibility": compatibility,
                "recommended_influence": self._recommend_influence_type(other_agent),
                "confidence": other_agent._confidence_history[-1] if other_agent._confidence_history else 0.5,
                "distance": abs(self._progress - other_agent._progress)
            }
            
            opportunities.append(opportunity)
        
        # Sort by potential value (compatibility * confidence, adjusted for distance)
        opportunities.sort(key=lambda x: x["compatibility"] * x["confidence"] * (1.0 - x["distance"] * 0.5), reverse=True)
        
        return opportunities
    
    def _recommend_influence_type(self, other_agent: "LLMAgent") -> str:
        """
        Recommend type of influence to apply to another agent.
        
        Args:
            other_agent: Agent to recommend influence for
            
        Returns:
            Recommended influence type
        """
        if not other_agent._confidence_history:
            return "accelerate"  # Default to acceleration for new agents
        
        other_confidence = other_agent._confidence_history[-1]
        my_confidence = self._confidence_history[-1] if self._confidence_history else 0.5
        
        # If other agent has low confidence and I have high confidence, accelerate them
        if other_confidence < 0.5 and my_confidence > 0.7:
            return "accelerate"
        
        # If other agent has much higher confidence, slow down to learn from them
        elif other_confidence > my_confidence + 0.3:
            return "slow"
        
        # If confidence gap is large and we're incompatible, suggest pause
        elif abs(other_confidence - my_confidence) > 0.4:
            return "pause"
        
        # Default to acceleration for collaborative growth
        return "accelerate"
    
    def get_agent_stats(self) -> Dict[str, Any]:
        """
        Get agent performance statistics including emergent behavior metrics.
        
        Returns:
            Dictionary with performance metrics
        """
        stats = {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "state": self.state.value,
            "spawn_time": self.spawn_time,
            "progress": self._progress,
            "total_tasks_processed": len(self.processing_results),
            "total_tokens_used": self.total_tokens_used,
            "total_processing_time": self.total_processing_time,
            "average_processing_time": (self.total_processing_time / len(self.processing_results)
                                      if self.processing_results else 0.0),
            "messages_received": len(self.received_messages),
            "shared_context_items": len(self.shared_context),
            
            # Emergent behavior metrics
            "influenced_by_count": len(self.influenced_by),
            "influences_given": len(self.influence_strength),
            "total_influence_received": sum(self.influence_strength.values()),
            "collaboration_count": len(self.collaboration_history),
            "velocity": self.velocity,
            "confidence_history": self._confidence_history.copy(),
            "progression_info": self.get_progression_info()
        }
        
        # Add token budget information if available
        if self.token_budget_manager:
            budget_status = self.token_budget_manager.get_agent_status(self.agent_id)
            if budget_status:
                stats["token_budget"] = budget_status
        
        return stats


def create_llm_agents(helix: HelixGeometry, llm_client: LMStudioClient,
                     agent_configs: List[Dict[str, Any]]) -> List[LLMAgent]:
    """
    Create multiple LLM agents with specified configurations.
    
    Args:
        helix: Helix geometry for agent paths
        llm_client: LM Studio client for LLM inference
        agent_configs: List of agent configuration dictionaries
        
    Returns:
        List of configured LLM agents
    """
    agents = []
    
    for config in agent_configs:
        agent = LLMAgent(
            agent_id=config["agent_id"],
            spawn_time=config["spawn_time"],
            helix=helix,
            llm_client=llm_client,
            agent_type=config.get("agent_type", "general"),
            temperature_range=config.get("temperature_range", (0.1, 0.9))
        )
        agents.append(agent)
    
    return agents


def create_specialized_agent_configs(num_research: int = 3, num_analysis: int = 2,
                                   num_synthesis: int = 1, random_seed: int = 42069) -> List[Dict[str, Any]]:
    """
    Create agent configurations for typical Felix multi-agent task.
    
    Args:
        num_research: Number of research agents (spawn early)
        num_analysis: Number of analysis agents (spawn mid)
        num_synthesis: Number of synthesis agents (spawn late)
        random_seed: Random seed for spawn timing
        
    Returns:
        List of agent configuration dictionaries
    """
    import random
    random.seed(random_seed)
    
    configs = []
    agent_id = 0
    
    # Research agents - spawn early (0.0-0.4)
    for i in range(num_research):
        configs.append({
            "agent_id": f"research_{agent_id:03d}",
            "spawn_time": random.uniform(0.0, 0.4),
            "agent_type": "research",
            "temperature_range": (0.3, 0.9)
        })
        agent_id += 1
    
    # Analysis agents - spawn mid (0.3-0.7)
    for i in range(num_analysis):
        configs.append({
            "agent_id": f"analysis_{agent_id:03d}",
            "spawn_time": random.uniform(0.3, 0.7),
            "agent_type": "analysis", 
            "temperature_range": (0.2, 0.7)
        })
        agent_id += 1
    
    # Synthesis agents - spawn late (0.6-1.0)
    for i in range(num_synthesis):
        configs.append({
            "agent_id": f"synthesis_{agent_id:03d}",
            "spawn_time": random.uniform(0.6, 1.0),
            "agent_type": "synthesis",
            "temperature_range": (0.1, 0.5)
        })
        agent_id += 1
    
    return configs