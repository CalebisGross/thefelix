"""
Token Budget Management for Felix Framework.

This module implements adaptive token allocation based on agent position
in the helix, enabling efficient resource usage without truncating content.

Key Features:
- Position-based token budgets (more at top, less at bottom)
- Stage-aware allocation to prevent overconsumption
- Content compression guidance for iterative refinement
- Adaptive prompting based on helix depth

Mathematical Foundation:
- Token allocation follows inverse relationship with depth
- Budget decreases as agents focus toward synthesis
- Maintains quality while improving efficiency
"""

import math
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class TokenAllocation:
    """Token allocation details for an agent at a specific position."""
    stage_budget: int           # Tokens for current processing stage
    remaining_budget: int       # Total remaining tokens for agent
    total_budget: int          # Original total budget for agent
    compression_ratio: float   # Suggested compression from previous stages
    style_guidance: str        # Recommended output style
    depth_ratio: float         # Agent's current depth in helix


class TokenBudgetManager:
    """
    Manages token budgets for agents based on helix position and processing stage.
    
    Implements adaptive allocation strategy where agents receive more tokens
    for exploration at the top of the helix and fewer tokens for focused
    synthesis at the bottom.
    """
    
    def __init__(self, base_budget: int = 1000, min_budget: int = 200, 
                 max_budget: int = 800):
        """
        Initialize token budget manager.
        
        Args:
            base_budget: Base token allocation per agent
            min_budget: Minimum tokens for any single stage
            max_budget: Maximum tokens for any single stage
        """
        self.base_budget = base_budget
        self.min_budget = min_budget  
        self.max_budget = max_budget
        
        # Track agent budgets
        self._agent_budgets: Dict[str, int] = {}  # remaining budget per agent
        self._agent_total_used: Dict[str, int] = {}  # total used per agent
    
    def initialize_agent_budget(self, agent_id: str, agent_type: str) -> int:
        """
        Initialize total budget for an agent based on type.
        
        Args:
            agent_id: Agent identifier
            agent_type: Type of agent (research, analysis, synthesis, etc.)
            
        Returns:
            Total allocated budget for the agent
        """
        # Different agent types get different base budgets
        type_multipliers = {
            "research": 1.2,      # More tokens for exploration
            "analysis": 1.0,      # Standard allocation  
            "synthesis": 0.8,     # Fewer tokens for focused synthesis
            "critic": 0.9         # Slightly fewer for critique
        }
        
        multiplier = type_multipliers.get(agent_type, 1.0)
        total_budget = int(self.base_budget * multiplier)
        
        self._agent_budgets[agent_id] = total_budget
        self._agent_total_used[agent_id] = 0
        
        return total_budget
    
    def calculate_stage_allocation(self, agent_id: str, depth_ratio: float, 
                                 stage: int) -> TokenAllocation:
        """
        Calculate token allocation for a specific processing stage.
        
        Args:
            agent_id: Agent identifier
            depth_ratio: Current depth in helix (0.0 = top, 1.0 = bottom)
            stage: Current processing stage number (1-based)
            
        Returns:
            TokenAllocation with budget and guidance information
        """
        if agent_id not in self._agent_budgets:
            raise ValueError(f"Agent {agent_id} not initialized")
        
        remaining_budget = self._agent_budgets[agent_id]
        total_used = self._agent_total_used[agent_id]
        original_budget = remaining_budget + total_used
        
        # Calculate stage budget based on position and remaining allocation
        stage_budget = self._calculate_position_budget(
            depth_ratio, remaining_budget, stage
        )
        
        # Ensure budget constraints
        stage_budget = max(self.min_budget, min(stage_budget, self.max_budget))
        stage_budget = min(stage_budget, remaining_budget)  # Don't exceed remaining
        
        # Calculate compression and style guidance
        compression_ratio = self._calculate_compression_ratio(depth_ratio, stage)
        style_guidance = self._generate_style_guidance(depth_ratio, stage_budget)
        
        return TokenAllocation(
            stage_budget=stage_budget,
            remaining_budget=remaining_budget,
            total_budget=original_budget,
            compression_ratio=compression_ratio,
            style_guidance=style_guidance,
            depth_ratio=depth_ratio
        )
    
    def record_usage(self, agent_id: str, tokens_used: int) -> None:
        """
        Record token usage for an agent.
        
        Args:
            agent_id: Agent identifier
            tokens_used: Number of tokens consumed in last operation
        """
        if agent_id not in self._agent_budgets:
            raise ValueError(f"Agent {agent_id} not initialized")
        
        self._agent_budgets[agent_id] -= tokens_used
        self._agent_total_used[agent_id] += tokens_used
        
        # Ensure non-negative budget
        self._agent_budgets[agent_id] = max(0, self._agent_budgets[agent_id])
    
    def _calculate_position_budget(self, depth_ratio: float, remaining_budget: int,
                                 stage: int) -> int:
        """Calculate token budget based on helix position and stage."""
        # More tokens available at top (exploration), fewer at bottom (synthesis)
        # Use exponential decay to concentrate budget at top
        position_factor = math.exp(-depth_ratio * 2.0)  # Decays from 1.0 to ~0.14
        
        # Distribute remaining budget across estimated remaining stages
        # Agents typically need 2-5 more stages depending on depth
        estimated_remaining_stages = max(1, int((1.0 - depth_ratio) * 5) + 1)
        
        # Calculate base allocation with position weighting
        base_allocation = remaining_budget / estimated_remaining_stages
        stage_budget = int(base_allocation * (1.0 + position_factor))
        
        return stage_budget
    
    def _calculate_compression_ratio(self, depth_ratio: float, stage: int) -> float:
        """Calculate suggested compression ratio for content refinement."""
        # Higher compression as agents descend (more focused output)
        base_compression = 0.3 + (depth_ratio * 0.5)  # 0.3 to 0.8 range
        
        # Increase compression with processing stages
        stage_factor = min(stage * 0.05, 0.2)  # Up to 20% additional compression
        
        return min(base_compression + stage_factor, 0.9)
    
    def _generate_style_guidance(self, depth_ratio: float, token_budget: int) -> str:
        """Generate style guidance based on position and budget."""
        if depth_ratio < 0.3:
            style = "exploratory and comprehensive"
            detail_level = "detailed analysis with examples"
        elif depth_ratio < 0.7:
            style = "analytical and focused"
            detail_level = "clear conclusions with supporting evidence"
        else:
            style = "concise and decisive"
            detail_level = "key insights and actionable recommendations"
        
        return f"Provide {style} response with {detail_level} (~{token_budget} tokens)"
    
    def get_agent_status(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get current budget status for an agent."""
        if agent_id not in self._agent_budgets:
            return None
        
        remaining = self._agent_budgets[agent_id]
        used = self._agent_total_used[agent_id]
        total = remaining + used
        
        return {
            "agent_id": agent_id,
            "total_budget": total,
            "tokens_used": used,
            "tokens_remaining": remaining,
            "usage_ratio": used / total if total > 0 else 0.0
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system token usage status."""
        agent_statuses = [
            self.get_agent_status(agent_id) 
            for agent_id in self._agent_budgets.keys()
        ]
        
        total_allocated = sum(status["total_budget"] for status in agent_statuses)
        total_used = sum(status["tokens_used"] for status in agent_statuses)
        total_remaining = sum(status["tokens_remaining"] for status in agent_statuses)
        
        return {
            "total_agents": len(self._agent_budgets),
            "total_allocated": total_allocated,
            "total_used": total_used,
            "total_remaining": total_remaining,
            "system_efficiency": total_used / total_allocated if total_allocated > 0 else 0.0,
            "agent_statuses": agent_statuses
        }