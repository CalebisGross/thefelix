"""
LM Studio client integration for Felix Framework.

This module provides a client interface to LM Studio's OpenAI-compatible API,
enabling LLM-powered agents in the Felix multi-agent system.

LM Studio runs a local server (typically http://localhost:1234/v1) that 
provides OpenAI-compatible API endpoints for local language model inference.
"""

import asyncio
import time
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from openai import OpenAI
import httpx

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Response from LLM completion."""
    content: str
    tokens_used: int
    response_time: float
    model: str
    temperature: float
    agent_id: str
    timestamp: float


class LMStudioConnectionError(Exception):
    """Raised when cannot connect to LM Studio."""
    pass


class LMStudioClient:
    """
    Client for communicating with LM Studio's local API server.
    
    Provides both synchronous and asynchronous methods for LLM completion,
    with built-in error handling, connection testing, and usage tracking.
    """
    
    def __init__(self, base_url: str = "http://localhost:1234/v1", 
                 timeout: float = 120.0):
        """
        Initialize LM Studio client.
        
        Args:
            base_url: LM Studio API endpoint
            timeout: Request timeout in seconds
        """
        self.base_url = base_url
        self.timeout = timeout
        self.client = OpenAI(
            base_url=base_url,
            api_key="not-needed",  # LM Studio doesn't require API keys
            timeout=timeout
        )
        
        # Usage tracking
        self.total_tokens = 0
        self.total_requests = 0
        self.total_response_time = 0.0
        
        # Connection state
        self._connection_verified = False
    
    def test_connection(self) -> bool:
        """
        Test connection to LM Studio server.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            response = httpx.get(f"{self.base_url.rstrip('/v1')}/health", 
                               timeout=5.0)
            self._connection_verified = response.status_code == 200
            return self._connection_verified
        except Exception as e:
            logger.warning(f"LM Studio connection test failed: {e}")
            return False
    
    def ensure_connection(self) -> None:
        """Ensure connection to LM Studio or raise exception."""
        if not self._connection_verified and not self.test_connection():
            raise LMStudioConnectionError(
                f"Cannot connect to LM Studio at {self.base_url}. "
                "Ensure LM Studio is running with a model loaded."
            )
    
    def complete(self, agent_id: str, system_prompt: str, user_prompt: str,
                 temperature: float = 0.7, max_tokens: Optional[int] = 500,
                 model: str = "local-model") -> LLMResponse:
        """
        Synchronous completion request to LM Studio.
        
        Args:
            agent_id: Identifier for the requesting agent
            system_prompt: System/context prompt
            user_prompt: User query/task
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens in response
            model: Model identifier (LM Studio will use loaded model)
            
        Returns:
            LLMResponse with content and metadata
            
        Raises:
            LMStudioConnectionError: If cannot connect to LM Studio
        """
        self.ensure_connection()
        
        start_time = time.perf_counter()
        
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            completion_args = {
                "model": model,
                "messages": messages,
                "temperature": temperature
            }
            
            if max_tokens:
                completion_args["max_tokens"] = max_tokens
            
            response = self.client.chat.completions.create(**completion_args)
            
            end_time = time.perf_counter()
            response_time = end_time - start_time
            
            # Extract response data
            content = response.choices[0].message.content
            tokens_used = response.usage.total_tokens if response.usage else 0
            
            # Update usage tracking
            self.total_tokens += tokens_used
            self.total_requests += 1
            self.total_response_time += response_time
            
            logger.debug(f"LLM completion for {agent_id}: {tokens_used} tokens, "
                        f"{response_time:.2f}s")
            
            return LLMResponse(
                content=content,
                tokens_used=tokens_used,
                response_time=response_time,
                model=model,
                temperature=temperature,
                agent_id=agent_id,
                timestamp=time.time()
            )
            
        except Exception as e:
            logger.error(f"LLM completion failed for {agent_id}: {e}")
            raise
    
    async def complete_async(self, agent_id: str, system_prompt: str, 
                           user_prompt: str, temperature: float = 0.7,
                           max_tokens: Optional[int] = None,
                           model: str = "local-model") -> LLMResponse:
        """
        Asynchronous completion request to LM Studio.
        
        Args:
            agent_id: Identifier for the requesting agent
            system_prompt: System/context prompt
            user_prompt: User query/task
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens in response
            model: Model identifier
            
        Returns:
            LLMResponse with content and metadata
        """
        # Run sync method in thread pool for now
        # TODO: Use true async client when available
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            self.complete,
            agent_id, system_prompt, user_prompt, temperature, max_tokens, model
        )
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """
        Get client usage statistics.
        
        Returns:
            Dictionary with usage metrics
        """
        avg_response_time = (self.total_response_time / self.total_requests 
                           if self.total_requests > 0 else 0.0)
        
        return {
            "total_requests": self.total_requests,
            "total_tokens": self.total_tokens,
            "total_response_time": self.total_response_time,
            "average_response_time": avg_response_time,
            "average_tokens_per_request": (self.total_tokens / self.total_requests
                                         if self.total_requests > 0 else 0.0),
            "connection_verified": self._connection_verified
        }
    
    def reset_stats(self) -> None:
        """Reset usage statistics."""
        self.total_tokens = 0
        self.total_requests = 0
        self.total_response_time = 0.0
    
    def create_agent_system_prompt(self, agent_type: str, position_info: Dict[str, float],
                                 task_context: str = "") -> str:
        """
        Create system prompt for Felix agent based on position and type.
        
        Args:
            agent_type: Type of agent (research, analysis, synthesis, critic)
            position_info: Agent's position on helix (x, y, z, radius, depth_ratio)
            task_context: Additional context about the current task
            
        Returns:
            Formatted system prompt
        """
        depth_ratio = position_info.get("depth_ratio", 0.0)
        radius = position_info.get("radius", 0.0)
        
        base_prompt = f"""You are a {agent_type} agent in the Felix multi-agent system.

Current Position:
- Depth: {depth_ratio:.2f} (0.0 = top/start, 1.0 = bottom/end)
- Radius: {radius:.2f} (decreasing as you progress)
- Processing Stage: {"Early/Broad" if depth_ratio < 0.3 else "Middle/Focused" if depth_ratio < 0.7 else "Final/Precise"}

Your Role Based on Position:
"""
        
        if agent_type == "research":
            if depth_ratio < 0.3:
                base_prompt += "- Conduct broad exploration and gather diverse information\n"
                base_prompt += "- Cast a wide net and explore multiple angles\n"
                base_prompt += "- Don't worry about precision yet - focus on coverage\n"
            else:
                base_prompt += "- Refine your research based on earlier findings\n"
                base_prompt += "- Focus on specific aspects that seem most relevant\n"
                base_prompt += "- Prepare findings for analysis agents\n"
        
        elif agent_type == "analysis":
            base_prompt += "- Process and organize information from research agents\n"
            base_prompt += "- Identify patterns, contradictions, and key insights\n"
            base_prompt += "- Structure findings for synthesis agents\n"
            
        elif agent_type == "synthesis":
            base_prompt += "- Integrate all previous work into coherent output\n"
            base_prompt += "- Make final decisions and create deliverables\n"
            base_prompt += "- Focus on quality and completeness\n"
            
        elif agent_type == "critic":
            base_prompt += "- Review and critique work from other agents\n"
            base_prompt += "- Identify gaps, errors, and improvement opportunities\n"
            base_prompt += "- Provide quality assurance\n"
        
        if task_context:
            base_prompt += f"\nTask Context: {task_context}\n"
        
        base_prompt += "\nRemember: Your behavior should adapt to your position on the helix. "
        base_prompt += "Early positions focus on breadth, later positions focus on depth and precision."
        
        return base_prompt


def create_default_client() -> LMStudioClient:
    """Create LM Studio client with default settings."""
    return LMStudioClient()