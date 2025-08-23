#!/usr/bin/env python3
"""
Code Reviewer Demo using Felix Framework Geometric Orchestration.

This demo showcases how the Felix Framework can be used for collaborative
code review, with multiple specialized agents examining code from different
perspectives and converging naturally toward a comprehensive review.

The demo demonstrates the geometric attention focusing mechanism - early
agents perform broad analysis while later agents focus on specific issues
and provide final synthesis.

Usage:
    python examples/code_reviewer.py path/to/code.py
    python examples/code_reviewer.py --code-string "def hello(): print('hello')"

Requirements:
    - LM Studio running with a model loaded (http://localhost:1234)
    - openai Python package installed
"""

import sys
import time
import argparse
from typing import List, Dict, Any, Optional
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.helix_geometry import HelixGeometry
from src.llm.lm_studio_client import LMStudioClient, LMStudioConnectionError
from src.agents.llm_agent import LLMTask
from src.agents.specialized_agents import (
    ResearchAgent, AnalysisAgent, SynthesisAgent, CriticAgent
)
from src.communication.central_post import CentralPost
from src.communication.spoke import SpokeManager


class FelixCodeReviewer:
    """
    Code review system using Felix geometric orchestration.
    
    Demonstrates how helix-based agent coordination can provide
    comprehensive code review through natural convergence of
    different analytical perspectives.
    """
    
    def __init__(self, lm_studio_url: str = "http://localhost:1234/v1"):
        """Initialize the Felix code review system."""
        # Create helix geometry
        self.helix = HelixGeometry(
            top_radius=33.0,
            bottom_radius=0.001,
            height=33.0,
            turns=33
        )
        
        # Initialize LLM client
        self.llm_client = LMStudioClient(base_url=lm_studio_url)
        
        # Initialize communication system
        self.central_post = CentralPost(max_agents=15, enable_metrics=True)
        self.spoke_manager = SpokeManager(self.central_post)
        
        # Review agents
        self.agents = []
        
        print(f"Felix Code Reviewer initialized")
    
    def test_connection(self) -> bool:
        """Test connection to LM Studio."""
        try:
            if self.llm_client.test_connection():
                print("✓ LM Studio connection successful")
                return True
            else:
                print("✗ LM Studio connection failed")
                return False
        except LMStudioConnectionError as e:
            print(f"✗ LM Studio connection error: {e}")
            return False
    
    def create_code_review_team(self) -> None:
        """Create specialized team for code review."""
        print(f"\nCreating code review team with geometric specialization...")
        
        # Research agents - broad analysis (spawn early, top of helix)
        self.agents.extend([
            ResearchAgent("code_analysis_001", 0.05, self.helix, self.llm_client, "code_structure"),
            ResearchAgent("code_analysis_002", 0.10, self.helix, self.llm_client, "functionality"), 
            ResearchAgent("code_analysis_003", 0.15, self.helix, self.llm_client, "style"),
        ])
        
        # Analysis agents - focused evaluation (spawn mid, middle of helix)
        self.agents.extend([
            AnalysisAgent("detailed_review_001", 0.35, self.helix, self.llm_client, "performance"),
            AnalysisAgent("detailed_review_002", 0.45, self.helix, self.llm_client, "security"),
            AnalysisAgent("detailed_review_003", 0.55, self.helix, self.llm_client, "maintainability"),
        ])
        
        # Critic agents - quality assurance (spawn late-mid, narrowing helix)
        self.agents.extend([
            CriticAgent("quality_check_001", 0.65, self.helix, self.llm_client, "bug_detection"),
            CriticAgent("quality_check_002", 0.75, self.helix, self.llm_client, "best_practices"),
        ])
        
        # Synthesis agent - final review (spawn latest, bottom of helix)
        self.agents.append(
            SynthesisAgent("final_review_001", 0.85, self.helix, self.llm_client, "code_review_report")
        )
        
        # Register all agents
        for agent in self.agents:
            self.spoke_manager.register_agent(agent)
        
        print(f"Created review team of {len(self.agents)} specialized agents:")
        for agent in self.agents:
            depth_at_spawn = agent.spawn_time  # Approximation
            print(f"  - {agent.agent_id}: {agent.agent_type} @ t={agent.spawn_time:.2f} "
                  f"(depth ≈ {depth_at_spawn:.2f})")
    
    def load_code_to_review(self, code_path: Optional[str] = None, 
                          code_string: Optional[str] = None) -> str:
        """Load code for review from file or string."""
        if code_string:
            return code_string
        elif code_path:
            try:
                with open(code_path, 'r') as f:
                    return f.read()
            except Exception as e:
                raise ValueError(f"Could not read code file {code_path}: {e}")
        else:
            raise ValueError("Must provide either code_path or code_string")
    
    def run_code_review_session(self, code: str, filename: str = "code.py") -> Dict[str, Any]:
        """
        Run collaborative code review session.
        
        Args:
            code: Code to review
            filename: Filename for context
            
        Returns:
            Comprehensive review results
        """
        print(f"\n{'='*60}")
        print(f"FELIX CODE REVIEW SESSION")
        print(f"File: {filename} ({len(code)} characters)")
        print(f"{'='*60}")
        
        # Preview code
        lines = code.split('\n')
        print(f"\nCode Preview (first 10 lines):")
        for i, line in enumerate(lines[:10], 1):
            print(f"{i:2d}: {line}")
        if len(lines) > 10:
            print(f"    ... and {len(lines) - 10} more lines")
        
        # Create review task
        review_task = LLMTask(
            task_id="code_review_001",
            description=f"Perform a thorough code review of the following code:\n\n```python\n{code}\n```",
            context=f"This is a collaborative code review. Multiple agents will examine "
                   f"different aspects (structure, functionality, style, performance, security, etc.) "
                   f"and converge toward a comprehensive review. File: {filename}",
            metadata={"filename": filename, "code_length": len(code), "line_count": len(lines)}
        )
        
        # Track review results
        results = {
            "filename": filename,
            "code": code,
            "review_participants": [],
            "review_timeline": [],
            "convergence_pattern": [],
            "final_review": None,
            "session_stats": {}
        }
        
        # Run review simulation
        current_time = 0.0
        time_step = 0.05
        simulation_time = 1.0
        session_start = time.perf_counter()
        
        print(f"\nStarting geometric code review orchestration...")
        print(f"Agents will spawn and converge from broad analysis → focused critique → final synthesis")
        
        while current_time <= simulation_time:
            # Check for agents ready to spawn
            for agent in self.agents:
                if (agent.can_spawn(current_time) and 
                    agent.state.value == "waiting"):
                    
                    # Calculate position info for visual feedback
                    pos_info = agent.get_position_info(current_time + 0.01)  # Slight offset for spawning
                    depth = pos_info.get("depth_ratio", 0.0)
                    radius = pos_info.get("radius", 0.0)
                    
                    print(f"\n[t={current_time:.2f}] 🔍 {agent.agent_id} ({agent.agent_type}) spawning")
                    print(f"    Helix position: depth={depth:.2f}, radius={radius:.1f}")
                    print(f"    Focus level: {'Broad exploration' if depth < 0.3 else 'Focused analysis' if depth < 0.7 else 'Precise critique'}")
                    
                    # Spawn and process
                    try:
                        agent.spawn(current_time, review_task)
                        result = agent.process_task_with_llm(review_task, current_time)
                        
                        # Share result with central post via spoke communication
                        message = agent.share_result_to_central(result)
                        self.spoke_manager.send_message(agent.agent_id, message)
                        # Central post will handle distribution through spoke system
                        
                        # Track participation
                        participant_info = {
                            "agent_id": agent.agent_id,
                            "agent_type": agent.agent_type,
                            "spawn_time": current_time,
                            "position_info": result.position_info,
                            "review_focus": getattr(agent, 'research_domain', None) or 
                                          getattr(agent, 'analysis_type', None) or
                                          getattr(agent, 'review_focus', None) or
                                          getattr(agent, 'output_format', None),
                            "content_length": len(result.content),
                            "tokens_used": result.llm_response.tokens_used,
                            "processing_time": result.processing_time,
                            "review_preview": result.content[:150] + "..."
                        }
                        
                        results["review_participants"].append(participant_info)
                        
                        # Track convergence pattern
                        results["convergence_pattern"].append({
                            "timestamp": current_time,
                            "agent_id": agent.agent_id,
                            "depth_ratio": result.position_info.get("depth_ratio", 0.0),
                            "radius": result.position_info.get("radius", 0.0),
                            "content_focus": self._analyze_content_focus(result.content)
                        })
                        
                        print(f"    ✓ Review completed ({result.llm_response.tokens_used} tokens, "
                              f"{result.processing_time:.2f}s)")
                        print(f"    Preview: {result.content[:100]}...")
                        
                        # Check for final synthesis
                        if agent.agent_type == "synthesis":
                            if hasattr(agent, 'finalize_output'):
                                final_review = agent.finalize_output(result)
                                results["final_review"] = final_review
                                print(f"    🎯 Final review synthesis completed!")
                        
                    except Exception as e:
                        print(f"    ✗ Error during review: {e}")
            
            # Update positions and process communication
            for agent in self.agents:
                if agent.state.value == "active":
                    agent.update_position(current_time)
            
            self.spoke_manager.process_all_messages()
            current_time += time_step
        
        session_end = time.perf_counter()
        
        # Collect session statistics
        results["session_stats"] = {
            "total_duration": session_end - session_start,
            "agents_participated": len(results["review_participants"]),
            "total_tokens_used": sum(p["tokens_used"] for p in results["review_participants"]),
            "total_messages": self.central_post.total_messages_processed,
            "llm_client_stats": self.llm_client.get_usage_stats(),
            "convergence_stages": len(set(p["agent_type"] for p in results["review_participants"]))
        }
        
        return results
    
    def _analyze_content_focus(self, content: str) -> str:
        """Analyze what aspect of code the review content focuses on."""
        content_lower = content.lower()
        
        focus_keywords = {
            "structure": ["class", "function", "method", "organization", "architecture"],
            "style": ["naming", "format", "convention", "readability", "style"],
            "performance": ["efficiency", "performance", "optimization", "speed", "memory"],
            "security": ["security", "vulnerability", "injection", "validation", "sanitize"],
            "bugs": ["bug", "error", "exception", "issue", "problem", "fix"],
            "maintainability": ["maintainability", "documentation", "testing", "modularity"]
        }
        
        focus_scores = {}
        for focus, keywords in focus_keywords.items():
            score = sum(1 for keyword in keywords if keyword in content_lower)
            if score > 0:
                focus_scores[focus] = score
        
        if focus_scores:
            return max(focus_scores, key=focus_scores.get)
        else:
            return "general"
    
    def display_review_results(self, results: Dict[str, Any]) -> None:
        """Display code review results."""
        print(f"\n{'='*60}")
        print(f"CODE REVIEW RESULTS")
        print(f"{'='*60}")
        
        stats = results["session_stats"]
        print(f"File: {results['filename']}")
        print(f"Review Duration: {stats['total_duration']:.2f} seconds")
        print(f"Participants: {stats['agents_participated']} agents")
        print(f"Total Tokens: {stats['total_tokens_used']}")
        print(f"Convergence Stages: {stats['convergence_stages']}")
        
        print(f"\nReview Progression (Geometric Convergence):")
        for participant in results["review_participants"]:
            depth = participant["position_info"].get("depth_ratio", 0.0)
            focus = participant["review_focus"] or "general"
            print(f"  {participant['spawn_time']:.2f}s: {participant['agent_id']} "
                  f"(depth={depth:.2f}, focus={focus})")
            print(f"        {participant['review_preview']}")
        
        if results["final_review"]:
            print(f"\n{'='*60}")
            print(f"FINAL COMPREHENSIVE REVIEW")
            print(f"{'='*60}")
            print(results["final_review"]["content"])
            
            metadata = results["final_review"]["metadata"]
            print(f"\n[Review completed by {metadata['agent_id']} "
                  f"using {metadata['tokens_used']} tokens at "
                  f"helix depth {metadata['position_info'].get('depth_ratio', 0):.2f}]")
        else:
            print(f"\n⚠️  No final synthesis was generated")
        
        # Show convergence pattern
        print(f"\nGeometric Convergence Pattern:")
        focus_by_depth = {}
        for entry in results["convergence_pattern"]:
            depth_bucket = round(entry["depth_ratio"], 1)
            if depth_bucket not in focus_by_depth:
                focus_by_depth[depth_bucket] = []
            focus_by_depth[depth_bucket].append(entry["content_focus"])
        
        for depth in sorted(focus_by_depth.keys()):
            focuses = focus_by_depth[depth]
            print(f"  Depth {depth:.1f}: {', '.join(set(focuses))}")


def main():
    """Main function for code reviewer demo."""
    parser = argparse.ArgumentParser(description="Felix Framework Code Reviewer Demo")
    parser.add_argument("code_file", nargs="?", help="Python file to review")
    parser.add_argument("--code-string", help="Code string to review")
    parser.add_argument("--lm-studio-url", default="http://localhost:1234/v1",
                       help="LM Studio API URL")
    
    args = parser.parse_args()
    
    if not args.code_file and not args.code_string:
        parser.error("Must provide either a code file or --code-string")
    
    # Create code reviewer
    reviewer = FelixCodeReviewer(lm_studio_url=args.lm_studio_url)
    
    # Test connection
    if not reviewer.test_connection():
        print("\nPlease ensure LM Studio is running with a model loaded.")
        sys.exit(1)
    
    # Load code
    try:
        code = reviewer.load_code_to_review(
            code_path=args.code_file,
            code_string=args.code_string
        )
        filename = args.code_file or "provided_code.py"
    except ValueError as e:
        print(f"Error loading code: {e}")
        sys.exit(1)
    
    # Create review team and run session
    reviewer.create_code_review_team()
    results = reviewer.run_code_review_session(code, filename)
    
    # Display results
    reviewer.display_review_results(results)


if __name__ == "__main__":
    main()