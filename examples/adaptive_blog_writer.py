#!/usr/bin/env python3
"""
Adaptive Blog Writer Demo - Felix Framework Enhanced Edition

This demo showcases the fully enhanced Felix Framework with all five priority systems:
1. Intelligent Output Chunking & Streaming - Progressive content delivery
2. Dynamic Agent Spawning - Confidence-based team adaptation  
3. Prompt Optimization Pipeline - Learning from performance metrics
4. Memory & Persistence Layer - Cross-run knowledge accumulation
5. Comprehensive Benchmarking - Real-time quality and performance metrics

The demo creates an adaptive team that:
- Starts with minimal agents (2-3)
- Monitors confidence and spawns specialists as needed
- Chunks long outputs progressively  
- Learns from previous runs using persistent memory
- Optimizes prompts in real-time based on performance
- Shows comprehensive quality metrics and benchmarking

Usage:
    python examples/adaptive_blog_writer.py "Write about quantum computing"
    
    # Enable specific enhancements
    python examples/adaptive_blog_writer.py "Topic" --enable-dynamic-spawning
    python examples/adaptive_blog_writer.py "Topic" --enable-chunking
    python examples/adaptive_blog_writer.py "Topic" --enable-memory
    python examples/adaptive_blog_writer.py "Topic" --enable-optimization
    python examples/adaptive_blog_writer.py "Topic" --enable-benchmarking
    
    # Enable all enhancements (default)
    python examples/adaptive_blog_writer.py "Topic" --all-enhancements

Requirements:
    - LM Studio running with a model loaded (http://localhost:1234)
    - All Felix Framework enhancements properly integrated
"""

import sys
import time
import asyncio
import argparse
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.helix_geometry import HelixGeometry
from llm.lm_studio_client import LMStudioClient, LMStudioConnectionError
from llm.token_budget import TokenBudgetManager
from agents.llm_agent import LLMTask
from agents.specialized_agents import create_specialized_team
from communication.central_post import CentralPost, AgentFactory
from communication.spoke import SpokeManager

# Enhanced system imports
from agents.dynamic_spawning import DynamicSpawning
from agents.prompt_optimization import PromptOptimizer, PromptContext
from pipeline.chunking import ProgressiveProcessor, ContentSummarizer
from memory.knowledge_store import KnowledgeStore, KnowledgeType, ConfidenceLevel
from comparison.quality_metrics import QualityMetricsCalculator, DomainType
from comparison.performance_benchmarks import PerformanceBenchmarker


class AdaptiveFelixBlogWriter:
    """
    Enhanced blog writing system demonstrating all Felix Framework capabilities.
    
    Shows real-time adaptation, learning, and optimization across multiple runs.
    """
    
    def __init__(self, lm_studio_url: str = "http://localhost:1234/v1", 
                 random_seed: int = None, enable_all: bool = True,
                 enable_dynamic_spawning: bool = True, enable_chunking: bool = True,
                 enable_memory: bool = True, enable_optimization: bool = True,
                 enable_benchmarking: bool = True, debug_mode: bool = False):
        """
        Initialize the adaptive Felix blog writing system.
        
        Args:
            lm_studio_url: LM Studio API endpoint
            random_seed: Seed for deterministic behavior (None for random)
            enable_all: Enable all enhancements (overrides individual flags)
            enable_dynamic_spawning: Enable intelligent agent spawning
            enable_chunking: Enable progressive output chunking
            enable_memory: Enable persistent memory across runs
            enable_optimization: Enable prompt optimization
            enable_benchmarking: Enable comprehensive benchmarking
            debug_mode: Enable verbose debugging output
        """
        self.random_seed = random_seed
        self.debug_mode = debug_mode
        
        # Feature flags
        if enable_all:
            self.enable_dynamic_spawning = True
            self.enable_chunking = True
            self.enable_memory = True
            self.enable_optimization = True
            self.enable_benchmarking = True
        else:
            self.enable_dynamic_spawning = enable_dynamic_spawning
            self.enable_chunking = enable_chunking
            self.enable_memory = enable_memory
            self.enable_optimization = enable_optimization
            self.enable_benchmarking = enable_benchmarking
        
        print("🚀 Initializing Enhanced Felix Framework")
        print("=" * 60)
        
        # Create helix geometry
        self.helix = HelixGeometry(
            top_radius=33.0,
            bottom_radius=0.001,
            height=33.0,
            turns=33
        )
        
        # Initialize LLM client
        self.llm_client = LMStudioClient(
            base_url=lm_studio_url, 
            max_concurrent_requests=4,
            debug_mode=debug_mode
        )
        
        # Initialize enhanced systems
        self._initialize_enhanced_systems()
        
        # Initialize core Felix components
        self._initialize_core_components()
        
        # Agent team
        self.agents = []
        self.execution_stats = {
            "start_time": None,
            "end_time": None,
            "agents_spawned": 0,
            "dynamic_spawns": 0,
            "prompts_optimized": 0,
            "chunks_processed": 0,
            "knowledge_retrieved": 0,
            "quality_scores": []
        }
        
        print(f"✅ Enhanced Felix Framework initialized")
        print(f"   Features: {self._get_feature_summary()}")
        
    def _initialize_enhanced_systems(self):
        """Initialize all enhancement systems."""
        # 1. Prompt Optimization Pipeline
        if self.enable_optimization:
            self.prompt_optimizer = PromptOptimizer()
            print("📚 Prompt optimization enabled - learning from performance")
        else:
            self.prompt_optimizer = None
            
        # 2. Memory & Knowledge Store
        if self.enable_memory:
            self.knowledge_store = KnowledgeStore("adaptive_felix_knowledge.db")
            print("🧠 Persistent memory enabled - learning across runs")
        else:
            self.knowledge_store = None
            
        # 3. Quality Metrics & Benchmarking
        if self.enable_benchmarking:
            self.quality_calculator = QualityMetricsCalculator()
            self.benchmarker = PerformanceBenchmarker()
            print("📊 Comprehensive benchmarking enabled")
        else:
            self.quality_calculator = None
            self.benchmarker = None
            
        # 4. Content Summarizer for chunking
        if self.enable_chunking:
            self.content_summarizer = ContentSummarizer(self.llm_client)
            print("📝 Progressive chunking enabled - handling large outputs")
        else:
            self.content_summarizer = None
    
    def _initialize_core_components(self):
        """Initialize core Felix Framework components."""
        # Central communication system
        self.central_post = CentralPost(
            max_agents=20, 
            enable_metrics=self.enable_benchmarking,
            enable_memory=self.enable_memory
        )
        self.spoke_manager = SpokeManager(self.central_post)
        
        # Token budget manager - increased budget for synthesis agents
        self.token_budget_manager = TokenBudgetManager(
            base_budget=1500,
            min_budget=150,
            max_budget=1200,  # Increased for synthesis agents
            strict_mode=False
        )
        
        # Enhanced agent factory with dynamic spawning
        self.agent_factory = AgentFactory(
            helix=self.helix,
            llm_client=self.llm_client,
            token_budget_manager=self.token_budget_manager,
            random_seed=self.random_seed,
            enable_dynamic_spawning=self.enable_dynamic_spawning,
            max_agents=15,
            token_budget_limit=10000
        )
        
        print(f"🔧 Core components initialized")
        
    def _get_feature_summary(self) -> str:
        """Get summary of enabled features."""
        features = []
        if self.enable_dynamic_spawning:
            features.append("Dynamic Spawning")
        if self.enable_chunking:
            features.append("Smart Chunking")
        if self.enable_memory:
            features.append("Persistent Memory")
        if self.enable_optimization:
            features.append("Prompt Learning")
        if self.enable_benchmarking:
            features.append("Real-time Metrics")
        
        return ", ".join(features) if features else "Basic Mode"
    
    def test_lm_studio_connection(self) -> bool:
        """Test connection to LM Studio."""
        try:
            if self.llm_client.test_connection():
                print("✅ LM Studio connection successful")
                return True
            else:
                print("❌ LM Studio connection failed")
                return False
        except Exception as e:
            print(f"❌ LM Studio connection error: {e}")
            return False
    
    async def write_adaptive_blog_post(self, topic: str, save_output: bool = True) -> Dict[str, Any]:
        """
        Write a blog post using the adaptive Felix system.
        
        Args:
            topic: Blog post topic
            save_output: Whether to save output to file
            
        Returns:
            Dictionary with results and statistics
        """
        self.execution_stats["start_time"] = time.time()
        
        print("\n🎯 Starting Adaptive Blog Writing Process")
        print("=" * 60)
        print(f"Topic: {topic}")
        
        # Start benchmarking if enabled
        benchmark_context = None
        if self.enable_benchmarking:
            benchmark_context = self.benchmarker.benchmark_context(
                f"adaptive_blog_{int(time.time())}",
                team_size=3,  # Initial team size
                token_budget=10000
            )
            benchmark_context.__enter__()
        
        try:
            # Phase 1: Check memory for relevant knowledge
            relevant_knowledge = await self._retrieve_relevant_knowledge(topic)
            
            # Phase 2: Create initial team with enhanced agents
            initial_team = await self._create_initial_team(topic, relevant_knowledge)
            
            # Phase 3: Execute collaborative writing with real-time adaptation
            blog_content = await self._execute_collaborative_writing(
                topic, initial_team, relevant_knowledge
            )
            
            # Phase 4: Process and chunk output if needed
            processed_content = await self._process_final_output(blog_content, topic)
            
            # Phase 5: Set end time and store results
            self.execution_stats["end_time"] = time.time()
            
            # Store results and calculate quality metrics
            final_results = await self._finalize_and_analyze(
                topic, processed_content, initial_team
            )
            
            # Phase 6: Save output if requested
            if save_output:
                await self._save_output(topic, final_results)
            
            return final_results
            
        except Exception as e:
            print(f"❌ Error during adaptive blog writing: {e}")
            if self.debug_mode:
                import traceback
                traceback.print_exc()
            return {"error": str(e), "stats": self.execution_stats}
        finally:
            if benchmark_context:
                benchmark_context.__exit__(None, None, None)
    
    async def _retrieve_relevant_knowledge(self, topic: str) -> List[str]:
        """Retrieve relevant knowledge from previous runs."""
        if not self.enable_memory or not self.knowledge_store:
            return []
        
        print("🧠 Retrieving relevant knowledge from memory...")
        
        try:
            from memory.knowledge_store import KnowledgeQuery
            
            # Search for relevant knowledge
            query = KnowledgeQuery(
                knowledge_types=[KnowledgeType.TASK_RESULT],
                content_keywords=topic.lower().split(),
                min_confidence=ConfidenceLevel.MEDIUM,
                limit=5
            )
            
            entries = self.knowledge_store.retrieve_knowledge(query)
            knowledge_items = []
            
            for entry in entries:
                if "result" in entry.content:
                    knowledge_items.append(entry.content["result"])
                    self.execution_stats["knowledge_retrieved"] += 1
            
            if knowledge_items:
                print(f"   📖 Found {len(knowledge_items)} relevant knowledge items")
            else:
                print("   📝 No previous knowledge found - starting fresh")
                
            return knowledge_items
            
        except Exception as e:
            print(f"   ⚠️ Memory retrieval failed: {e}")
            return []
    
    async def _create_initial_team(self, topic: str, knowledge: List[str]) -> List:
        """Create initial team with enhanced agents."""
        print("👥 Creating initial agent team...")
        
        # Create enhanced agents with all systems integrated
        initial_agents = []
        
        # Research agent
        research_agent = self.agent_factory.create_research_agent(
            domain="general",
            spawn_time_range=(0.0, 0.1)
        )
        
        # Pass enhanced systems to agents
        if self.prompt_optimizer:
            research_agent.prompt_optimizer = self.prompt_optimizer
            
        initial_agents.append(research_agent)
        
        # Analysis agent 
        analysis_agent = self.agent_factory.create_analysis_agent(
            analysis_type="structured",
            spawn_time_range=(0.1, 0.3)
        )
        
        if self.prompt_optimizer:
            analysis_agent.prompt_optimizer = self.prompt_optimizer
            
        initial_agents.append(analysis_agent)
        
        # Synthesis agent
        synthesis_agent = self.agent_factory.create_synthesis_agent(
            output_format="blog_post",
            spawn_time_range=(0.6, 0.8)
        )
        
        if self.prompt_optimizer:
            synthesis_agent.prompt_optimizer = self.prompt_optimizer
            
        initial_agents.append(synthesis_agent)
        
        # Add knowledge to shared context if available
        if knowledge:
            shared_knowledge = "Previous relevant insights:\n" + "\n".join(f"- {k}" for k in knowledge[:3])
            for agent in initial_agents:
                agent.shared_context["previous_knowledge"] = shared_knowledge
        
        self.agents.extend(initial_agents)
        self.execution_stats["agents_spawned"] = len(initial_agents)
        
        print(f"   ✅ Initial team created: {len(initial_agents)} agents")
        return initial_agents
    
    async def _execute_collaborative_writing(self, topic: str, team: List, knowledge: List[str]) -> str:
        """Execute collaborative writing with real-time adaptation."""
        print("✍️ Executing collaborative writing with adaptation...")
        
        # Create main task
        task = LLMTask(
            task_id=f"blog_{int(time.time())}",
            description=f"Write a comprehensive blog post about: {topic}",
            context="Focus on accuracy, clarity, and engaging content.",
            metadata={"domain": "writing", "format": "blog_post"}
        )
        
        results = []
        current_time = 0.0
        time_step = 0.2
        
        # Process agents in helix order with dynamic spawning
        for step in range(5):  # 5 processing steps
            current_time += time_step
            
            print(f"   🔄 Processing step {step + 1}/5 (t={current_time:.1f})")
            
            # Process active agents
            for agent in self.agents:
                if agent.spawn_time <= current_time and agent.state.name != "COMPLETED":
                    try:
                        result = await agent.process_task_with_llm_async(task, current_time)
                        results.append(result)
                        
                        # Mark agent as completed after successful processing
                        # This prevents duplicate content generation
                        agent.state = agent.state.__class__.COMPLETED
                        
                        # Update benchmarking
                        if self.enable_benchmarking:
                            self.benchmarker.record_response_time(result.processing_time)
                            self.benchmarker.record_llm_call(
                                result.processing_time, 
                                getattr(result.llm_response, 'tokens_used', 0)
                            )
                        
                        print(f"      📝 {agent.agent_id}: confidence {result.confidence:.2f}")
                        
                    except Exception as e:
                        print(f"      ❌ {agent.agent_id} failed: {e}")
            
            # Check for dynamic spawning if enabled
            if self.enable_dynamic_spawning and len(results) > 0:
                new_agents = await self._check_dynamic_spawning(results, current_time)
                if new_agents:
                    self.agents.extend(new_agents)
                    self.execution_stats["dynamic_spawns"] += len(new_agents)
                    print(f"      🚀 Spawned {len(new_agents)} additional agents")
        
        # Combine results into final blog post
        blog_content = self._synthesize_results(results, topic)
        
        print("   ✅ Collaborative writing completed")
        return blog_content
    
    async def _check_dynamic_spawning(self, results: List, current_time: float) -> List:
        """Check if dynamic spawning should occur."""
        if not self.agent_factory.dynamic_spawner:
            return []
        
        # Convert results to messages for spawning analysis
        messages = []
        for result in results[-3:]:  # Last 3 results
            from communication.central_post import Message, MessageType
            
            message = Message(
                message_id=f"msg_{result.task_id}_{result.agent_id}",
                sender_id=result.agent_id,
                message_type=MessageType.TASK_COMPLETE,
                content={
                    "confidence": result.confidence,
                    "agent_type": getattr(result, 'agent_type', 'general'),
                    "position_info": result.position_info,
                    "result": result.content
                },
                timestamp=result.timestamp
            )
            messages.append(message)
        
        # Check spawning needs
        new_agents = self.agent_factory.dynamic_spawner.analyze_and_spawn(
            messages, self.agents, current_time
        )
        
        # Add prompt optimizer to new agents
        if self.prompt_optimizer:
            for agent in new_agents:
                agent.prompt_optimizer = self.prompt_optimizer
        
        return new_agents
    
    def _synthesize_results(self, results: List, topic: str) -> str:
        """Synthesize individual agent results into cohesive blog post."""
        if not results:
            return f"# {topic}\n\nNo content generated."
        
        # Group results by agent type and select the best content
        agent_content = {
            'research': [],
            'analysis': [],
            'synthesis': [],
            'critic': []
        }
        
        for result in results:
            # Try to get agent type from the result, or fall back to agent_id
            agent_type = getattr(result, 'agent_type', getattr(result, 'agent_id', 'general'))
            content = result.content.strip()
            confidence = getattr(result, 'confidence', 0.0)
            timestamp = getattr(result, 'timestamp', 0)
            
            # Categorize by agent type
            if 'research' in agent_type.lower():
                agent_content['research'].append((content, confidence, timestamp, result))
            elif 'analysis' in agent_type.lower():
                agent_content['analysis'].append((content, confidence, timestamp, result))
            elif 'synthesis' in agent_type.lower():
                agent_content['synthesis'].append((content, confidence, timestamp, result))
            elif 'critic' in agent_type.lower():
                agent_content['critic'].append((content, confidence, timestamp, result))
        
        # Select the best content for each type (highest confidence, then latest)
        def select_best_content(content_list):
            if not content_list:
                return []
            # Sort by confidence (desc), then timestamp (desc) to get the best and most recent
            sorted_content = sorted(content_list, key=lambda x: (x[1], x[2]), reverse=True)
            # Return only the best result for each agent type to avoid duplicates
            return [sorted_content[0][0]]
        
        research_content = select_best_content(agent_content['research'])
        analysis_content = select_best_content(agent_content['analysis'])
        synthesis_content = select_best_content(agent_content['synthesis'])
        critic_content = select_best_content(agent_content['critic'])
        
        # Build cohesive blog post
        blog_parts = [f"# {topic}\n"]
        
        if synthesis_content:
            # Use synthesis as main content (only the best one)
            blog_parts.extend(synthesis_content)
        else:
            # Fallback: combine research and analysis
            if research_content:
                blog_parts.append("## Research Findings\n")
                blog_parts.extend(research_content)
            
            if analysis_content:
                blog_parts.append("\n## Analysis\n")
                blog_parts.extend(analysis_content)
        
        if critic_content:
            blog_parts.append("\n## Additional Insights\n")
            blog_parts.extend(critic_content)
        
        return "\n\n".join(blog_parts)
    
    async def _process_final_output(self, content: str, topic: str) -> str:
        """Process final output with chunking if needed."""
        if not self.enable_chunking or len(content) < 2000:
            return content
        
        print("📝 Processing output with intelligent chunking...")
        
        try:
            # Create progressive processor for large content
            processor = ProgressiveProcessor(
                task_id=f"blog_output_{int(time.time())}",
                agent_id="adaptive_blog_writer",
                full_content=content,
                chunk_size=800
            )
            
            chunks = []
            current_token = None
            
            # Process all chunks
            while True:
                chunk = processor.get_next_chunk(current_token)
                if chunk is None:
                    break
                    
                chunks.append(chunk)
                current_token = chunk.continuation_token
                self.execution_stats["chunks_processed"] += 1
                
                if chunk.is_final:
                    break
            
            print(f"   📊 Content processed into {len(chunks)} chunks")
            
            # For demo, return full content (chunks could be streamed in real app)
            return content
            
        except Exception as e:
            print(f"   ⚠️ Chunking failed: {e}")
            return content
    
    async def _finalize_and_analyze(self, topic: str, content: str, team: List) -> Dict[str, Any]:
        """Finalize results and perform quality analysis."""
        print("📊 Analyzing results and updating knowledge...")
        
        results = {
            "topic": topic,
            "content": content,
            "stats": self.execution_stats.copy(),
            "team_composition": [agent.agent_id for agent in team],
            "quality_metrics": {},
            "performance_metrics": {},
            "learning_status": {}
        }
        
        # Calculate quality metrics if enabled
        if self.enable_benchmarking and self.quality_calculator:
            try:
                quality_score = self.quality_calculator.calculate_quality_score(
                    content, domain=DomainType.GENERAL
                )
                
                results["quality_metrics"] = {
                    "overall_score": quality_score.overall_score,
                    "coherence": quality_score.coherence_score,
                    "clarity": quality_score.clarity_score,
                    "completeness": quality_score.completeness_score,
                    "word_count": quality_score.word_count,
                    "sentence_count": quality_score.sentence_count
                }
                
                self.execution_stats["quality_scores"].append(quality_score.overall_score)
                print(f"   📈 Quality Score: {quality_score.overall_score:.3f}")
                
            except Exception as e:
                print(f"   ⚠️ Quality analysis failed: {e}")
        
        # Store knowledge for future runs if enabled
        if self.enable_memory and self.knowledge_store:
            try:
                avg_quality = sum(self.execution_stats["quality_scores"]) / max(len(self.execution_stats["quality_scores"]), 1)
                confidence_level = ConfidenceLevel.HIGH if avg_quality > 0.8 else ConfidenceLevel.MEDIUM
                
                knowledge_id = self.knowledge_store.store_knowledge(
                    knowledge_type=KnowledgeType.TASK_RESULT,
                    content={"result": content, "topic": topic, "quality": avg_quality},
                    confidence_level=confidence_level,
                    source_agent="adaptive_blog_writer",
                    domain="writing",
                    tags=["blog_post", topic.lower().replace(" ", "_")]
                )
                
                results["learning_status"]["knowledge_stored"] = knowledge_id
                print(f"   🧠 Knowledge stored for future runs: {knowledge_id}")
                
            except Exception as e:
                print(f"   ⚠️ Knowledge storage failed: {e}")
        
        # Get prompt optimization status if enabled
        if self.enable_optimization and self.prompt_optimizer:
            try:
                system_status = self.prompt_optimizer.get_system_status()
                results["learning_status"]["optimization"] = {
                    "prompts_tracked": system_status["total_prompts_tracked"],
                    "active_tests": system_status["active_tests"],
                    "system_health": system_status["system_health"],
                    "overall_performance": system_status["overall_performance"]
                }
                
                print(f"   📚 Prompt optimization: {system_status['system_health']} ({system_status['total_prompts_tracked']} prompts tracked)")
                
            except Exception as e:
                print(f"   ⚠️ Optimization status failed: {e}")
        
        # Get dynamic spawning summary if enabled
        if self.enable_dynamic_spawning and self.agent_factory.dynamic_spawner:
            try:
                spawning_summary = self.agent_factory.get_spawning_summary()
                results["learning_status"]["dynamic_spawning"] = spawning_summary
                
                print(f"   🚀 Dynamic spawning: {spawning_summary['total_spawns']} intelligent spawns")
                
            except Exception as e:
                print(f"   ⚠️ Spawning summary failed: {e}")
        
        return results
    
    async def _save_output(self, topic: str, results: Dict[str, Any]) -> str:
        """Save output to file with timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_topic = "".join(c for c in topic if c.isalnum() or c in (' ', '-', '_')).rstrip()[:50]
        filename = f"adaptive_blog_{safe_topic}_{timestamp}.json"
        filepath = Path("output") / filename
        
        # Create output directory if it doesn't exist
        filepath.parent.mkdir(exist_ok=True)
        
        # Save comprehensive results
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"💾 Results saved to: {filepath}")
        return str(filepath)
    
    def print_execution_summary(self, results: Dict[str, Any]):
        """Print a comprehensive execution summary."""
        print("\n" + "=" * 60)
        print("🎉 ADAPTIVE BLOG WRITING COMPLETED")
        print("=" * 60)
        
        stats = results.get("stats", {})
        end_time = stats.get("end_time", 0)
        start_time = stats.get("start_time", 0)
        execution_time = (end_time - start_time) if end_time and start_time else 0
        
        print(f"⏱️  Execution Time: {execution_time:.2f} seconds")
        print(f"👥 Team Composition: {len(results.get('team_composition', []))} agents")
        print(f"🚀 Dynamic Spawns: {stats.get('dynamic_spawns', 0)}")
        print(f"📝 Content Chunks: {stats.get('chunks_processed', 0)}")
        print(f"🧠 Knowledge Items: {stats.get('knowledge_retrieved', 0)}")
        
        # Quality metrics
        quality = results.get("quality_metrics", {})
        if quality:
            print(f"\n📊 QUALITY METRICS:")
            print(f"   Overall Score: {quality.get('overall_score', 0):.3f}")
            print(f"   Coherence: {quality.get('coherence', 0):.3f}")
            print(f"   Clarity: {quality.get('clarity', 0):.3f}")
            print(f"   Completeness: {quality.get('completeness', 0):.3f}")
            print(f"   Word Count: {quality.get('word_count', 0):,}")
        
        # Learning status
        learning = results.get("learning_status", {})
        if learning:
            print(f"\n🎓 LEARNING STATUS:")
            if "optimization" in learning:
                opt = learning["optimization"]
                print(f"   Prompt Optimization: {opt.get('system_health', 'unknown')} ({opt.get('prompts_tracked', 0)} prompts)")
            if "dynamic_spawning" in learning:
                spawn = learning["dynamic_spawning"]
                print(f"   Dynamic Spawning: {spawn.get('total_spawns', 0)} intelligent decisions")
            if "knowledge_stored" in learning:
                print(f"   Memory: Knowledge stored for future runs")
        
        print("=" * 60)


async def main():
    """Main function for the adaptive blog writer demo."""
    parser = argparse.ArgumentParser(description="Adaptive Felix Blog Writer Demo")
    parser.add_argument("topic", help="Blog post topic")
    parser.add_argument("--lm-studio-url", default="http://localhost:1234/v1", 
                       help="LM Studio API URL")
    parser.add_argument("--random-seed", type=int, help="Random seed for deterministic behavior")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--save-output", action="store_true", default=True, help="Save output to file")
    
    # Feature flags
    parser.add_argument("--enable-dynamic-spawning", action="store_true", default=True,
                       help="Enable dynamic agent spawning")
    parser.add_argument("--enable-chunking", action="store_true", default=True,
                       help="Enable progressive chunking")
    parser.add_argument("--enable-memory", action="store_true", default=True,
                       help="Enable persistent memory")
    parser.add_argument("--enable-optimization", action="store_true", default=True,
                       help="Enable prompt optimization")
    parser.add_argument("--enable-benchmarking", action="store_true", default=True,
                       help="Enable comprehensive benchmarking")
    parser.add_argument("--all-enhancements", action="store_true", default=True,
                       help="Enable all enhancements (default)")
    parser.add_argument("--basic-mode", action="store_true",
                       help="Run in basic mode without enhancements")
    
    args = parser.parse_args()
    
    # Handle basic mode
    if args.basic_mode:
        enable_all = False
        enable_individual = False
    else:
        enable_all = args.all_enhancements
        enable_individual = not enable_all
    
    # Create the adaptive blog writer
    writer = AdaptiveFelixBlogWriter(
        lm_studio_url=args.lm_studio_url,
        random_seed=args.random_seed,
        debug_mode=args.debug,
        enable_all=enable_all,
        enable_dynamic_spawning=enable_individual and args.enable_dynamic_spawning,
        enable_chunking=enable_individual and args.enable_chunking,
        enable_memory=enable_individual and args.enable_memory,
        enable_optimization=enable_individual and args.enable_optimization,
        enable_benchmarking=enable_individual and args.enable_benchmarking
    )
    
    # Test connection
    if not writer.test_lm_studio_connection():
        print("\n❌ Cannot connect to LM Studio. Please ensure:")
        print("   1. LM Studio is running")
        print("   2. A model is loaded")
        print("   3. Server is accessible at", args.lm_studio_url)
        return 1
    
    # Execute adaptive blog writing
    try:
        results = await writer.write_adaptive_blog_post(
            args.topic, 
            save_output=args.save_output
        )
        
        # Print comprehensive summary
        writer.print_execution_summary(results)
        
        # Print content preview
        content = results.get("content", "")
        if content:
            print(f"\n📖 CONTENT PREVIEW (first 500 chars):")
            print("-" * 60)
            print(content[:500] + ("..." if len(content) > 500 else ""))
            print("-" * 60)
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⏹️ Interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Execution failed: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    import asyncio
    exit_code = asyncio.run(main())
    exit(exit_code)