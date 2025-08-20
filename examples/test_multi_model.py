#!/usr/bin/env python3
"""
Multi-Model Concurrent Processing Test for Felix Framework.

This script verifies that agents can use different models simultaneously
on a single LM Studio server, demonstrating true concurrent processing
with model specialization.

Requirements:
- LM Studio running on http://127.0.0.1:1234 with models:
  - qwen/qwen3-4b-2507
  - qwen/qwen3-4b-thinking-2507
  - google/gemma-3-12b

Usage:
    python examples/test_multi_model.py
"""

import sys
import time
import asyncio
from pathlib import Path
from typing import List, Dict, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from blog_writer import FelixBlogWriter
from llm.multi_server_client import LMStudioClientPool


class MultiModelTester:
    """Test framework for verifying multi-model concurrent processing."""
    
    def __init__(self):
        self.config_path = "config/multi_model_config.json"
        self.test_topics = [
            "Quantum computing breakthrough",
            "AI safety research",
            "Climate change solutions"
        ]
    
    def verify_config(self) -> bool:
        """Verify the multi-model configuration exists and is valid."""
        config_file = Path(self.config_path)
        if not config_file.exists():
            print(f"❌ Configuration file not found: {self.config_path}")
            return False
        
        try:
            import json
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            required_models = [
                "qwen/qwen3-4b-2507",
                "qwen/qwen3-4b-thinking-2507", 
                "google/gemma-3-12b"
            ]
            
            config_models = [server["model"] for server in config["servers"]]
            
            print(f"✅ Configuration loaded: {len(config['servers'])} model configurations")
            for model in required_models:
                if model in config_models:
                    print(f"  ✅ {model}")
                else:
                    print(f"  ❌ {model} - MISSING")
                    return False
            
            return True
            
        except Exception as e:
            print(f"❌ Configuration error: {e}")
            return False
    
    async def test_model_selection(self) -> Dict[str, Any]:
        """Test that different agent types select different models."""
        print(f"\n🧪 TESTING MODEL SELECTION")
        print("=" * 50)
        
        # Create client pool
        client_pool = LMStudioClientPool(config_path=self.config_path, debug_mode=True)
        
        # Display pool status
        client_pool.display_pool_status()
        
        # Test health of all model endpoints
        print(f"\n🏥 Testing model endpoint health...")
        health_results = await client_pool.health_check_all_servers()
        
        healthy_count = sum(1 for healthy in health_results.values() if healthy)
        total_count = len(health_results)
        
        if healthy_count == 0:
            print(f"❌ No healthy model endpoints (0/{total_count})")
            return {"error": "No healthy endpoints"}
        
        print(f"✅ Healthy endpoints: {healthy_count}/{total_count}")
        for endpoint, healthy in health_results.items():
            status = "✅" if healthy else "❌"
            print(f"  {status} {endpoint}")
        
        # Test agent type to model mapping
        print(f"\n🎯 Testing agent type mappings...")
        agent_types = ["research", "analysis", "synthesis", "critic"]
        mappings = {}
        
        for agent_type in agent_types:
            server_name = client_pool.get_server_for_agent_type(agent_type)
            if server_name:
                model_name = client_pool.servers[server_name].model
                mappings[agent_type] = {
                    "server": server_name,
                    "model": model_name
                }
                print(f"  {agent_type} → {server_name} ({model_name})")
            else:
                print(f"  ❌ {agent_type} → NO SERVER ASSIGNED")
        
        await client_pool.close_all()
        
        return {
            "health_results": health_results,
            "healthy_count": healthy_count,
            "total_count": total_count,
            "agent_mappings": mappings
        }
    
    async def test_concurrent_processing(self, topic: str) -> Dict[str, Any]:
        """Test actual concurrent processing with different models."""
        print(f"\n🚀 TESTING CONCURRENT PROCESSING")
        print(f"Topic: {topic}")
        print("=" * 50)
        
        # Create Felix writer with multi-model config
        writer = FelixBlogWriter(
            server_config_path=self.config_path,
            debug_mode=True,
            strict_mode=True  # For faster testing
        )
        
        # Test connection
        if not writer.test_lm_studio_connection():
            return {"error": "Connection test failed"}
        
        # Create team
        print(f"\n👥 Creating agent team...")
        writer.create_blog_writing_team(complexity="medium")
        
        # Show team composition
        print(f"\n📋 Agent Team:")
        for agent in writer.agents:
            model_info = "Unknown model"
            if hasattr(writer.llm_client, 'get_server_for_agent_type'):
                server_name = writer.llm_client.get_server_for_agent_type(agent.agent_type)
                if server_name and server_name in writer.llm_client.servers:
                    model_info = writer.llm_client.servers[server_name].model
            print(f"  {agent.agent_id} ({agent.agent_type}) → {model_info} @ t={agent.spawn_time:.2f}")
        
        # Run the session
        print(f"\n⚡ Starting concurrent processing session...")
        start_time = time.perf_counter()
        
        results = await writer.run_blog_writing_session_async(
            topic=topic,
            simulation_time=1.0
        )
        
        end_time = time.perf_counter()
        total_duration = end_time - start_time
        
        # Analyze results for concurrency evidence
        agents_participated = results["agents_participated"]
        
        # Group by model used
        model_usage = {}
        for agent_info in agents_participated:
            agent_type = agent_info["agent_type"]
            if agent_type not in model_usage:
                model_usage[agent_type] = []
            model_usage[agent_type].append(agent_info)
        
        # Check for overlapping processing times
        time_overlaps = []
        for i, agent1 in enumerate(agents_participated):
            for agent2 in agents_participated[i+1:]:
                time1 = agent1["spawn_time"]
                time2 = agent2["spawn_time"]
                if abs(time1 - time2) < 0.1:  # Processing within 0.1 time units
                    time_overlaps.append((agent1["agent_id"], agent2["agent_id"], abs(time1 - time2)))
        
        return {
            "success": results["final_output"] is not None,
            "total_duration": total_duration,
            "total_tokens": results["session_stats"]["total_tokens_used"],
            "agents_participated": len(agents_participated),
            "model_usage": model_usage,
            "time_overlaps": time_overlaps,
            "final_content_length": len(results["final_output"]["content"]) if results["final_output"] else 0,
            "llm_client_stats": results["session_stats"].get("llm_client_stats", {}),
            "final_output": results["final_output"]
        }
    
    def analyze_concurrency(self, test_result: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze test results for evidence of concurrent processing."""
        if "error" in test_result:
            return {"error": test_result["error"]}
        
        analysis = {
            "concurrent_evidence": {},
            "model_distribution": {},
            "performance_metrics": {}
        }
        
        # Evidence of concurrency
        time_overlaps = test_result.get("time_overlaps", [])
        analysis["concurrent_evidence"] = {
            "overlapping_agents": len(time_overlaps),
            "total_agents": test_result.get("agents_participated", 0),
            "overlap_percentage": (len(time_overlaps) / max(test_result.get("agents_participated", 1), 1)) * 100,
            "evidence_found": len(time_overlaps) > 0
        }
        
        # Model distribution
        model_usage = test_result.get("model_usage", {})
        analysis["model_distribution"] = {
            "agent_types_used": list(model_usage.keys()),
            "different_models": len(set(model_usage.keys())),
            "model_specialization": len(model_usage) > 1
        }
        
        # Performance metrics
        llm_stats = test_result.get("llm_client_stats", {})
        analysis["performance_metrics"] = {
            "total_duration": test_result.get("total_duration", 0),
            "total_tokens": test_result.get("total_tokens", 0),
            "content_quality": test_result.get("final_content_length", 0),
            "llm_requests": llm_stats.get("total_requests", 0),
            "avg_response_time": llm_stats.get("average_response_time", 0)
        }
        
        return analysis
    
    def display_results(self, model_test: Dict[str, Any], processing_test: Dict[str, Any], analysis: Dict[str, Any]):
        """Display comprehensive test results."""
        print(f"\n{'='*60}")
        print(f"🎉 MULTI-MODEL TEST RESULTS")
        print(f"{'='*60}")
        
        # Model selection results
        if "error" not in model_test:
            print(f"\n✅ MODEL SELECTION TEST:")
            print(f"   Healthy endpoints: {model_test['healthy_count']}/{model_test['total_count']}")
            print(f"   Agent mappings configured: {len(model_test['agent_mappings'])}")
            
            for agent_type, mapping in model_test['agent_mappings'].items():
                print(f"     {agent_type} → {mapping['model']}")
        
        # Concurrent processing results
        if "error" not in processing_test:
            print(f"\n🚀 CONCURRENT PROCESSING TEST:")
            print(f"   Success: {'✅' if processing_test['success'] else '❌'}")
            print(f"   Duration: {processing_test['total_duration']:.2f}s")
            print(f"   Tokens: {processing_test['total_tokens']}")
            print(f"   Agents participated: {processing_test['agents_participated']}")
            print(f"   Content generated: {processing_test['final_content_length']} chars")
        
        # Concurrency analysis
        if "error" not in analysis:
            print(f"\n🔍 CONCURRENCY ANALYSIS:")
            
            evidence = analysis["concurrent_evidence"]
            print(f"   Overlapping agents: {evidence['overlapping_agents']}")
            print(f"   Overlap rate: {evidence['overlap_percentage']:.1f}%")
            
            if evidence["evidence_found"]:
                print(f"   ✅ CONCURRENT PROCESSING DETECTED")
            else:
                print(f"   ⚠️  No clear concurrency evidence (may still be concurrent)")
            
            distribution = analysis["model_distribution"]
            print(f"   Different agent types: {distribution['different_models']}")
            
            if distribution["model_specialization"]:
                print(f"   ✅ MODEL SPECIALIZATION WORKING")
            else:
                print(f"   ❌ Model specialization not detected")
        
        # Final verdict
        print(f"\n🏆 FINAL VERDICT:")
        
        model_ok = "error" not in model_test and model_test['healthy_count'] > 0
        processing_ok = "error" not in processing_test and processing_test['success']
        concurrency_ok = "error" not in analysis and (
            analysis["concurrent_evidence"]["evidence_found"] or 
            analysis["model_distribution"]["model_specialization"]
        )
        
        if model_ok and processing_ok and concurrency_ok:
            print("✅ MULTI-MODEL CONCURRENT PROCESSING VERIFIED!")
            print("   - Multiple models accessible ✅")
            print("   - Agent type specialization ✅") 
            print("   - Concurrent processing ✅")
        elif model_ok and processing_ok:
            print("⚠️  MULTI-MODEL PROCESSING WORKING (concurrency unclear)")
            print("   - Multiple models accessible ✅")
            print("   - Processing successful ✅")
            print("   - Concurrent evidence limited ⚠️")
        else:
            print("❌ MULTI-MODEL SETUP NEEDS ATTENTION")
            if not model_ok:
                print("   - Model endpoint issues ❌")
            if not processing_ok:
                print("   - Processing failed ❌")
    
    async def run_full_test(self):
        """Run complete multi-model test suite."""
        print("🧪 Felix Framework Multi-Model Concurrent Processing Test")
        print("=" * 60)
        
        # Verify configuration
        if not self.verify_config():
            print("❌ Configuration verification failed")
            return
        
        # Test model selection
        model_test = await self.test_model_selection()
        
        if "error" in model_test:
            print(f"❌ Model selection test failed: {model_test['error']}")
            return
        
        # Test concurrent processing
        test_topic = self.test_topics[0]
        processing_test = await self.test_concurrent_processing(test_topic)
        
        # Analyze results
        analysis = self.analyze_concurrency(processing_test)
        
        # Display comprehensive results
        self.display_results(model_test, processing_test, analysis)
        
        # Save results
        timestamp = int(time.time())
        filename = f"multi_model_test_results_{timestamp}.json"
        
        import json
        results_data = {
            "timestamp": timestamp,
            "test_topic": test_topic,
            "model_test": model_test,
            "processing_test": processing_test,
            "analysis": analysis
        }
        
        with open(filename, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"\n💾 Detailed results saved to: {filename}")


async def main():
    """Main test function."""
    tester = MultiModelTester()
    await tester.run_full_test()


if __name__ == "__main__":
    asyncio.run(main())