#!/usr/bin/env python3
"""
Multi-Server Performance Comparison Test for Felix Framework.

This script compares performance between single-server and multi-server
configurations to demonstrate true parallel processing benefits.

Requirements:
- Multiple LM Studio servers running on different ports
- Configuration files in config/ directory

Usage:
    python examples/test_multi_server_performance.py
"""

import sys
import time
import asyncio
import statistics
from pathlib import Path
from typing import List, Dict, Any, Tuple

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from examples.blog_writer import FelixBlogWriter


class PerformanceTest:
    """Test framework for comparing single vs multi-server performance."""
    
    def __init__(self, debug_mode: bool = False):
        self.debug_mode = debug_mode
        self.test_topics = [
            "Quantum computing applications",
            "Sustainable energy solutions", 
            "Artificial intelligence ethics",
            "Space exploration technology",
            "Blockchain and cryptocurrency"
        ]
    
    async def run_single_test(self, topic: str, config_path: str, test_name: str) -> Dict[str, Any]:
        """
        Run a single performance test.
        
        Args:
            topic: Blog post topic
            config_path: Server configuration path
            test_name: Name for this test
            
        Returns:
            Test results dictionary
        """
        print(f"\n{'='*60}")
        print(f"🧪 RUNNING TEST: {test_name}")
        print(f"Topic: {topic}")
        print(f"Config: {config_path}")
        print(f"{'='*60}")
        
        # Create writer with specified configuration
        writer = FelixBlogWriter(
            server_config_path=config_path,
            strict_mode=True,  # Use strict mode for consistent testing
            debug_mode=self.debug_mode
        )
        
        # Test connection
        if not writer.test_lm_studio_connection():
            return {
                "error": "Connection failed",
                "test_name": test_name,
                "topic": topic,
                "config": config_path
            }
        
        # Create team
        writer.create_blog_writing_team(complexity="medium")
        
        # Run test
        start_time = time.perf_counter()
        results = await writer.run_blog_writing_session_async(
            topic=topic,
            simulation_time=1.0
        )
        end_time = time.perf_counter()
        
        total_duration = end_time - start_time
        
        # Extract metrics
        stats = results["session_stats"]
        
        test_result = {
            "test_name": test_name,
            "topic": topic,
            "config": config_path,
            "success": results["final_output"] is not None,
            "total_duration": total_duration,
            "simulation_duration": stats["total_duration"],
            "total_tokens": stats["total_tokens_used"],
            "agents_participated": stats["agents_participated"],
            "agents_created": stats["agents_created"],
            "messages_processed": stats["total_messages_processed"],
            "final_confidence": results["final_output"]["confidence"] if results["final_output"] else 0.0,
            "llm_stats": stats.get("llm_client_stats", {}),
            "content_length": len(results["final_output"]["content"]) if results["final_output"] else 0
        }
        
        # Add server-specific metrics if available
        if hasattr(writer.llm_client, 'get_pool_stats'):
            test_result["pool_stats"] = writer.llm_client.get_pool_stats()
        
        print(f"✅ Test completed: {total_duration:.2f}s, {test_result['total_tokens']} tokens")
        
        return test_result
    
    async def run_comparison_tests(self, num_iterations: int = 3) -> Dict[str, Any]:
        """
        Run comprehensive comparison tests.
        
        Args:
            num_iterations: Number of test iterations per configuration
            
        Returns:
            Comparison results
        """
        print(f"🚀 STARTING MULTI-SERVER PERFORMANCE COMPARISON")
        print(f"Iterations per config: {num_iterations}")
        print(f"Topics: {len(self.test_topics)}")
        print(f"Total tests: {len(self.test_topics) * num_iterations * 2}")
        
        all_results = {
            "single_server": [],
            "multi_server": [],
            "test_config": {
                "num_iterations": num_iterations,
                "topics": self.test_topics,
                "debug_mode": self.debug_mode,
                "timestamp": time.time()
            }
        }
        
        # Test each topic with both configurations
        for topic in self.test_topics:
            print(f"\n🎯 Testing topic: {topic}")
            
            for iteration in range(num_iterations):
                print(f"\n--- Iteration {iteration + 1}/{num_iterations} ---")
                
                # Test single server
                try:
                    single_result = await self.run_single_test(
                        topic, 
                        "config/single_server_config.json",
                        f"Single-Server-{iteration+1}"
                    )
                    all_results["single_server"].append(single_result)
                except Exception as e:
                    print(f"❌ Single server test failed: {e}")
                    all_results["single_server"].append({
                        "error": str(e),
                        "test_name": f"Single-Server-{iteration+1}",
                        "topic": topic
                    })
                
                # Brief pause between tests
                await asyncio.sleep(2)
                
                # Test multi server
                try:
                    multi_result = await self.run_single_test(
                        topic,
                        "config/server_config.json", 
                        f"Multi-Server-{iteration+1}"
                    )
                    all_results["multi_server"].append(multi_result)
                except Exception as e:
                    print(f"❌ Multi server test failed: {e}")
                    all_results["multi_server"].append({
                        "error": str(e),
                        "test_name": f"Multi-Server-{iteration+1}",
                        "topic": topic
                    })
                
                # Brief pause between tests
                await asyncio.sleep(2)
        
        return all_results
    
    def analyze_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze and compare test results.
        
        Args:
            results: Results from run_comparison_tests
            
        Returns:
            Analysis summary
        """
        print(f"\n{'='*60}")
        print(f"📊 ANALYZING RESULTS")
        print(f"{'='*60}")
        
        # Filter successful tests
        single_success = [r for r in results["single_server"] if "error" not in r and r.get("success", False)]
        multi_success = [r for r in results["multi_server"] if "error" not in r and r.get("success", False)]
        
        print(f"Successful tests: Single={len(single_success)}, Multi={len(multi_success)}")
        
        if not single_success or not multi_success:
            return {"error": "Insufficient successful tests for comparison"}
        
        # Calculate metrics
        def calc_stats(test_list: List[Dict], metric: str) -> Dict[str, float]:
            values = [t[metric] for t in test_list if metric in t]
            if not values:
                return {"mean": 0, "median": 0, "min": 0, "max": 0, "std": 0}
            
            return {
                "mean": statistics.mean(values),
                "median": statistics.median(values),
                "min": min(values),
                "max": max(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0,
                "count": len(values)
            }
        
        analysis = {
            "single_server": {
                "duration": calc_stats(single_success, "total_duration"),
                "tokens": calc_stats(single_success, "total_tokens"),
                "confidence": calc_stats(single_success, "final_confidence"),
                "content_length": calc_stats(single_success, "content_length"),
                "agents_participated": calc_stats(single_success, "agents_participated")
            },
            "multi_server": {
                "duration": calc_stats(multi_success, "total_duration"),
                "tokens": calc_stats(multi_success, "total_tokens"),
                "confidence": calc_stats(multi_success, "final_confidence"),
                "content_length": calc_stats(multi_success, "content_length"),
                "agents_participated": calc_stats(multi_success, "agents_participated")
            }
        }
        
        # Calculate performance improvements
        improvements = {}
        for metric in ["duration", "tokens", "confidence", "content_length"]:
            single_mean = analysis["single_server"][metric]["mean"]
            multi_mean = analysis["multi_server"][metric]["mean"]
            
            if single_mean > 0:
                if metric == "duration":  # Lower is better
                    improvement = ((single_mean - multi_mean) / single_mean) * 100
                    improvements[metric] = improvement
                else:  # Higher is better
                    improvement = ((multi_mean - single_mean) / single_mean) * 100
                    improvements[metric] = improvement
            else:
                improvements[metric] = 0
        
        analysis["improvements"] = improvements
        analysis["summary"] = {
            "single_tests": len(single_success),
            "multi_tests": len(multi_success),
            "speed_improvement": improvements.get("duration", 0),
            "token_difference": improvements.get("tokens", 0),
            "confidence_improvement": improvements.get("confidence", 0),
            "quality_improvement": improvements.get("content_length", 0)
        }
        
        return analysis
    
    def display_analysis(self, analysis: Dict[str, Any]):
        """Display analysis results in readable format."""
        if "error" in analysis:
            print(f"❌ Analysis Error: {analysis['error']}")
            return
        
        print(f"\n{'='*60}")
        print(f"📈 PERFORMANCE COMPARISON RESULTS")
        print(f"{'='*60}")
        
        summary = analysis["summary"]
        print(f"Tests: {summary['single_tests']} single-server, {summary['multi_tests']} multi-server")
        
        print(f"\n🏃 SPEED COMPARISON:")
        single_duration = analysis["single_server"]["duration"]["mean"]
        multi_duration = analysis["multi_server"]["duration"]["mean"]
        speed_improvement = summary["speed_improvement"]
        
        print(f"  Single-server: {single_duration:.2f}s average")
        print(f"  Multi-server:  {multi_duration:.2f}s average")
        if speed_improvement > 0:
            print(f"  ✅ Multi-server is {speed_improvement:.1f}% FASTER")
        else:
            print(f"  ❌ Multi-server is {abs(speed_improvement):.1f}% slower")
        
        print(f"\n🪙 TOKEN USAGE:")
        single_tokens = analysis["single_server"]["tokens"]["mean"]
        multi_tokens = analysis["multi_server"]["tokens"]["mean"]
        token_difference = summary["token_difference"]
        
        print(f"  Single-server: {single_tokens:.0f} tokens average")
        print(f"  Multi-server:  {multi_tokens:.0f} tokens average")
        print(f"  Difference: {token_difference:.1f}%")
        
        print(f"\n🎯 QUALITY METRICS:")
        single_conf = analysis["single_server"]["confidence"]["mean"]
        multi_conf = analysis["multi_server"]["confidence"]["mean"]
        single_length = analysis["single_server"]["content_length"]["mean"]
        multi_length = analysis["multi_server"]["content_length"]["mean"]
        
        print(f"  Confidence: {single_conf:.2f} vs {multi_conf:.2f} ({summary['confidence_improvement']:+.1f}%)")
        print(f"  Content length: {single_length:.0f} vs {multi_length:.0f} chars ({summary['quality_improvement']:+.1f}%)")
        
        print(f"\n🔍 DETAILED STATISTICS:")
        print(f"  Single-server duration: {single_duration:.2f}±{analysis['single_server']['duration']['std']:.2f}s")
        print(f"  Multi-server duration:  {multi_duration:.2f}±{analysis['multi_server']['duration']['std']:.2f}s")
        
        print(f"\n🏆 VERDICT:")
        if speed_improvement > 10:
            print("✅ MULTI-SERVER PROVIDES SIGNIFICANT PERFORMANCE IMPROVEMENT")
        elif speed_improvement > 0:
            print("✅ Multi-server provides modest performance improvement")
        else:
            print("❌ Multi-server shows no performance benefit (check server setup)")
        
        # Server utilization analysis
        multi_results = [r for r in analysis if r.get("pool_stats")]
        if multi_results:
            print(f"\n🌐 SERVER UTILIZATION:")
            # This would show which servers were used most
            print("  (Server utilization metrics available in detailed results)")
    
    def save_results(self, results: Dict[str, Any], analysis: Dict[str, Any], filename: str = None):
        """Save test results and analysis to file."""
        if filename is None:
            timestamp = int(time.time())
            filename = f"performance_comparison_{timestamp}.json"
        
        import json
        output_data = {
            "test_results": results,
            "analysis": analysis,
            "timestamp": time.time()
        }
        
        with open(filename, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n💾 Results saved to: {filename}")


async def main():
    """Main performance test function."""
    print("🚀 Felix Framework Multi-Server Performance Test")
    print("=" * 60)
    
    # Check if config files exist
    single_config = Path("config/single_server_config.json")
    multi_config = Path("config/server_config.json")
    
    if not single_config.exists():
        print(f"❌ Single server config not found: {single_config}")
        return
    
    if not multi_config.exists():
        print(f"❌ Multi server config not found: {multi_config}")
        return
    
    # Run performance tests
    test_runner = PerformanceTest(debug_mode=False)
    
    print("⚡ Running performance comparison tests...")
    results = await test_runner.run_comparison_tests(num_iterations=2)
    
    print("\n📊 Analyzing results...")
    analysis = test_runner.analyze_results(results)
    
    # Display results
    test_runner.display_analysis(analysis)
    
    # Save results
    test_runner.save_results(results, analysis)
    
    print("\n🎉 Performance comparison complete!")


if __name__ == "__main__":
    asyncio.run(main())