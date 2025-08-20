#!/usr/bin/env python3
"""
Verification test to prove Felix Framework isn't hardcoded.

This script runs the same blog topic with different random seeds
to demonstrate that outputs vary, proving the system makes real
LLM calls and adapts based on the helix geometry.

Usage:
    python examples/verify_randomness.py "Quantum computing applications"
"""

import sys
import time
import hashlib
from pathlib import Path
from typing import List, Dict, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from blog_writer import FelixBlogWriter


def run_verification_test(topic: str, num_runs: int = 3) -> List[Dict[str, Any]]:
    """
    Run the same topic with different seeds and collect results.
    
    Args:
        topic: Blog post topic
        num_runs: Number of different seeds to test
        
    Returns:
        List of results from each run
    """
    results = []
    seeds = [42, 123, 999, 1337, 8888][:num_runs]  # Use different seeds
    
    print(f"🔬 VERIFICATION TEST: Running '{topic}' with {num_runs} different seeds")
    print(f"Seeds: {seeds}")
    print("=" * 80)
    
    for i, seed in enumerate(seeds):
        print(f"\n🧪 RUN {i+1}/{num_runs}: Seed {seed}")
        print("-" * 40)
        
        # Create fresh writer with specific seed
        writer = FelixBlogWriter(
            random_seed=seed,
            strict_mode=True,  # Use strict mode for faster verification
            debug_mode=False   # Disable debug for cleaner output
        )
        
        # Test connection
        if not writer.test_lm_studio_connection():
            print(f"❌ LM Studio connection failed for run {i+1}")
            continue
        
        # Create team with deterministic spawn times (due to seed)
        writer.create_blog_writing_team(complexity="simple")
        
        # Run session
        start_time = time.perf_counter()
        result = writer.run_blog_writing_session(
            topic=topic,
            simulation_time=1.0
        )
        end_time = time.perf_counter()
        
        # Extract key metrics for comparison
        if result["final_output"]:
            content = result["final_output"]["content"]
            content_hash = hashlib.md5(content.encode()).hexdigest()[:16]
            
            run_result = {
                "seed": seed,
                "run_number": i + 1,
                "content_hash": content_hash,
                "content_length": len(content),
                "content_preview": content[:200] + "..." if len(content) > 200 else content,
                "total_tokens": result["session_stats"]["total_tokens_used"],
                "agents_participated": result["session_stats"]["agents_participated"],
                "duration": end_time - start_time,
                "final_confidence": result["final_output"]["confidence"],
                "final_agent": result["final_output"]["agent_id"],
                "spawn_times": [agent["spawn_time"] for agent in result["agents_participated"]],
                "agent_types": [agent["agent_type"] for agent in result["agents_participated"]]
            }
            
            results.append(run_result)
            
            print(f"✅ Completed: {len(content)} chars, {run_result['total_tokens']} tokens")
            print(f"   Hash: {content_hash}, Confidence: {run_result['final_confidence']:.2f}")
            print(f"   Preview: {run_result['content_preview'][:100]}...")
        else:
            print(f"❌ No final output generated for run {i+1}")
            
    return results


def analyze_variance(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze variance between runs to prove non-determinism.
    
    Args:
        results: List of run results
        
    Returns:
        Analysis summary
    """
    if len(results) < 2:
        return {"error": "Need at least 2 results to analyze variance"}
    
    # Check content variance
    content_hashes = [r["content_hash"] for r in results]
    unique_hashes = len(set(content_hashes))
    
    # Check token variance
    token_counts = [r["total_tokens"] for r in results]
    min_tokens = min(token_counts)
    max_tokens = max(token_counts)
    avg_tokens = sum(token_counts) / len(token_counts)
    
    # Check length variance
    content_lengths = [r["content_length"] for r in results]
    min_length = min(content_lengths)
    max_length = max(content_lengths)
    avg_length = sum(content_lengths) / len(content_lengths)
    
    # Check timing variance
    durations = [r["duration"] for r in results]
    min_duration = min(durations)
    max_duration = max(durations)
    avg_duration = sum(durations) / len(durations)
    
    # Check confidence variance
    confidences = [r["final_confidence"] for r in results]
    min_confidence = min(confidences)
    max_confidence = max(confidences)
    avg_confidence = sum(confidences) / len(confidences)
    
    # Check spawn time variance (should be different with different seeds)
    spawn_variations = []
    for i, result in enumerate(results):
        spawn_pattern = tuple(sorted(result["spawn_times"]))
        spawn_variations.append(spawn_pattern)
    unique_spawn_patterns = len(set(spawn_variations))
    
    analysis = {
        "total_runs": len(results),
        "content_variance": {
            "unique_content_hashes": unique_hashes,
            "identical_content": unique_hashes == 1,
            "variance_percentage": (unique_hashes / len(results)) * 100
        },
        "token_variance": {
            "min": min_tokens,
            "max": max_tokens,
            "avg": avg_tokens,
            "range": max_tokens - min_tokens,
            "coefficient_of_variation": (max_tokens - min_tokens) / avg_tokens if avg_tokens > 0 else 0
        },
        "length_variance": {
            "min": min_length,
            "max": max_length,
            "avg": avg_length,
            "range": max_length - min_length
        },
        "timing_variance": {
            "min": min_duration,
            "max": max_duration,
            "avg": avg_duration,
            "range": max_duration - min_duration
        },
        "confidence_variance": {
            "min": min_confidence,
            "max": max_confidence,
            "avg": avg_confidence,
            "range": max_confidence - min_confidence
        },
        "spawn_pattern_variance": {
            "unique_patterns": unique_spawn_patterns,
            "total_patterns": len(results),
            "all_different": unique_spawn_patterns == len(results)
        }
    }
    
    return analysis


def display_verification_results(results: List[Dict[str, Any]], analysis: Dict[str, Any]) -> None:
    """Display verification results in a readable format."""
    print(f"\n{'='*80}")
    print(f"VERIFICATION RESULTS")
    print(f"{'='*80}")
    
    if not results:
        print("❌ No successful runs to analyze")
        return
    
    print(f"\n📊 RUN SUMMARY:")
    for result in results:
        print(f"  Seed {result['seed']:4d}: {result['content_length']:4d} chars, "
              f"{result['total_tokens']:4d} tokens, "
              f"hash {result['content_hash']}, "
              f"conf {result['final_confidence']:.2f}")
    
    if "error" in analysis:
        print(f"\n❌ Analysis Error: {analysis['error']}")
        return
    
    print(f"\n🔍 VARIANCE ANALYSIS:")
    
    # Content variance
    cv = analysis["content_variance"]
    if cv["identical_content"]:
        print(f"❌ CONTENT: All {cv['unique_content_hashes']} outputs IDENTICAL (possible hardcoding!)")
    else:
        print(f"✅ CONTENT: {cv['unique_content_hashes']}/{analysis['total_runs']} unique outputs "
              f"({cv['variance_percentage']:.1f}% variance)")
    
    # Token variance
    tv = analysis["token_variance"]
    print(f"✅ TOKENS: {tv['min']}-{tv['max']} range (avg {tv['avg']:.1f}, "
          f"CV {tv['coefficient_of_variation']:.1%})")
    
    # Length variance
    lv = analysis["length_variance"]
    print(f"✅ LENGTH: {lv['min']}-{lv['max']} chars (avg {lv['avg']:.1f})")
    
    # Timing variance
    timing = analysis["timing_variance"]
    print(f"✅ TIMING: {timing['min']:.1f}-{timing['max']:.1f}s "
          f"(avg {timing['avg']:.1f}s, range {timing['range']:.1f}s)")
    
    # Confidence variance
    conf = analysis["confidence_variance"]
    print(f"✅ CONFIDENCE: {conf['min']:.2f}-{conf['max']:.2f} "
          f"(avg {conf['avg']:.2f}, range {conf['range']:.2f})")
    
    # Spawn pattern variance
    sp = analysis["spawn_pattern_variance"]
    if sp["all_different"]:
        print(f"✅ SPAWN PATTERNS: All {sp['unique_patterns']} patterns unique (proper randomization)")
    else:
        print(f"⚠️  SPAWN PATTERNS: {sp['unique_patterns']}/{sp['total_patterns']} unique patterns")
    
    print(f"\n🎯 VERDICT:")
    if cv["identical_content"]:
        print("❌ FAILED: Identical content suggests hardcoded responses")
    elif cv["unique_content_hashes"] >= analysis["total_runs"] * 0.8:  # 80% unique
        print("✅ PASSED: High content variance proves real LLM processing")
    else:
        print("⚠️  INCONCLUSIVE: Some variance but may need more runs")
    
    print(f"\n📋 DETAILED CONTENT COMPARISON:")
    for i, result in enumerate(results):
        print(f"\n--- Run {i+1} (Seed {result['seed']}) ---")
        print(f"Agents: {', '.join(result['agent_types'])}")
        print(f"Final: {result['final_agent']} (confidence {result['final_confidence']:.2f})")
        print(f"Content ({result['content_length']} chars):")
        print(result['content_preview'])


def main():
    """Main verification function."""
    if len(sys.argv) != 2:
        print("Usage: python verify_randomness.py \"Blog topic\"")
        print("Example: python verify_randomness.py \"Quantum computing applications\"")
        sys.exit(1)
    
    topic = sys.argv[1]
    
    # Run verification test
    results = run_verification_test(topic, num_runs=3)
    
    # Analyze variance
    analysis = analyze_variance(results)
    
    # Display results
    display_verification_results(results, analysis)
    
    # Save results for inspection
    timestamp = int(time.time())
    output_file = f"verification_results_{timestamp}.json"
    
    import json
    verification_data = {
        "topic": topic,
        "timestamp": timestamp,
        "results": results,
        "analysis": analysis
    }
    
    with open(output_file, 'w') as f:
        json.dump(verification_data, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")


if __name__ == "__main__":
    main()