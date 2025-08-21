"""
Test suite for Priority 3: Prompt Optimization Pipeline

Tests all components of the prompt optimization system:
- PromptMetrics and performance calculation
- PromptMetricsTracker for tracking and analysis
- PromptTester for A/B testing framework
- FailureAnalyzer for learning from failures
- PromptOptimizer as the main coordinator

Uses mock-based testing approach for external dependencies.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import time
import statistics
from collections import deque

from src.agents.prompt_optimization import (
    PromptContext, PromptMetrics, PromptMetricsTracker,
    PromptTester, PromptVariation, FailurePattern, FailureAnalyzer,
    PromptOptimizer
)


class TestPromptMetrics(unittest.TestCase):
    """Test PromptMetrics dataclass and overall_score calculation."""
    
    def test_prompt_metrics_creation(self):
        """Test basic PromptMetrics creation."""
        metrics = PromptMetrics(
            output_quality=0.8,
            confidence=0.9,
            completion_time=5.0,
            token_efficiency=0.7,
            truncation_occurred=False,
            context=PromptContext.RESEARCH_EARLY
        )
        
        self.assertEqual(metrics.output_quality, 0.8)
        self.assertEqual(metrics.confidence, 0.9)
        self.assertEqual(metrics.completion_time, 5.0)
        self.assertEqual(metrics.token_efficiency, 0.7)
        self.assertFalse(metrics.truncation_occurred)
        self.assertEqual(metrics.context, PromptContext.RESEARCH_EARLY)
        self.assertIsInstance(metrics.timestamp, float)
        
    def test_overall_score_calculation(self):
        """Test overall score calculation with various metrics."""
        # High quality metrics
        metrics = PromptMetrics(
            output_quality=0.9,
            confidence=0.8,
            completion_time=2.0,  # Fast completion
            token_efficiency=0.85,
            truncation_occurred=False,
            context=PromptContext.ANALYSIS_MID
        )
        
        score = metrics.overall_score()
        self.assertGreater(score, 0.8)  # Should be high score
        self.assertLessEqual(score, 1.0)
        
    def test_overall_score_with_truncation_penalty(self):
        """Test overall score with truncation penalty."""
        # Same metrics but with truncation
        metrics_no_truncation = PromptMetrics(
            output_quality=0.8,
            confidence=0.7,
            completion_time=5.0,
            token_efficiency=0.6,
            truncation_occurred=False,
            context=PromptContext.SYNTHESIS_LATE
        )
        
        metrics_with_truncation = PromptMetrics(
            output_quality=0.8,
            confidence=0.7,
            completion_time=5.0,
            token_efficiency=0.6,
            truncation_occurred=True,
            context=PromptContext.SYNTHESIS_LATE
        )
        
        score_no_truncation = metrics_no_truncation.overall_score()
        score_with_truncation = metrics_with_truncation.overall_score()
        
        # Truncation should reduce score
        self.assertLess(score_with_truncation, score_no_truncation)
        self.assertAlmostEqual(score_with_truncation, score_no_truncation - 0.2, places=2)
        
    def test_overall_score_slow_completion(self):
        """Test overall score with slow completion time."""
        fast_metrics = PromptMetrics(
            output_quality=0.8,
            confidence=0.8,
            completion_time=2.0,  # Fast
            token_efficiency=0.8,
            truncation_occurred=False,
            context=PromptContext.GENERAL
        )
        
        slow_metrics = PromptMetrics(
            output_quality=0.8,
            confidence=0.8,
            completion_time=15.0,  # Very slow
            token_efficiency=0.8,
            truncation_occurred=False,
            context=PromptContext.GENERAL
        )
        
        fast_score = fast_metrics.overall_score()
        slow_score = slow_metrics.overall_score()
        
        # Slow completion should reduce score
        self.assertLess(slow_score, fast_score)


class TestPromptMetricsTracker(unittest.TestCase):
    """Test PromptMetricsTracker for performance tracking."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.tracker = PromptMetricsTracker(history_size=100)
        
    def test_record_metrics(self):
        """Test recording metrics for a prompt."""
        metrics = PromptMetrics(
            output_quality=0.8,
            confidence=0.9,
            completion_time=3.0,
            token_efficiency=0.7,
            truncation_occurred=False,
            context=PromptContext.RESEARCH_EARLY
        )
        
        self.tracker.record_metrics("test_prompt_1", metrics)
        
        # Check metrics were recorded
        self.assertIn("test_prompt_1", self.tracker.metrics_history)
        self.assertEqual(len(self.tracker.metrics_history["test_prompt_1"]), 1)
        self.assertEqual(len(self.tracker.context_performance[PromptContext.RESEARCH_EARLY]), 1)
        
    def test_get_prompt_performance(self):
        """Test getting performance statistics for a prompt."""
        # Add multiple metrics for same prompt
        for i in range(5):
            metrics = PromptMetrics(
                output_quality=0.8 + i * 0.02,  # Varying quality
                confidence=0.7 + i * 0.05,
                completion_time=3.0 + i * 0.5,
                token_efficiency=0.6 + i * 0.03,
                truncation_occurred=i % 2 == 0,  # Alternate truncation
                context=PromptContext.ANALYSIS_MID
            )
            self.tracker.record_metrics("test_prompt", metrics)
            
        performance = self.tracker.get_prompt_performance("test_prompt")
        
        self.assertIsNotNone(performance)
        if performance is not None:
            self.assertEqual(performance['sample_size'], 5)
            self.assertIn('mean_score', performance)
            self.assertIn('std_score', performance)
            self.assertIn('mean_quality', performance)
            self.assertIn('mean_confidence', performance)
            self.assertIn('truncation_rate', performance)
            
            # Check truncation rate calculation
            self.assertEqual(performance['truncation_rate'], 0.6)  # 3 out of 5
        
    def test_get_prompt_performance_nonexistent(self):
        """Test getting performance for non-existent prompt."""
        performance = self.tracker.get_prompt_performance("nonexistent")
        self.assertIsNone(performance)
        
    def test_get_context_performance(self):
        """Test getting performance statistics for a context."""
        # Add metrics for same context
        for i in range(3):
            metrics = PromptMetrics(
                output_quality=0.7 + i * 0.1,
                confidence=0.6 + i * 0.1,
                completion_time=4.0,
                token_efficiency=0.5 + i * 0.1,
                truncation_occurred=False,
                context=PromptContext.SYNTHESIS_LATE
            )
            self.tracker.record_metrics(f"prompt_{i}", metrics)
            
        context_perf = self.tracker.get_context_performance(PromptContext.SYNTHESIS_LATE)
        
        self.assertIsNotNone(context_perf)
        if context_perf is not None:
            self.assertEqual(context_perf['sample_size'], 3)
            self.assertIn('mean_score', context_perf)
            self.assertIn('std_score', context_perf)
        
    def test_get_best_prompts(self):
        """Test getting best performing prompts."""
        # Add prompts with different performance levels
        prompts_data = [
            ("excellent_prompt", 0.9, 0.9, 2.0, 0.8),
            ("good_prompt", 0.7, 0.8, 5.0, 0.6),
            ("poor_prompt", 0.4, 0.5, 10.0, 0.3),
        ]
        
        for prompt_id, quality, confidence, time, efficiency in prompts_data:
            metrics = PromptMetrics(
                output_quality=quality,
                confidence=confidence,
                completion_time=time,
                token_efficiency=efficiency,
                truncation_occurred=False,
                context=PromptContext.GENERAL
            )
            self.tracker.record_metrics(prompt_id, metrics)
            
        best_prompts = self.tracker.get_best_prompts(limit=2)
        
        self.assertEqual(len(best_prompts), 2)
        # Should be sorted by score descending
        self.assertEqual(best_prompts[0][0], "excellent_prompt")
        self.assertEqual(best_prompts[1][0], "good_prompt")
        self.assertGreater(best_prompts[0][1], best_prompts[1][1])
        
    def test_get_best_prompts_filtered_by_context(self):
        """Test getting best prompts filtered by specific context."""
        # Add prompts with different contexts
        contexts_data = [
            ("research_prompt", PromptContext.RESEARCH_EARLY, 0.8),
            ("analysis_prompt", PromptContext.ANALYSIS_MID, 0.9),
            ("synthesis_prompt", PromptContext.SYNTHESIS_LATE, 0.7),
        ]
        
        for prompt_id, context, quality in contexts_data:
            metrics = PromptMetrics(
                output_quality=quality,
                confidence=0.8,
                completion_time=5.0,
                token_efficiency=0.7,
                truncation_occurred=False,
                context=context
            )
            self.tracker.record_metrics(prompt_id, metrics)
            
        # Get best prompts for analysis context only
        best_analysis = self.tracker.get_best_prompts(
            context=PromptContext.ANALYSIS_MID, limit=5
        )
        
        self.assertEqual(len(best_analysis), 1)
        self.assertEqual(best_analysis[0][0], "analysis_prompt")


class TestPromptTester(unittest.TestCase):
    """Test PromptTester for A/B testing framework."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.tester = PromptTester(min_samples_per_variant=3, confidence_level=0.95)
        
    def test_create_test(self):
        """Test creating a new A/B test."""
        base_prompt = "What is machine learning?"
        variations = [
            "Explain machine learning concepts",
            "Describe machine learning in simple terms"
        ]
        
        test_id = self.tester.create_test("ml_test", base_prompt, variations)
        
        self.assertEqual(test_id, "ml_test")
        self.assertIn("ml_test", self.tester.active_tests)
        
        variations_list = self.tester.active_tests["ml_test"]
        self.assertEqual(len(variations_list), 3)  # Control + 2 variations
        
        # Check control variation
        control = variations_list[0]
        self.assertIn("control", control.variation_id)
        self.assertEqual(control.prompt_text, base_prompt)
        self.assertEqual(control.generation_method, "control")
        
        # Check test variations
        for i, variation in enumerate(variations_list[1:]):
            self.assertIn(f"var_{i}", variation.variation_id)
            self.assertEqual(variation.prompt_text, variations[i])
            self.assertEqual(variation.generation_method, "systematic")
            
    def test_add_test_result(self):
        """Test adding test results to a variation."""
        base_prompt = "Test prompt"
        variations = ["Variation 1"]
        test_id = self.tester.create_test("test_1", base_prompt, variations)
        
        metrics = PromptMetrics(
            output_quality=0.8,
            confidence=0.9,
            completion_time=3.0,
            token_efficiency=0.7,
            truncation_occurred=False,
            context=PromptContext.GENERAL
        )
        
        # Add result to control
        control_id = f"{test_id}_control"
        result = self.tester.add_test_result(test_id, control_id, metrics)
        
        self.assertTrue(result)  # Should continue testing
        
        # Check result was added
        control_variation = self.tester.active_tests[test_id][0]
        self.assertEqual(len(control_variation.test_results), 1)
        
    def test_add_test_result_invalid_test(self):
        """Test adding result to non-existent test."""
        metrics = PromptMetrics(
            output_quality=0.8,
            confidence=0.9,
            completion_time=3.0,
            token_efficiency=0.7,
            truncation_occurred=False,
            context=PromptContext.GENERAL
        )
        
        result = self.tester.add_test_result("nonexistent", "variation", metrics)
        self.assertFalse(result)
        
    def test_early_stopping_large_effect(self):
        """Test early stopping when large effect size is detected."""
        base_prompt = "Test prompt"
        variations = ["Better variation"]
        test_id = self.tester.create_test("early_stop_test", base_prompt, variations)
        
        # Add poor results for control
        for i in range(5):
            poor_metrics = PromptMetrics(
                output_quality=0.3,
                confidence=0.4,
                completion_time=8.0,
                token_efficiency=0.2,
                truncation_occurred=True,
                context=PromptContext.GENERAL
            )
            self.tester.add_test_result(test_id, f"{test_id}_control", poor_metrics)
            
        # Add excellent results for variation
        should_continue = True  # Initialize to avoid unbound variable
        for i in range(5):
            excellent_metrics = PromptMetrics(
                output_quality=0.95,
                confidence=0.9,
                completion_time=2.0,
                token_efficiency=0.85,
                truncation_occurred=False,
                context=PromptContext.GENERAL
            )
            should_continue = self.tester.add_test_result(test_id, f"{test_id}_var_0", excellent_metrics)
            
        # With such a large difference, test might stop early
        # Check if test was completed (moved to completed_tests)
        if not should_continue:
            self.assertIn(test_id, self.tester.completed_tests)
            self.assertNotIn(test_id, self.tester.active_tests)
            
    def test_get_test_status_active(self):
        """Test getting status of active test."""
        base_prompt = "Test prompt"
        variations = ["Variation 1"]
        test_id = self.tester.create_test("status_test", base_prompt, variations)
        
        # Add some results
        metrics = PromptMetrics(
            output_quality=0.8,
            confidence=0.9,
            completion_time=3.0,
            token_efficiency=0.7,
            truncation_occurred=False,
            context=PromptContext.GENERAL
        )
        self.tester.add_test_result(test_id, f"{test_id}_control", metrics)
        
        status = self.tester.get_test_status(test_id)
        
        self.assertIsNotNone(status)
        if status is not None:
            self.assertEqual(status['status'], 'active')
            self.assertIn('variations', status)
            self.assertEqual(len(status['variations']), 2)  # Control + 1 variation
        
    def test_get_test_status_nonexistent(self):
        """Test getting status of non-existent test."""
        status = self.tester.get_test_status("nonexistent")
        self.assertIsNone(status)


class TestPromptVariation(unittest.TestCase):
    """Test PromptVariation dataclass."""
    
    def test_prompt_variation_creation(self):
        """Test creating a prompt variation."""
        variation = PromptVariation(
            prompt_text="Test prompt",
            variation_id="test_var_1",
            parent_prompt_id="test_control",
            generation_method="systematic"
        )
        
        self.assertEqual(variation.prompt_text, "Test prompt")
        self.assertEqual(variation.variation_id, "test_var_1")
        self.assertEqual(variation.parent_prompt_id, "test_control")
        self.assertEqual(variation.generation_method, "systematic")
        self.assertEqual(len(variation.test_results), 0)
        
    def test_add_result_and_performance(self):
        """Test adding results and calculating performance."""
        variation = PromptVariation(
            prompt_text="Test prompt",
            variation_id="test_var"
        )
        
        # No results initially
        self.assertIsNone(variation.get_performance())
        
        # Add some results
        for i in range(3):
            metrics = PromptMetrics(
                output_quality=0.7 + i * 0.1,
                confidence=0.8,
                completion_time=4.0,
                token_efficiency=0.6,
                truncation_occurred=False,
                context=PromptContext.GENERAL
            )
            variation.add_result(metrics)
            
        self.assertEqual(len(variation.test_results), 3)
        
        # Check performance calculation
        performance = variation.get_performance()
        self.assertIsNotNone(performance)
        if performance is not None:
            self.assertIsInstance(performance, float)
            self.assertGreaterEqual(performance, 0.0)
            self.assertLessEqual(performance, 1.0)


class TestFailureAnalyzer(unittest.TestCase):
    """Test FailureAnalyzer for learning from failures."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = FailureAnalyzer(failure_threshold=0.5)
        
    def test_analyze_failure_truncation(self):
        """Test analyzing truncation failures."""
        metrics = PromptMetrics(
            output_quality=0.3,  # Low quality
            confidence=0.6,
            completion_time=5.0,
            token_efficiency=0.4,
            truncation_occurred=True,  # Truncated
            context=PromptContext.RESEARCH_EARLY
        )
        
        self.analyzer.analyze_failure("truncated_prompt", metrics)
        
        # Should record truncation pattern
        pattern_key = f"truncation_{PromptContext.RESEARCH_EARLY.value}"
        self.assertIn(pattern_key, self.analyzer.failure_patterns)
        
        pattern = self.analyzer.failure_patterns[pattern_key]
        self.assertEqual(pattern.pattern_type, "truncation")
        self.assertEqual(pattern.context, PromptContext.RESEARCH_EARLY)
        self.assertEqual(pattern.frequency, 1)
        self.assertIn("truncated_prompt", pattern.examples)
        
    def test_analyze_failure_low_confidence(self):
        """Test analyzing low confidence failures."""
        metrics = PromptMetrics(
            output_quality=0.4,  # Reduce quality to ensure overall score < 0.5
            confidence=0.2,  # Very low confidence
            completion_time=8.0,  # Slower completion
            token_efficiency=0.3,  # Lower efficiency
            truncation_occurred=False,
            context=PromptContext.ANALYSIS_MID
        )
        
        self.analyzer.analyze_failure("low_conf_prompt", metrics)
        
        pattern_key = f"low_confidence_{PromptContext.ANALYSIS_MID.value}"
        self.assertIn(pattern_key, self.analyzer.failure_patterns)
        
        pattern = self.analyzer.failure_patterns[pattern_key]
        self.assertEqual(pattern.pattern_type, "low_confidence")
        self.assertEqual(pattern.frequency, 1)
        
    def test_analyze_failure_low_quality(self):
        """Test analyzing low quality failures."""
        metrics = PromptMetrics(
            output_quality=0.1,  # Very low quality
            confidence=0.8,
            completion_time=3.0,
            token_efficiency=0.2,
            truncation_occurred=False,
            context=PromptContext.SYNTHESIS_LATE
        )
        
        self.analyzer.analyze_failure("low_quality_prompt", metrics)
        
        pattern_key = f"low_quality_{PromptContext.SYNTHESIS_LATE.value}"
        self.assertIn(pattern_key, self.analyzer.failure_patterns)
        
        pattern = self.analyzer.failure_patterns[pattern_key]
        self.assertEqual(pattern.pattern_type, "low_quality")
        self.assertEqual(pattern.frequency, 1)
        
    def test_analyze_non_failure(self):
        """Test that good metrics are not recorded as failures."""
        metrics = PromptMetrics(
            output_quality=0.8,
            confidence=0.9,
            completion_time=3.0,
            token_efficiency=0.7,
            truncation_occurred=False,
            context=PromptContext.GENERAL
        )
        
        initial_patterns = len(self.analyzer.failure_patterns)
        self.analyzer.analyze_failure("good_prompt", metrics)
        
        # Should not add any new failure patterns
        self.assertEqual(len(self.analyzer.failure_patterns), initial_patterns)
        
    def test_get_improvement_suggestions(self):
        """Test getting improvement suggestions for a context."""
        # Create frequent failure pattern
        for i in range(5):
            metrics = PromptMetrics(
                output_quality=0.25,  # Change to < 0.3 to trigger low_quality pattern
                confidence=0.2,
                completion_time=10.0,
                token_efficiency=0.2,
                truncation_occurred=True,
                context=PromptContext.RESEARCH_EARLY
            )
            self.analyzer.analyze_failure(f"bad_prompt_{i}", metrics)
            
        suggestions = self.analyzer.get_improvement_suggestions(PromptContext.RESEARCH_EARLY)
        
        self.assertGreater(len(suggestions), 0)
        
        # Should include multiple failure types
        pattern_types = {s['pattern_type'] for s in suggestions}
        self.assertIn('truncation', pattern_types)
        self.assertIn('low_confidence', pattern_types)
        self.assertIn('low_quality', pattern_types)
        
        # Check suggestion structure
        for suggestion in suggestions:
            self.assertIn('pattern_type', suggestion)
            self.assertIn('frequency', suggestion)
            self.assertIn('suggested_fix', suggestion)
            self.assertIn('severity', suggestion)
            
    def test_generate_improved_prompts(self):
        """Test generating improved prompts based on failure analysis."""
        # Create failure patterns
        for i in range(4):  # Above threshold of 3
            metrics = PromptMetrics(
                output_quality=0.2,
                confidence=0.3,
                completion_time=12.0,
                token_efficiency=0.1,
                truncation_occurred=True,
                context=PromptContext.ANALYSIS_MID
            )
            self.analyzer.analyze_failure(f"failing_prompt_{i}", metrics)
            
        base_prompt = "Analyze the following complex data and provide insights"
        improved_prompts = self.analyzer.generate_improved_prompts(
            base_prompt, PromptContext.ANALYSIS_MID
        )
        
        self.assertGreater(len(improved_prompts), 0)
        
        # Each improved prompt should contain the original plus improvements
        for improved in improved_prompts:
            self.assertIn(base_prompt, improved)
            # Should contain improvement suggestions
            self.assertTrue(
                any(keyword in improved.lower() for keyword in 
                    ['concise', 'confidence', 'quality', 'structured'])
            )


class TestPromptOptimizer(unittest.TestCase):
    """Test PromptOptimizer main coordinator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.optimizer = PromptOptimizer()
        
    def test_record_prompt_execution(self):
        """Test recording a prompt execution."""
        metrics = PromptMetrics(
            output_quality=0.8,
            confidence=0.9,
            completion_time=3.0,
            token_efficiency=0.7,
            truncation_occurred=False,
            context=PromptContext.GENERAL
        )
        
        result = self.optimizer.record_prompt_execution(
            "test_prompt", "What is AI?", metrics
        )
        
        self.assertEqual(result['status'], 'recorded')
        self.assertIn('prompt_performance', result)
        
        # Check metrics were recorded in tracker
        performance = self.optimizer.metrics_tracker.get_prompt_performance("test_prompt")
        self.assertIsNotNone(performance)
        
    def test_record_prompt_execution_triggers_optimization(self):
        """Test that poor performance triggers optimization."""
        # Add multiple poor results to trigger optimization
        for i in range(6):  # Above threshold of 5
            poor_metrics = PromptMetrics(
                output_quality=0.3,  # Poor quality
                confidence=0.4,
                completion_time=10.0,
                token_efficiency=0.2,
                truncation_occurred=True,  # High truncation
                context=PromptContext.RESEARCH_EARLY
            )
            
            result = self.optimizer.record_prompt_execution(
                "poor_prompt", "Generate comprehensive analysis", poor_metrics
            )
            
        # Should trigger optimization after 5+ samples with poor performance
        self.assertIn("poor_prompt", self.optimizer.active_optimizations)
        
        optimization = self.optimizer.active_optimizations["poor_prompt"]
        self.assertIn('test_id', optimization)
        self.assertIn('started_at', optimization)
        self.assertEqual(optimization['context'], PromptContext.RESEARCH_EARLY)
        
    def test_get_optimization_recommendations(self):
        """Test getting optimization recommendations for a context."""
        # Add some metrics for the context
        for i, quality in enumerate([0.9, 0.7, 0.5]):
            metrics = PromptMetrics(
                output_quality=quality,
                confidence=0.8,
                completion_time=4.0,
                token_efficiency=0.6,
                truncation_occurred=False,
                context=PromptContext.SYNTHESIS_LATE
            )
            self.optimizer.record_prompt_execution(f"prompt_{i}", f"Test prompt {i}", metrics)
            
        recommendations = self.optimizer.get_optimization_recommendations(
            PromptContext.SYNTHESIS_LATE
        )
        
        self.assertEqual(recommendations['context'], PromptContext.SYNTHESIS_LATE.value)
        self.assertIn('context_performance', recommendations)
        self.assertIn('best_prompts', recommendations)
        self.assertIn('improvement_suggestions', recommendations)
        self.assertIn('active_tests', recommendations)
        
        # Should have context performance data
        context_perf = recommendations['context_performance']
        self.assertIsNotNone(context_perf)
        self.assertEqual(context_perf['sample_size'], 3)
        
    def test_get_system_status(self):
        """Test getting overall system status."""
        # Add some data to the system
        for i in range(3):
            metrics = PromptMetrics(
                output_quality=0.7 + i * 0.1,
                confidence=0.8,
                completion_time=4.0,
                token_efficiency=0.6,
                truncation_occurred=False,
                context=PromptContext.GENERAL
            )
            self.optimizer.record_prompt_execution(f"status_prompt_{i}", f"Test {i}", metrics)
            
        status = self.optimizer.get_system_status()
        
        self.assertIn('total_prompts_tracked', status)
        self.assertIn('active_tests', status)
        self.assertIn('completed_tests', status)
        self.assertIn('failure_patterns_identified', status)
        self.assertIn('overall_performance', status)
        self.assertIn('active_optimizations', status)
        self.assertIn('system_health', status)
        
        self.assertEqual(status['total_prompts_tracked'], 3)
        self.assertIn(status['system_health'], ['excellent', 'good', 'needs_improvement'])
        
    def test_system_health_classification(self):
        """Test system health classification based on performance."""
        # Add excellent performance data
        for i in range(5):
            excellent_metrics = PromptMetrics(
                output_quality=0.95,
                confidence=0.9,
                completion_time=2.0,
                token_efficiency=0.85,
                truncation_occurred=False,
                context=PromptContext.GENERAL
            )
            self.optimizer.record_prompt_execution(f"excellent_{i}", "Perfect prompt", excellent_metrics)
            
        status = self.optimizer.get_system_status()
        
        # Should have excellent health with high performance
        self.assertEqual(status['system_health'], 'excellent')
        self.assertGreater(status['overall_performance'], 0.8)


class TestFailurePattern(unittest.TestCase):
    """Test FailurePattern dataclass."""
    
    def test_failure_pattern_creation(self):
        """Test creating a failure pattern."""
        pattern = FailurePattern(
            pattern_type="truncation",
            context=PromptContext.RESEARCH_EARLY,
            frequency=5,
            avg_failure_score=0.3,
            suggested_fix="Make prompts more concise",
            examples=["prompt1", "prompt2"]
        )
        
        self.assertEqual(pattern.pattern_type, "truncation")
        self.assertEqual(pattern.context, PromptContext.RESEARCH_EARLY)
        self.assertEqual(pattern.frequency, 5)
        self.assertEqual(pattern.avg_failure_score, 0.3)
        self.assertEqual(pattern.suggested_fix, "Make prompts more concise")
        self.assertEqual(len(pattern.examples), 2)


if __name__ == '__main__':
    unittest.main()
