"""
Unit tests for Quality Metrics System.

Tests the QualityMetricsCalculator, BLEUCalculator, CoherenceAnalyzer, 
FactAccuracyChecker, and supporting classes to ensure accurate quality assessment.
"""

import pytest
import math
from unittest.mock import MagicMock, patch
from src.comparison.quality_metrics import (
    QualityMetricsCalculator, BLEUCalculator, CoherenceAnalyzer, 
    FactAccuracyChecker, QualityScore, QualityDimension, DomainType,
    ReferenceText
)


class TestQualityScore:
    """Test QualityScore data class and serialization."""
    
    def test_quality_score_creation(self):
        """Test basic QualityScore creation."""
        score = QualityScore(overall_score=0.85)
        assert score.overall_score == 0.85
        assert score.coherence_score == 0.0
        assert score.accuracy_score == 0.0
        assert score.word_count == 0
        assert score.domain == DomainType.GENERAL
    
    def test_quality_score_with_dimensions(self):
        """Test QualityScore with dimension scores."""
        dimension_scores = {
            QualityDimension.COHERENCE: 0.8,
            QualityDimension.ACCURACY: 0.9,
            QualityDimension.CLARITY: 0.7
        }
        
        score = QualityScore(
            overall_score=0.8,
            dimension_scores=dimension_scores,
            coherence_score=0.8,
            accuracy_score=0.9,
            clarity_score=0.7,
            word_count=100,
            sentence_count=5,
            domain=DomainType.TECHNICAL
        )
        
        assert score.overall_score == 0.8
        assert score.dimension_scores[QualityDimension.COHERENCE] == 0.8
        assert score.coherence_score == 0.8
        assert score.word_count == 100
        assert score.domain == DomainType.TECHNICAL
    
    def test_to_dict_serialization(self):
        """Test QualityScore serialization to dictionary."""
        dimension_scores = {
            QualityDimension.COHERENCE: 0.8,
            QualityDimension.ACCURACY: 0.9
        }
        
        score = QualityScore(
            overall_score=0.85,
            dimension_scores=dimension_scores,
            bleu_score=0.75,
            coherence_score=0.8,
            word_count=150,
            domain=DomainType.BUSINESS
        )
        
        score_dict = score.to_dict()
        
        assert score_dict["overall_score"] == 0.85
        assert score_dict["dimension_scores"]["coherence"] == 0.8
        assert score_dict["dimension_scores"]["accuracy"] == 0.9
        assert score_dict["bleu_score"] == 0.75
        assert score_dict["coherence_score"] == 0.8
        assert score_dict["word_count"] == 150
        assert score_dict["domain"] == "business"
        assert "assessment_time" in score_dict


class TestReferenceText:
    """Test ReferenceText data class."""
    
    def test_reference_text_creation(self):
        """Test basic ReferenceText creation."""
        ref = ReferenceText(
            text="This is a sample text for testing.",
            domain=DomainType.TECHNICAL,
            quality_level="high",
            source="test_suite"
        )
        
        assert ref.text == "This is a sample text for testing."
        assert ref.domain == DomainType.TECHNICAL
        assert ref.quality_level == "high"
        assert ref.source == "test_suite"
        assert isinstance(ref.metadata, dict)


class TestBLEUCalculator:
    """Test BLEU score calculation."""
    
    def setUp(self):
        """Set up BLEU calculator for tests."""
        self.calculator = BLEUCalculator(max_n=4)
    
    def test_bleu_calculator_initialization(self):
        """Test BLEU calculator initialization."""
        calc = BLEUCalculator(max_n=3)
        assert calc.max_n == 3
        
        calc_default = BLEUCalculator()
        assert calc_default.max_n == 4
    
    def test_tokenize_basic(self):
        """Test basic tokenization."""
        calc = BLEUCalculator()
        tokens = calc._tokenize("The quick brown fox jumps.")
        expected = ["The", "quick", "brown", "fox", "jumps"]
        assert tokens == expected
    
    def test_tokenize_with_punctuation(self):
        """Test tokenization with various punctuation."""
        calc = BLEUCalculator()
        tokens = calc._tokenize("Hello, world! How are you?")
        expected = ["Hello", "world", "How", "are", "you"]
        assert tokens == expected
    
    def test_get_n_grams(self):
        """Test n-gram extraction."""
        calc = BLEUCalculator()
        tokens = ["the", "quick", "brown", "fox"]
        
        # Test unigrams
        unigrams = calc._get_n_grams(tokens, 1)
        assert unigrams == [("the",), ("quick",), ("brown",), ("fox",)]
        
        # Test bigrams
        bigrams = calc._get_n_grams(tokens, 2)
        assert bigrams == [("the", "quick"), ("quick", "brown"), ("brown", "fox")]
        
        # Test trigrams
        trigrams = calc._get_n_grams(tokens, 3)
        assert trigrams == [("the", "quick", "brown"), ("quick", "brown", "fox")]
    
    def test_get_n_grams_insufficient_tokens(self):
        """Test n-gram extraction with insufficient tokens."""
        calc = BLEUCalculator()
        tokens = ["hello", "world"]
        
        # Should return empty list for n > token count
        result = calc._get_n_grams(tokens, 5)
        assert result == []
    
    def test_calculate_bleu_identical_texts(self):
        """Test BLEU score for identical texts."""
        calc = BLEUCalculator()
        candidate = "The quick brown fox jumps over the lazy dog."
        references = ["The quick brown fox jumps over the lazy dog."]
        
        bleu_score = calc.calculate_bleu(candidate, references)
        assert bleu_score == pytest.approx(1.0, rel=1e-3)
    
    def test_calculate_bleu_empty_input(self):
        """Test BLEU score for empty inputs."""
        calc = BLEUCalculator()
        
        # Empty candidate
        assert calc.calculate_bleu("", ["reference text"]) == 0.0
        
        # Empty references
        assert calc.calculate_bleu("candidate text", []) == 0.0
        
        # Both empty
        assert calc.calculate_bleu("", []) == 0.0
    
    def test_calculate_bleu_partial_match(self):
        """Test BLEU score for partially matching texts."""
        calc = BLEUCalculator()
        candidate = "The brown fox jumps quickly."
        references = ["The quick brown fox jumps over the lazy dog."]
        
        bleu_score = calc.calculate_bleu(candidate, references)
        # Should be between 0 and 1, but not perfect
        assert 0.0 < bleu_score < 1.0
    
    def test_calculate_bleu_multiple_references(self):
        """Test BLEU score with multiple reference texts."""
        calc = BLEUCalculator()
        candidate = "The brown fox runs fast."
        references = [
            "The quick brown fox jumps.",
            "A fast brown fox runs quickly.",
            "The brown fox moves rapidly."
        ]
        
        bleu_score = calc.calculate_bleu(candidate, references)
        assert 0.0 <= bleu_score <= 1.0


class TestCoherenceAnalyzer:
    """Test coherence analysis functionality."""
    
    def setUp(self):
        """Set up coherence analyzer for tests."""
        self.analyzer = CoherenceAnalyzer()
    
    def test_coherence_analyzer_initialization(self):
        """Test coherence analyzer initialization."""
        analyzer = CoherenceAnalyzer()
        assert "sequence" in analyzer.transition_words
        assert "contrast" in analyzer.transition_words
        assert len(analyzer.coherence_patterns) > 0
    
    def test_split_sentences_basic(self):
        """Test basic sentence splitting."""
        analyzer = CoherenceAnalyzer()
        text = "This is first sentence. This is second! Is this third?"
        sentences = analyzer._split_sentences(text)
        expected = ["This is first sentence", "This is second", "Is this third"]
        assert sentences == expected
    
    def test_analyze_coherence_empty_text(self):
        """Test coherence analysis on empty or very short text."""
        analyzer = CoherenceAnalyzer()
        
        assert analyzer.analyze_coherence("") == 0.0
        assert analyzer.analyze_coherence("short") == 0.0
        assert analyzer.analyze_coherence("A bit longer text here.") == 0.5  # Single sentence
    
    def test_analyze_coherence_high_coherence_text(self):
        """Test coherence analysis on highly coherent text."""
        analyzer = CoherenceAnalyzer()
        text = """First, we need to understand the problem. Then, we can analyze the data. 
                  Furthermore, this approach allows us to identify patterns. Finally, we can 
                  draw conclusions from these findings."""
        
        coherence_score = analyzer.analyze_coherence(text)
        assert coherence_score > 0.6  # Should be high coherence
    
    def test_analyze_coherence_low_coherence_text(self):
        """Test coherence analysis on low coherence text."""
        analyzer = CoherenceAnalyzer()
        text = """Random sentence about cats. Database optimization is important. 
                  The weather today is nice. Programming languages have syntax."""
        
        coherence_score = analyzer.analyze_coherence(text)
        assert 0.0 <= coherence_score <= 1.0  # Valid range, but likely lower
    
    def test_calculate_transition_score(self):
        """Test transition word scoring."""
        analyzer = CoherenceAnalyzer()
        
        # Text with good transitions
        text_with_transitions = "First, we analyze the data. However, there are limitations. Therefore, we need more research."
        transition_score = analyzer._calculate_transition_score(text_with_transitions)
        assert transition_score >= 0.5
        
        # Text without transitions
        text_without_transitions = "We analyze the data. There are limitations. We need more research."
        no_transition_score = analyzer._calculate_transition_score(text_without_transitions)
        assert no_transition_score < transition_score
    
    def test_calculate_reference_score(self):
        """Test cross-sentence reference scoring."""
        analyzer = CoherenceAnalyzer()
        
        # Text with good references
        sentences_with_refs = [
            "The algorithm processes data efficiently.",
            "This approach reduces computational complexity.",
            "It also improves memory usage significantly."
        ]
        ref_score = analyzer._calculate_reference_score(sentences_with_refs)
        assert ref_score > 0.0
        
        # Text without references
        sentences_without_refs = [
            "Algorithms process data.",
            "Complexity matters in computing.",
            "Memory usage affects performance."
        ]
        no_ref_score = analyzer._calculate_reference_score(sentences_without_refs)
        assert no_ref_score <= ref_score


class TestFactAccuracyChecker:
    """Test fact accuracy checking functionality."""
    
    def setUp(self):
        """Set up fact accuracy checker for tests."""
        self.checker = FactAccuracyChecker()
    
    def test_fact_checker_initialization(self):
        """Test fact accuracy checker initialization."""
        checker = FactAccuracyChecker()
        assert len(checker.high_confidence_patterns) > 0
        assert len(checker.uncertainty_patterns) > 0
        assert len(checker.contradiction_patterns) > 0
    
    def test_check_accuracy_empty_text(self):
        """Test accuracy checking on empty or very short text."""
        checker = FactAccuracyChecker()
        
        assert checker.check_accuracy("") == 0.5
        assert checker.check_accuracy("short") == 0.5
    
    def test_check_accuracy_high_confidence_text(self):
        """Test accuracy checking on high-confidence text."""
        checker = FactAccuracyChecker()
        text = "Studies show that 85% of users prefer this approach. Research indicates significant improvements. Data shows clear patterns."
        
        accuracy_score = checker.check_accuracy(text)
        assert accuracy_score > 0.7  # Should be high confidence
    
    def test_check_accuracy_uncertain_text(self):
        """Test accuracy checking on uncertain text."""
        checker = FactAccuracyChecker()
        text = "I think this might work. Perhaps the approach could be beneficial. It seems like there may be advantages."
        
        accuracy_score = checker.check_accuracy(text)
        assert accuracy_score < 0.7  # Should indicate some uncertainty, but not necessarily inaccurate
    
    def test_calculate_confidence_indicators(self):
        """Test confidence indicator calculation."""
        checker = FactAccuracyChecker()
        
        # High confidence text
        high_conf_text = "research indicates proven results with 95% accuracy"
        high_score = checker._calculate_confidence_indicators(high_conf_text.lower())
        assert high_score > 0.5
        
        # Uncertain text
        uncertain_text = "might possibly work perhaps maybe"
        low_score = checker._calculate_confidence_indicators(uncertain_text.lower())
        assert low_score < 0.5
    
    def test_calculate_internal_consistency(self):
        """Test internal consistency calculation."""
        checker = FactAccuracyChecker()
        
        # Consistent text
        consistent_text = "The algorithm works well. It processes data efficiently. Performance is excellent."
        consistent_score = checker._calculate_internal_consistency(consistent_text)
        
        # Text with contradictions
        contradictory_text = "The algorithm works well. However, it fails often. But it's also very reliable."
        contradictory_score = checker._calculate_internal_consistency(contradictory_text)
        
        assert consistent_score >= contradictory_score
    
    def test_calculate_specificity_score(self):
        """Test specificity score calculation."""
        checker = FactAccuracyChecker()
        
        # Specific text with numbers, dates, names (optimal specificity range)
        specific_text = "Study ABC-123 showed 87% improvement in performance metrics."
        specific_score = checker._calculate_specificity_score(specific_text)
        
        # General text with minimal specificity
        general_text = "Some people saw good results with the new approach."
        general_score = checker._calculate_specificity_score(general_text)
        
        assert specific_score > general_score


class TestQualityMetricsCalculator:
    """Test main quality metrics calculator."""
    
    def setUp(self):
        """Set up quality metrics calculator for tests."""
        self.calculator = QualityMetricsCalculator()
    
    def test_quality_calculator_initialization(self):
        """Test quality metrics calculator initialization."""
        calc = QualityMetricsCalculator()
        assert isinstance(calc.bleu_calculator, BLEUCalculator)
        assert isinstance(calc.coherence_analyzer, CoherenceAnalyzer)
        assert isinstance(calc.accuracy_checker, FactAccuracyChecker)
        assert DomainType.TECHNICAL in calc.reference_texts
    
    def test_calculate_quality_score_empty_text(self):
        """Test quality score calculation on empty text."""
        calc = QualityMetricsCalculator()
        score = calc.calculate_quality_score("")
        
        assert score.overall_score == 0.0
        assert score.word_count == 0
        assert score.domain == DomainType.GENERAL
    
    def test_calculate_quality_score_basic_text(self):
        """Test quality score calculation on basic text."""
        calc = QualityMetricsCalculator()
        text = """This is a well-structured document that demonstrates clear thinking. 
                  First, we establish the foundation. Then, we build upon these concepts. 
                  Furthermore, the approach shows promising results. Therefore, we can 
                  conclude that this method is effective."""
        
        score = calc.calculate_quality_score(text, DomainType.GENERAL)
        
        assert 0.0 <= score.overall_score <= 1.0
        assert score.word_count > 0
        assert score.sentence_count > 0
        assert score.coherence_score > 0.0
        assert score.accuracy_score > 0.0
        assert score.completeness_score >= 0.0
        assert score.domain == DomainType.GENERAL
    
    def test_calculate_quality_score_technical_domain(self):
        """Test quality score calculation for technical domain."""
        calc = QualityMetricsCalculator()
        text = """The algorithm implements a helix-based architecture for multi-agent systems. 
                  Technical implementation uses geometric constraints to optimize performance. 
                  System design follows established engineering principles with measurable results."""
        
        score = calc.calculate_quality_score(text, DomainType.TECHNICAL)
        
        assert score.domain == DomainType.TECHNICAL
        assert score.relevance_score > 0.5  # Should have good relevance for technical content
        assert score.bleu_score is not None  # Should calculate BLEU with reference texts
    
    def test_calculate_quality_score_business_domain(self):
        """Test quality score calculation for business domain."""
        calc = QualityMetricsCalculator()
        text = """Market analysis reveals strong competitive advantages in our approach. 
                  Business strategy optimization leads to improved revenue performance. 
                  Cost-benefit analysis demonstrates significant efficiency gains."""
        
        score = calc.calculate_quality_score(text, DomainType.BUSINESS)
        
        assert score.domain == DomainType.BUSINESS
        assert score.relevance_score > 0.5  # Should have good relevance for business content
    
    def test_calculate_quality_score_with_custom_references(self):
        """Test quality score calculation with custom reference texts."""
        calc = QualityMetricsCalculator()
        text = "The quick brown fox jumps over the lazy dog."
        references = ["A quick brown fox leaps over a sleeping dog."]
        
        score = calc.calculate_quality_score(text, reference_texts=references)
        
        assert score.bleu_score is not None
        assert 0.0 <= score.bleu_score <= 1.0
    
    def test_calculate_completeness(self):
        """Test completeness calculation."""
        calc = QualityMetricsCalculator()
        
        # Complete text with introduction, content, conclusion, examples
        complete_text = """Introduction: This document outlines our approach. 
                          For example, we can demonstrate the concept with specific cases. 
                          Because of these factors, we achieve better results. 
                          In conclusion, the method proves effective."""
        
        completeness_score = calc._calculate_completeness(complete_text)
        assert completeness_score > 0.7  # Should be high completeness
        
        # Incomplete text
        incomplete_text = "Just some random content without structure."
        incomplete_score = calc._calculate_completeness(incomplete_text)
        assert incomplete_score < completeness_score
    
    def test_calculate_clarity(self):
        """Test clarity calculation."""
        calc = QualityMetricsCalculator()
        
        # Clear text with good sentence length and clear indicators
        clear_text = """First, we need to understand the problem clearly. 
                        This means analyzing all available data systematically. 
                        Therefore, our approach follows a structured methodology."""
        
        clarity_score = calc._calculate_clarity(clear_text)
        assert clarity_score > 0.5
        
        # Unclear text with very long sentences
        unclear_text = """This is an extremely long sentence that goes on and on without providing clear structure or easy-to-understand explanations which makes it very difficult for readers to follow the logic and comprehend the intended meaning effectively."""
        
        unclear_score = calc._calculate_clarity(unclear_text)
        assert unclear_score < clarity_score
    
    def test_calculate_relevance_domain_specific(self):
        """Test domain-specific relevance calculation."""
        calc = QualityMetricsCalculator()
        
        # Technical text
        tech_text = "Algorithm implementation using advanced system architecture and engineering design frameworks."
        tech_relevance = calc._calculate_relevance(tech_text, DomainType.TECHNICAL)
        
        # Business text  
        business_text = "Market analysis reveals competitive strategy optimization for revenue performance."
        business_relevance = calc._calculate_relevance(business_text, DomainType.BUSINESS)
        
        # Tech text should score higher on technical relevance
        tech_business_score = calc._calculate_relevance(tech_text, DomainType.BUSINESS)
        assert tech_relevance > tech_business_score
        
        # General domain should return neutral score
        general_relevance = calc._calculate_relevance("Some general content", DomainType.GENERAL)
        assert general_relevance == 0.8
    
    def test_calculate_originality(self):
        """Test originality calculation."""
        calc = QualityMetricsCalculator()
        
        # Original creative text
        original_text = "Imagine an innovative approach that envisions groundbreaking solutions remarkably."
        original_score = calc._calculate_originality(original_text)
        
        # Formulaic text
        formulaic_text = "In conclusion, it is important to note that there are many factors to consider."
        formulaic_score = calc._calculate_originality(formulaic_text)
        
        assert original_score > formulaic_score
    
    def test_calculate_structure(self):
        """Test structure calculation."""
        calc = QualityMetricsCalculator()
        
        # Well-structured text
        structured_text = """# Introduction
                            
                            This document covers important topics.
                            
                            ## Main Points
                            
                            1. First point with details
                            2. Second point with examples
                            • Additional bullet point
                            
                            Furthermore, we can add more content. However, we must consider limitations."""
        
        structure_score = calc._calculate_structure(structured_text)
        assert structure_score > 0.7
        
        # Unstructured text
        unstructured_text = "Just plain text without any formatting or structure indicators at all."
        unstructured_score = calc._calculate_structure(unstructured_text)
        
        assert structure_score > unstructured_score
    
    def test_add_reference_text(self):
        """Test adding reference texts."""
        calc = QualityMetricsCalculator()
        
        # Add custom reference text
        custom_ref = ReferenceText(
            text="Custom reference text for testing purposes.",
            domain=DomainType.CREATIVE,
            quality_level="medium",
            source="test_suite"
        )
        
        calc.add_reference_text(custom_ref)
        
        assert DomainType.CREATIVE in calc.reference_texts
        assert custom_ref in calc.reference_texts[DomainType.CREATIVE]
    
    def test_batch_calculate_scores(self):
        """Test batch score calculation."""
        calc = QualityMetricsCalculator()
        
        texts = [
            "First document with good content structure.",
            "Second document demonstrates clear analysis.",
            "Third document provides comprehensive coverage."
        ]
        
        scores = calc.batch_calculate_scores(texts, DomainType.GENERAL)
        
        assert len(scores) == 3
        assert all(isinstance(score, QualityScore) for score in scores)
        assert all(score.domain == DomainType.GENERAL for score in scores)
        assert all(0.0 <= score.overall_score <= 1.0 for score in scores)


class TestIntegrationScenarios:
    """Test integration scenarios for quality metrics system."""
    
    def test_felix_blog_post_quality_assessment(self):
        """Test quality assessment on Felix Framework blog post content."""
        calc = QualityMetricsCalculator()
        
        blog_content = """
        # Understanding the Felix Framework: A Helix-Based Approach to Multi-Agent Systems
        
        The Felix Framework represents a revolutionary approach to multi-agent orchestration. 
        Unlike traditional systems that rely on explicit graph definitions, Felix uses geometric 
        principles to coordinate autonomous agents through a helix-based architecture.
        
        ## Core Architecture Principles
        
        First, agents spawn at different times but follow identical geometric constraints. 
        This creates natural attention focusing as agents progress from the wide top (radius 33) 
        to the narrow bottom (radius 0.001) of the helix structure.
        
        Furthermore, the framework implements O(N) spoke-based communication rather than 
        O(N²) mesh architectures. This design choice significantly improves scalability 
        while maintaining coordination effectiveness.
        
        ## Practical Applications
        
        Research indicates that helix-based coordination offers measurable advantages:
        • 4,119x geometric concentration ratio for attention focusing
        • Natural temperature adjustment based on helix position
        • Distributed processing with central coordination
        
        For example, in blog writing tasks, research agents spawn early for exploration, 
        while synthesis agents spawn later for consolidation. This temporal orchestration 
        creates efficient task distribution patterns.
        
        ## Conclusion
        
        Therefore, the Felix Framework demonstrates how geometric principles can enhance 
        multi-agent system design. Studies show improved task distribution efficiency 
        with statistical significance (p=0.0441) compared to traditional approaches.
        """
        
        score = calc.calculate_quality_score(blog_content, DomainType.TECHNICAL)
        
        # Verify comprehensive assessment
        assert score.overall_score > 0.6  # Should be good quality
        assert score.coherence_score > 0.7  # Well-structured with transitions
        assert score.accuracy_score > 0.6  # Has confidence indicators and specifics
        assert score.completeness_score >= 0.8  # Has intro, content, examples, conclusion
        assert score.clarity_score > 0.6  # Good sentence structure and explanations
        assert score.relevance_score >= 0.8  # High technical relevance
        assert score.structure_score >= 0.5  # Headers, bullets, paragraphs
        assert score.word_count >= 190
        assert score.sentence_count > 10
        assert score.bleu_score is not None  # Should compare against technical references
    
    def test_comparative_quality_assessment(self):
        """Test quality assessment comparing different content types."""
        calc = QualityMetricsCalculator()
        
        # High-quality technical content
        high_quality = """
        Advanced algorithm implementations require systematic design approaches. 
        Research demonstrates that structured methodologies yield superior results. 
        For example, the geometric helix model provides 4,119x concentration ratios 
        with statistical significance. Therefore, this framework offers measurable 
        performance improvements over traditional architectures.
        """
        
        # Medium-quality general content
        medium_quality = """
        This approach works well for many situations. It seems like a good solution 
        that might help with various problems. People often find it useful, although 
        results may vary depending on circumstances.
        """
        
        # Low-quality fragmented content
        low_quality = """
        Random thoughts. No structure here. Different topic now about weather. 
        Algorithms maybe work sometimes. Conclusion: things happen.
        """
        
        high_score = calc.calculate_quality_score(high_quality, DomainType.TECHNICAL)
        medium_score = calc.calculate_quality_score(medium_quality, DomainType.GENERAL)
        low_score = calc.calculate_quality_score(low_quality, DomainType.GENERAL)
        
        # Verify quality ordering (overall scores should be clearly ordered)
        assert high_score.overall_score > medium_score.overall_score > low_score.overall_score
        
        # High quality should have better accuracy than medium quality at least
        assert high_score.accuracy_score > medium_score.accuracy_score
        
        # High quality should have significantly better coherence than low quality
        assert high_score.coherence_score > low_score.coherence_score
    
    def test_domain_specific_weighting_effects(self):
        """Test how domain-specific weighting affects scores."""
        calc = QualityMetricsCalculator()
        
        # Content that's accurate but not very original
        accurate_content = """
        Studies show that 95% of systems demonstrate improved performance. 
        Research indicates significant efficiency gains with proven methodologies. 
        Data confirms statistical significance across multiple test scenarios.
        """
        
        # Content that's original but less accurate
        creative_content = """
        Imagine a revolutionary paradigm that envisions unprecedented solutions. 
        This groundbreaking approach surprisingly transforms conventional thinking. 
        Remarkably innovative concepts emerge through visionary methodologies.
        """
        
        # Test different domain weightings
        accurate_scientific = calc.calculate_quality_score(accurate_content, DomainType.SCIENTIFIC)
        accurate_creative = calc.calculate_quality_score(accurate_content, DomainType.CREATIVE)
        
        creative_scientific = calc.calculate_quality_score(creative_content, DomainType.SCIENTIFIC)
        creative_creative = calc.calculate_quality_score(creative_content, DomainType.CREATIVE)
        
        # Accurate content should score higher in scientific domain (accuracy weighted 35%)
        assert accurate_scientific.overall_score > accurate_creative.overall_score
        
        # Creative content should score higher in creative domain (originality weighted 30%)
        assert creative_creative.overall_score > creative_scientific.overall_score