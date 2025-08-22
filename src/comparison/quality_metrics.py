"""
Quality Metrics Suite for Felix Framework

Implements Priority 4 of the enhancement plan:
- BLEU scores for text quality assessment
- Coherence metrics for logical flow analysis
- Fact accuracy checks with confidence scoring
- Integration with existing confidence calculation system
- Domain-specific quality assessments

This provides comprehensive validation against industry standards
for multi-agent system output quality.
"""

import re
import time
import statistics
import json
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import Counter, defaultdict
import math


class QualityDimension(Enum):
    """Dimensions of output quality assessment."""
    COHERENCE = "coherence"
    ACCURACY = "accuracy"
    COMPLETENESS = "completeness"
    CLARITY = "clarity"
    RELEVANCE = "relevance"
    ORIGINALITY = "originality"
    STRUCTURE = "structure"


class DomainType(Enum):
    """Domain types for specialized quality assessment."""
    TECHNICAL = "technical"
    BUSINESS = "business"
    CREATIVE = "creative"
    ANALYTICAL = "analytical"
    SCIENTIFIC = "scientific"
    GENERAL = "general"


@dataclass
class QualityScore:
    """Comprehensive quality score with detailed breakdown."""
    overall_score: float  # 0.0 to 1.0
    dimension_scores: Dict[QualityDimension, float] = field(default_factory=dict)
    bleu_score: Optional[float] = None
    coherence_score: float = 0.0
    accuracy_score: float = 0.0
    completeness_score: float = 0.0
    clarity_score: float = 0.0
    relevance_score: float = 0.0
    originality_score: float = 0.0
    structure_score: float = 0.0
    
    # Metadata
    word_count: int = 0
    sentence_count: int = 0
    domain: DomainType = DomainType.GENERAL
    assessment_time: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "overall_score": self.overall_score,
            "dimension_scores": {dim.value: score for dim, score in self.dimension_scores.items()},
            "bleu_score": self.bleu_score,
            "coherence_score": self.coherence_score,
            "accuracy_score": self.accuracy_score,
            "completeness_score": self.completeness_score,
            "clarity_score": self.clarity_score,
            "relevance_score": self.relevance_score,
            "originality_score": self.originality_score,
            "structure_score": self.structure_score,
            "word_count": self.word_count,
            "sentence_count": self.sentence_count,
            "domain": self.domain.value,
            "assessment_time": self.assessment_time
        }


@dataclass
class ReferenceText:
    """Reference text for quality comparison."""
    text: str
    domain: DomainType
    quality_level: str  # "low", "medium", "high", "expert"
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class BLEUCalculator:
    """
    BLEU score calculator for text quality assessment.
    
    Implements modified BLEU that doesn't require exact reference matches,
    focusing on n-gram overlap patterns for quality assessment.
    """
    
    def __init__(self, max_n: int = 4):
        """
        Initialize BLEU calculator.
        
        Args:
            max_n: Maximum n-gram size for calculation
        """
        self.max_n = max_n
    
    def calculate_bleu(self, candidate: str, references: List[str]) -> float:
        """
        Calculate BLEU score against reference texts.
        
        Args:
            candidate: Text to evaluate
            references: List of reference texts
            
        Returns:
            BLEU score between 0.0 and 1.0
        """
        if not candidate or not references:
            return 0.0
        
        # Tokenize texts
        candidate_tokens = self._tokenize(candidate.lower())
        reference_tokens_list = [self._tokenize(ref.lower()) for ref in references]
        
        # Calculate precision for each n-gram size
        precisions = []
        for n in range(1, self.max_n + 1):
            precision = self._calculate_n_gram_precision(
                candidate_tokens, reference_tokens_list, n
            )
            precisions.append(precision)
        
        # Calculate geometric mean of precisions with smoothing
        # Apply smoothing to avoid zero precision for partial matches
        smoothed_precisions = []
        for p in precisions:
            # Add small epsilon for smoothing when precision is 0
            smoothed_p = max(p, 1e-7)
            smoothed_precisions.append(smoothed_p)
        
        geometric_mean = math.exp(sum(math.log(p) for p in smoothed_precisions) / len(smoothed_precisions))
        
        # Apply brevity penalty
        candidate_length = len(candidate_tokens)
        closest_ref_length = min(
            (len(ref_tokens) for ref_tokens in reference_tokens_list),
            key=lambda x: abs(x - candidate_length)
        )
        
        if candidate_length >= closest_ref_length:
            brevity_penalty = 1.0
        else:
            brevity_penalty = math.exp(1 - closest_ref_length / candidate_length)
        
        return geometric_mean * brevity_penalty
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization by whitespace and punctuation."""
        # Basic tokenization - split on whitespace and common punctuation
        tokens = re.findall(r'\b\w+\b', text)
        return tokens
    
    def _get_n_grams(self, tokens: List[str], n: int) -> List[Tuple[str, ...]]:
        """Extract n-grams from token list."""
        if len(tokens) < n:
            return []
        return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    
    def _calculate_n_gram_precision(self, candidate_tokens: List[str], 
                                  reference_tokens_list: List[List[str]], n: int) -> float:
        """Calculate n-gram precision."""
        candidate_n_grams = self._get_n_grams(candidate_tokens, n)
        if not candidate_n_grams:
            return 0.0
        
        # Count n-grams in candidate
        candidate_counts = Counter(candidate_n_grams)
        
        # Get maximum counts from all references
        max_ref_counts = Counter()
        for ref_tokens in reference_tokens_list:
            ref_n_grams = self._get_n_grams(ref_tokens, n)
            ref_counts = Counter(ref_n_grams)
            for ngram, count in ref_counts.items():
                max_ref_counts[ngram] = max(max_ref_counts[ngram], count)
        
        # Calculate clipped counts
        clipped_counts = 0
        total_counts = 0
        for ngram, count in candidate_counts.items():
            clipped_counts += min(count, max_ref_counts[ngram])
            total_counts += count
        
        return clipped_counts / total_counts if total_counts > 0 else 0.0


class CoherenceAnalyzer:
    """
    Analyzes text coherence through multiple linguistic features.
    """
    
    def __init__(self):
        """Initialize coherence analyzer with linguistic patterns."""
        # Transition words and phrases
        self.transition_words = {
            "sequence": ["first", "second", "third", "then", "next", "finally", "subsequently"],
            "contrast": ["however", "but", "although", "despite", "conversely", "nevertheless"],
            "cause": ["because", "therefore", "thus", "consequently", "as a result", "due to"],
            "addition": ["furthermore", "moreover", "additionally", "also", "besides"],
            "example": ["for example", "for instance", "such as", "specifically", "namely"],
            "conclusion": ["in conclusion", "therefore", "thus", "overall", "to summarize"]
        }
        
        # Coherence indicators
        self.coherence_patterns = [
            r"\b(this|that|these|those)\b",  # Demonstratives
            r"\b(it|they|them|its|their)\b",  # Pronouns
            r"\b(the|this|that)\s+(\w+)\b",  # Definite articles with repetition
        ]
    
    def analyze_coherence(self, text: str) -> float:
        """
        Analyze text coherence returning score 0.0 to 1.0.
        
        Args:
            text: Text to analyze
            
        Returns:
            Coherence score between 0.0 and 1.0
        """
        if not text or len(text.strip()) < 10:
            return 0.0
        
        sentences = self._split_sentences(text)
        if len(sentences) < 2:
            return 0.5  # Single sentence gets moderate score
        
        # Calculate multiple coherence metrics
        transition_score = self._calculate_transition_score(text)
        reference_score = self._calculate_reference_score(sentences)
        repetition_score = self._calculate_repetition_score(sentences)
        flow_score = self._calculate_flow_score(sentences)
        
        # Weighted combination
        coherence_score = (
            0.3 * transition_score +
            0.3 * reference_score +
            0.2 * repetition_score +
            0.2 * flow_score
        )
        
        return min(1.0, max(0.0, coherence_score))
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting on periods, exclamations, questions
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _calculate_transition_score(self, text: str) -> float:
        """Calculate score based on transition word usage."""
        text_lower = text.lower()
        total_transitions = 0
        
        for category, words in self.transition_words.items():
            for word in words:
                total_transitions += len(re.findall(r'\b' + re.escape(word) + r'\b', text_lower))
        
        # Normalize by text length (transitions per 100 words)
        word_count = len(text.split())
        if word_count == 0:
            return 0.0
        
        transition_density = total_transitions / word_count * 100
        
        # Optimal transition density is around 2-5 per 100 words
        if transition_density == 0:
            return 0.0
        elif transition_density <= 2:
            return transition_density / 2.0
        elif transition_density <= 5:
            return 1.0
        else:
            # Too many transitions can hurt coherence
            return max(0.5, 1.0 - (transition_density - 5) / 10)
    
    def _calculate_reference_score(self, sentences: List[str]) -> float:
        """Calculate score based on cross-sentence references."""
        if len(sentences) < 2:
            return 0.5
        
        reference_count = 0
        total_opportunities = len(sentences) - 1
        
        for i in range(1, len(sentences)):
            sentence_lower = sentences[i].lower()
            
            # Check for reference patterns
            for pattern in self.coherence_patterns:
                if re.search(pattern, sentence_lower):
                    reference_count += 1
                    break
        
        return reference_count / total_opportunities if total_opportunities > 0 else 0.0
    
    def _calculate_repetition_score(self, sentences: List[str]) -> float:
        """Calculate score based on appropriate word repetition."""
        if len(sentences) < 2:
            return 0.5
        
        # Extract content words (nouns, verbs, adjectives approximation)
        all_words = []
        for sentence in sentences:
            words = [word.lower() for word in re.findall(r'\b\w{3,}\b', sentence)]
            all_words.extend(words)
        
        if not all_words:
            return 0.0
        
        # Calculate repetition ratio
        word_counts = Counter(all_words)
        repeated_words = sum(1 for count in word_counts.values() if count > 1)
        unique_words = len(word_counts)
        
        # Optimal repetition ratio is around 20-40%
        repetition_ratio = repeated_words / unique_words if unique_words > 0 else 0.0
        
        if repetition_ratio <= 0.1:
            return repetition_ratio / 0.1 * 0.5  # Too little repetition
        elif repetition_ratio <= 0.4:
            return 0.5 + (repetition_ratio - 0.1) / 0.3 * 0.5
        else:
            return max(0.3, 1.0 - (repetition_ratio - 0.4) / 0.3)  # Too much repetition
    
    def _calculate_flow_score(self, sentences: List[str]) -> float:
        """Calculate score based on sentence flow patterns."""
        if len(sentences) < 3:
            return 0.6
        
        # Analyze sentence length variation
        lengths = [len(sentence.split()) for sentence in sentences]
        
        if not lengths:
            return 0.0
        
        # Calculate length variation coefficient
        mean_length = statistics.mean(lengths)
        if mean_length == 0:
            return 0.0
        
        length_std = statistics.stdev(lengths) if len(lengths) > 1 else 0.0
        variation_coefficient = length_std / mean_length
        
        # Optimal variation is around 0.3-0.7
        if variation_coefficient <= 0.1:
            return 0.3  # Too monotonous
        elif variation_coefficient <= 0.7:
            return 0.6 + (variation_coefficient - 0.1) / 0.6 * 0.4
        else:
            return max(0.4, 1.0 - (variation_coefficient - 0.7) / 0.5)


class FactAccuracyChecker:
    """
    Checks factual accuracy through pattern analysis and consistency checking.
    
    Note: This is a simplified version focusing on consistency and confidence
    indicators rather than external fact verification.
    """
    
    def __init__(self):
        """Initialize fact checker with accuracy patterns."""
        # Confidence indicators (higher confidence in facts)
        self.high_confidence_patterns = [
            r"\b(studies show|research indicates|data shows|proven|established|confirmed)\b",
            r"\b(according to|based on|evidence suggests|findings indicate)\b",
            r"\b(\d+%|\d+\.\d+%|statistics|numbers|figures)\b"
        ]
        
        # Uncertainty indicators (lower confidence in facts)
        self.uncertainty_patterns = [
            r"\b(might|may|could|possibly|perhaps|likely|probably|seems|appears)\b",
            r"\b(I think|I believe|I assume|presumably|allegedly)\b",
            r"\b(unclear|uncertain|unknown|unconfirmed|unverified)\b"
        ]
        
        # Contradiction indicators
        self.contradiction_patterns = [
            r"\b(however|but|although|despite|conversely|on the other hand)\b",
            r"\b(disagree|contradict|conflict|inconsistent|opposite)\b"
        ]
    
    def check_accuracy(self, text: str) -> float:
        """
        Check factual accuracy returning confidence score 0.0 to 1.0.
        
        Args:
            text: Text to check for accuracy
            
        Returns:
            Accuracy confidence score between 0.0 and 1.0
        """
        if not text or len(text.strip()) < 10:
            return 0.5  # Neutral for very short text
        
        text_lower = text.lower()
        
        # Calculate confidence indicators
        confidence_score = self._calculate_confidence_indicators(text_lower)
        consistency_score = self._calculate_internal_consistency(text)
        specificity_score = self._calculate_specificity_score(text)
        
        # Weighted combination
        accuracy_score = (
            0.4 * confidence_score +
            0.4 * consistency_score +
            0.2 * specificity_score
        )
        
        return min(1.0, max(0.0, accuracy_score))
    
    def _calculate_confidence_indicators(self, text_lower: str) -> float:
        """Calculate confidence based on language patterns."""
        # Count high confidence patterns
        high_confidence_count = 0
        for pattern in self.high_confidence_patterns:
            high_confidence_count += len(re.findall(pattern, text_lower))
        
        # Count uncertainty patterns
        uncertainty_count = 0
        for pattern in self.uncertainty_patterns:
            uncertainty_count += len(re.findall(pattern, text_lower))
        
        # Calculate net confidence
        net_confidence = high_confidence_count - uncertainty_count
        word_count = len(text_lower.split())
        
        if word_count == 0:
            return 0.5
        
        # Normalize by text length
        confidence_ratio = net_confidence / word_count * 100
        
        # Map to 0.0-1.0 scale
        if confidence_ratio <= -2:
            return 0.2  # Very uncertain
        elif confidence_ratio <= 0:
            return 0.5 + confidence_ratio / 2 * 0.3  # Somewhat uncertain
        elif confidence_ratio <= 2:
            return 0.5 + confidence_ratio / 2 * 0.4  # Confident
        else:
            return min(1.0, 0.9)  # Very confident
    
    def _calculate_internal_consistency(self, text: str) -> float:
        """Check for internal contradictions."""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) < 2:
            return 0.8  # Single sentence assumed consistent
        
        text_lower = text.lower()
        
        # Count contradiction indicators
        contradiction_count = 0
        for pattern in self.contradiction_patterns:
            contradiction_count += len(re.findall(pattern, text_lower))
        
        # More contradictions = lower consistency
        total_sentences = len(sentences)
        contradiction_ratio = contradiction_count / total_sentences
        
        # Some contradictions are natural (presenting both sides)
        if contradiction_ratio <= 0.1:
            return 1.0  # Highly consistent
        elif contradiction_ratio <= 0.3:
            return 0.9 - (contradiction_ratio - 0.1) / 0.2 * 0.3  # Good consistency
        else:
            return max(0.4, 0.6 - (contradiction_ratio - 0.3) / 0.4 * 0.2)  # Lower consistency
    
    def _calculate_specificity_score(self, text: str) -> float:
        """Calculate score based on specificity and detail."""
        # Specific indicators: numbers, dates, names, technical terms
        specific_patterns = [
            r'\b\d+\b',  # Numbers
            r'\b\d{4}\b',  # Years
            r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',  # Proper names
            r'\b\w{8,}\b',  # Long words (often technical)
        ]
        
        specific_count = 0
        for pattern in specific_patterns:
            specific_count += len(re.findall(pattern, text))
        
        word_count = len(text.split())
        if word_count == 0:
            return 0.0
        
        specificity_ratio = specific_count / word_count
        
        # Map to 0.0-1.0 scale (optimal around 10-20% specific terms)
        if specificity_ratio <= 0.05:
            return specificity_ratio / 0.05 * 0.4  # Too general
        elif specificity_ratio <= 0.2:
            return 0.4 + (specificity_ratio - 0.05) / 0.15 * 0.6  # Good balance
        else:
            return max(0.7, 1.0 - (specificity_ratio - 0.2) / 0.3 * 0.3)  # Too dense


class QualityMetricsCalculator:
    """
    Main quality metrics calculator combining all assessment dimensions.
    """
    
    def __init__(self):
        """Initialize quality metrics calculator."""
        self.bleu_calculator = BLEUCalculator()
        self.coherence_analyzer = CoherenceAnalyzer()
        self.accuracy_checker = FactAccuracyChecker()
        
        # Reference texts for comparison (can be expanded)
        self.reference_texts = {
            DomainType.TECHNICAL: [
                ReferenceText(
                    text="The algorithm implements a helix-based cognitive architecture where autonomous agents navigate spiral processing paths. Each agent spawns at different times but follows the same geometric constraints, creating natural attention focusing as the radius decreases from 33 to 0.001 units.",
                    domain=DomainType.TECHNICAL,
                    quality_level="high",
                    source="felix_documentation"
                ),
            ],
            DomainType.BUSINESS: [
                ReferenceText(
                    text="Market analysis indicates strong demand for multi-agent orchestration systems. Current solutions like LangGraph require explicit graph definitions, while Felix offers geometric convergence with O(N) communication complexity compared to traditional O(N²) mesh architectures.",
                    domain=DomainType.BUSINESS,
                    quality_level="high",
                    source="business_analysis"
                ),
            ],
            DomainType.GENERAL: [
                ReferenceText(
                    text="This system demonstrates innovative approaches to complex problem solving through coordinated agent interaction. The framework provides both theoretical foundations and practical implementation guidance for researchers and developers.",
                    domain=DomainType.GENERAL,
                    quality_level="medium",
                    source="general_description"
                ),
            ]
        }
    
    def calculate_quality_score(self, text: str, domain: DomainType = DomainType.GENERAL,
                              reference_texts: Optional[List[str]] = None) -> QualityScore:
        """
        Calculate comprehensive quality score for text.
        
        Args:
            text: Text to evaluate
            domain: Domain type for specialized assessment
            reference_texts: Optional custom reference texts for BLEU
            
        Returns:
            Comprehensive quality score
        """
        if not text or len(text.strip()) < 5:
            return QualityScore(overall_score=0.0, domain=domain)
        
        # Basic metrics
        word_count = len(text.split())
        sentence_count = len(re.split(r'[.!?]+', text))
        
        # Calculate dimension scores
        coherence_score = self.coherence_analyzer.analyze_coherence(text)
        accuracy_score = self.accuracy_checker.check_accuracy(text)
        completeness_score = self._calculate_completeness(text)
        clarity_score = self._calculate_clarity(text)
        relevance_score = self._calculate_relevance(text, domain)
        originality_score = self._calculate_originality(text)
        structure_score = self._calculate_structure(text)
        
        # Calculate BLEU score if references available
        bleu_score = None
        if reference_texts:
            bleu_score = self.bleu_calculator.calculate_bleu(text, reference_texts)
        elif domain in self.reference_texts:
            ref_texts = [rt.text for rt in self.reference_texts[domain]]
            bleu_score = self.bleu_calculator.calculate_bleu(text, ref_texts)
        
        # Create dimension scores dictionary
        dimension_scores = {
            QualityDimension.COHERENCE: coherence_score,
            QualityDimension.ACCURACY: accuracy_score,
            QualityDimension.COMPLETENESS: completeness_score,
            QualityDimension.CLARITY: clarity_score,
            QualityDimension.RELEVANCE: relevance_score,
            QualityDimension.ORIGINALITY: originality_score,
            QualityDimension.STRUCTURE: structure_score,
        }
        
        # Calculate overall score with domain-specific weighting
        overall_score = self._calculate_overall_score(dimension_scores, domain, bleu_score)
        
        return QualityScore(
            overall_score=overall_score,
            dimension_scores=dimension_scores,
            bleu_score=bleu_score,
            coherence_score=coherence_score,
            accuracy_score=accuracy_score,
            completeness_score=completeness_score,
            clarity_score=clarity_score,
            relevance_score=relevance_score,
            originality_score=originality_score,
            structure_score=structure_score,
            word_count=word_count,
            sentence_count=sentence_count,
            domain=domain
        )
    
    def _calculate_completeness(self, text: str) -> float:
        """Calculate completeness based on text structure and coverage."""
        # Check for key structural elements
        has_introduction = bool(re.search(r'\b(introduction|overview|summary|background)\b', text.lower()))
        has_main_content = len(text.split()) > 50  # Substantial content
        has_conclusion = bool(re.search(r'\b(conclusion|summary|therefore|thus|overall)\b', text.lower()))
        
        # Check for development patterns
        has_examples = bool(re.search(r'\b(example|for instance|such as|specifically)\b', text.lower()))
        has_explanation = bool(re.search(r'\b(because|therefore|thus|since|due to)\b', text.lower()))
        
        completeness_indicators = [
            has_introduction,
            has_main_content,
            has_conclusion,
            has_examples,
            has_explanation
        ]
        
        return sum(completeness_indicators) / len(completeness_indicators)
    
    def _calculate_clarity(self, text: str) -> float:
        """Calculate clarity based on readability and language complexity."""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return 0.0
        
        # Calculate average sentence length
        sentence_lengths = [len(sentence.split()) for sentence in sentences]
        avg_sentence_length = statistics.mean(sentence_lengths)
        
        # Optimal sentence length is 15-20 words
        if avg_sentence_length <= 5:
            length_score = 0.6  # Too short
        elif avg_sentence_length <= 20:
            length_score = 1.0
        else:
            length_score = max(0.4, 1.0 - (avg_sentence_length - 20) / 30)  # Too long
        
        # Check for clear language patterns
        clear_patterns = [
            r'\b(first|second|third|finally)\b',  # Enumeration
            r'\b(in other words|that is|specifically)\b',  # Clarification
            r'\b(this means|therefore|thus)\b',  # Explanation
        ]
        
        clear_indicators = 0
        for pattern in clear_patterns:
            clear_indicators += len(re.findall(pattern, text.lower()))
        
        word_count = len(text.split())
        clarity_ratio = clear_indicators / word_count if word_count > 0 else 0.0
        
        # Combine length and clarity indicators
        return 0.7 * length_score + 0.3 * min(1.0, clarity_ratio * 50)
    
    def _calculate_relevance(self, text: str, domain: DomainType) -> float:
        """Calculate relevance based on domain-specific keywords."""
        domain_keywords = {
            DomainType.TECHNICAL: [
                "algorithm", "implementation", "system", "architecture", "framework",
                "code", "design", "technical", "engineering", "development"
            ],
            DomainType.BUSINESS: [
                "market", "revenue", "cost", "strategy", "business", "analysis",
                "performance", "efficiency", "optimization", "competitive"
            ],
            DomainType.CREATIVE: [
                "creative", "design", "aesthetic", "artistic", "visual", "innovative",
                "original", "inspiration", "concept", "imagination"
            ],
            DomainType.ANALYTICAL: [
                "analysis", "data", "statistics", "metrics", "measurement", "evaluation",
                "assessment", "research", "findings", "results"
            ],
            DomainType.SCIENTIFIC: [
                "research", "study", "experiment", "hypothesis", "methodology", "findings",
                "evidence", "theory", "validation", "scientific"
            ],
            DomainType.GENERAL: []  # No specific keywords for general domain
        }
        
        if domain == DomainType.GENERAL:
            return 0.8  # Neutral score for general content
        
        keywords = domain_keywords.get(domain, [])
        if not keywords:
            return 0.5
        
        text_lower = text.lower()
        keyword_matches = sum(1 for keyword in keywords if keyword in text_lower)
        
        relevance_score = keyword_matches / len(keywords)
        return min(1.0, relevance_score * 2)  # Boost relevance scoring
    
    def _calculate_originality(self, text: str) -> float:
        """Calculate originality based on unique patterns and expressions."""
        # Check for formulaic vs creative language
        formulaic_patterns = [
            r'\b(in conclusion|to summarize|first of all|second of all)\b',
            r'\b(it is important to|it should be noted|it is worth mentioning)\b',
            r'\b(there are many|there are several|there are various)\b'
        ]
        
        creative_patterns = [
            r'\b(imagine|envision|picture|consider)\b',
            r'\b(innovative|novel|unique|unprecedented|groundbreaking)\b',
            r'\b(surprisingly|remarkably|interestingly|notably)\b'
        ]
        
        formulaic_count = 0
        for pattern in formulaic_patterns:
            formulaic_count += len(re.findall(pattern, text.lower()))
        
        creative_count = 0
        for pattern in creative_patterns:
            creative_count += len(re.findall(pattern, text.lower()))
        
        word_count = len(text.split())
        if word_count == 0:
            return 0.5
        
        # Calculate originality ratio
        formulaic_ratio = formulaic_count / word_count
        creative_ratio = creative_count / word_count
        
        originality_score = 0.7 - formulaic_ratio * 10 + creative_ratio * 15
        return min(1.0, max(0.0, originality_score))
    
    def _calculate_structure(self, text: str) -> float:
        """Calculate structural quality of the text."""
        # Check for paragraph structure (approximate with double newlines)
        paragraphs = text.split('\n\n')
        paragraph_count = len([p for p in paragraphs if p.strip()])
        
        # Check for hierarchical structure
        has_headers = bool(re.search(r'^(#|\*\*|[A-Z][A-Z\s]+:)', text, re.MULTILINE))
        has_bullet_points = bool(re.search(r'^\s*[-*•]\s', text, re.MULTILINE))
        has_numbering = bool(re.search(r'^\s*\d+\.\s', text, re.MULTILINE))
        
        # Check for logical flow indicators
        flow_indicators = [
            r'\b(first|second|third|next|then|finally)\b',
            r'\b(furthermore|moreover|additionally|also)\b',
            r'\b(however|but|although|despite)\b'
        ]
        
        flow_count = 0
        for pattern in flow_indicators:
            flow_count += len(re.findall(pattern, text.lower()))
        
        # Calculate structure score
        structure_elements = [
            paragraph_count > 1,  # Multiple paragraphs
            has_headers,
            has_bullet_points or has_numbering,
            flow_count > 0
        ]
        
        return sum(structure_elements) / len(structure_elements)
    
    def _calculate_overall_score(self, dimension_scores: Dict[QualityDimension, float],
                               domain: DomainType, bleu_score: Optional[float]) -> float:
        """Calculate weighted overall score based on domain priorities."""
        # Domain-specific weights
        domain_weights = {
            DomainType.TECHNICAL: {
                QualityDimension.ACCURACY: 0.25,
                QualityDimension.CLARITY: 0.20,
                QualityDimension.STRUCTURE: 0.15,
                QualityDimension.COHERENCE: 0.15,
                QualityDimension.COMPLETENESS: 0.10,
                QualityDimension.RELEVANCE: 0.10,
                QualityDimension.ORIGINALITY: 0.05,
            },
            DomainType.BUSINESS: {
                QualityDimension.RELEVANCE: 0.25,
                QualityDimension.CLARITY: 0.20,
                QualityDimension.ACCURACY: 0.15,
                QualityDimension.STRUCTURE: 0.15,
                QualityDimension.COMPLETENESS: 0.10,
                QualityDimension.COHERENCE: 0.10,
                QualityDimension.ORIGINALITY: 0.05,
            },
            DomainType.CREATIVE: {
                QualityDimension.ORIGINALITY: 0.30,
                QualityDimension.CLARITY: 0.20,
                QualityDimension.COHERENCE: 0.15,
                QualityDimension.STRUCTURE: 0.10,
                QualityDimension.RELEVANCE: 0.10,
                QualityDimension.COMPLETENESS: 0.10,
                QualityDimension.ACCURACY: 0.05,
            },
            DomainType.ANALYTICAL: {
                QualityDimension.ACCURACY: 0.30,
                QualityDimension.STRUCTURE: 0.20,
                QualityDimension.COHERENCE: 0.15,
                QualityDimension.COMPLETENESS: 0.15,
                QualityDimension.CLARITY: 0.10,
                QualityDimension.RELEVANCE: 0.05,
                QualityDimension.ORIGINALITY: 0.05,
            },
            DomainType.SCIENTIFIC: {
                QualityDimension.ACCURACY: 0.35,
                QualityDimension.STRUCTURE: 0.20,
                QualityDimension.COMPLETENESS: 0.15,
                QualityDimension.COHERENCE: 0.10,
                QualityDimension.CLARITY: 0.10,
                QualityDimension.RELEVANCE: 0.05,
                QualityDimension.ORIGINALITY: 0.05,
            },
            DomainType.GENERAL: {
                # Balanced weights for general content
                QualityDimension.COHERENCE: 0.20,
                QualityDimension.CLARITY: 0.20,
                QualityDimension.ACCURACY: 0.15,
                QualityDimension.COMPLETENESS: 0.15,
                QualityDimension.STRUCTURE: 0.10,
                QualityDimension.RELEVANCE: 0.10,
                QualityDimension.ORIGINALITY: 0.10,
            }
        }
        
        weights = domain_weights[domain]
        
        # Calculate weighted sum
        weighted_score = sum(
            weights[dimension] * score 
            for dimension, score in dimension_scores.items()
        )
        
        # Incorporate BLEU score if available (10% weight)
        if bleu_score is not None:
            weighted_score = 0.9 * weighted_score + 0.1 * bleu_score
        
        return min(1.0, max(0.0, weighted_score))
    
    def add_reference_text(self, reference: ReferenceText) -> None:
        """Add reference text for quality comparison."""
        if reference.domain not in self.reference_texts:
            self.reference_texts[reference.domain] = []
        self.reference_texts[reference.domain].append(reference)
    
    def batch_calculate_scores(self, texts: List[str], domain: DomainType = DomainType.GENERAL) -> List[QualityScore]:
        """Calculate quality scores for multiple texts."""
        return [self.calculate_quality_score(text, domain) for text in texts]
