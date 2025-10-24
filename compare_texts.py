#!/usr/bin/env python3
"""
Text Intensification Comparison Tool
Compares two texts for intensifying adjectives and provides detailed analysis.
"""

import spacy
import numpy as np
from collections import defaultdict, Counter
import re
from dataclasses import dataclass
from typing import List, Tuple, Dict, Set
import math
import sys


@dataclass
class IntensifierScore:
    word: str
    confidence: float
    reasons: List[str]
    context: str = ""


class TextIntensificationComparator:
    def __init__(self):
        # Load the required spaCy model with word vectors
        self.nlp = self._load_or_download_spacy_model()

        # Seed words for each intensification type
        self.seed_vectors = {
            'magnitude': ['enormous', 'massive', 'tremendous', 'substantial', 'significant'],
            'extremity': ['unprecedented', 'extraordinary', 'remarkable', 'outstanding'],
            'urgency': ['critical', 'crucial', 'vital', 'urgent', 'imperative'],
            'impact': ['groundbreaking', 'revolutionary', 'transformative', 'game-changing'],
            'emotion': ['alarming', 'stunning', 'devastating', 'compelling', 'dramatic'],
            'comprehensiveness': ['comprehensive', 'extensive', 'thorough', 'detailed', 'exhaustive']
        }

        # Initialize category_vectors
        self.category_vectors = {}

        # Build semantic vectors for each category
        self._build_semantic_vectors()

        # Thresholds
        self.semantic_threshold = 0.55  # Lowered for better recall

    def _load_or_download_spacy_model(self):
        """Load the required spaCy model or exit with instructions."""
        try:
            nlp = spacy.load("en_core_web_lg")
            print("✓ Loaded en_core_web_lg model")
            return nlp
        except OSError:
            print("❌ The spaCy model 'en_core_web_lg' is not installed.")
            print("   Please install it by running: python -m spacy download en_core_web_lg")
            sys.exit(1)

    def _build_semantic_vectors(self):
        """Build average vectors for each intensification category."""
        self.category_vectors = {}

        for category, words in self.seed_vectors.items():
            vectors = []
            for word in words:
                doc = self.nlp(word)
                if doc[0].has_vector:
                    vectors.append(doc[0].vector)

            if vectors:
                self.category_vectors[category] = np.mean(vectors, axis=0)

    def _semantic_similarity_score(self, word: str) -> Tuple[float, str]:
        """Calculate semantic similarity to intensifying categories."""
        doc = self.nlp(word)
        if not doc[0].has_vector:
            return 0.0, "no_vector"

        word_vector = doc[0].vector
        max_similarity = 0.0
        best_category = ""

        for category, category_vector in self.category_vectors.items():
            if category_vector is not None:
                similarity = np.dot(word_vector, category_vector) / (
                        np.linalg.norm(word_vector) * np.linalg.norm(category_vector)
                )
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_category = category

        return max_similarity, best_category

    def _morphological_analysis(self, word: str) -> Tuple[float, List[str]]:
        """Analyze morphological patterns indicating intensification."""
        score = 0.0
        reasons = []
        word_lower = word.lower()

        # Intensifying prefixes
        intensifying_prefixes = ['ultra', 'super', 'mega', 'hyper', 'extra', 'over', 'out']
        for prefix in intensifying_prefixes:
            if word_lower.startswith(prefix):
                score += 0.4
                reasons.append(f"intensifying_prefix_{prefix}")

        # Superlative forms
        if word_lower.endswith('est') and len(word) > 4:
            score += 0.5
            reasons.append("superlative_form")

        # Intensifying suffixes
        if word_lower.endswith('ous') or word_lower.endswith('ful'):
            score += 0.2
            reasons.append("intensifying_suffix")

        return min(score, 0.6), reasons

    def detect_intensifying_adjectives(self, text: str, min_confidence: float = 0.4) -> List[IntensifierScore]:
        """Detect intensifying adjectives in text."""
        doc = self.nlp(text)
        results = []

        for token in doc:
            if token.pos_ == "ADJ" and not token.is_stop and len(token.text) > 2:
                word = token.text.lower()
                context = token.sent.text

                # Calculate scores
                semantic_score, semantic_category = self._semantic_similarity_score(word)
                morphological_score, morphological_reasons = self._morphological_analysis(word)

                # Combine scores
                final_score = semantic_score * 0.7 + morphological_score * 0.3

                # Compile reasons
                reasons = []
                if semantic_score > self.semantic_threshold:
                    reasons.append(f"semantic_similarity_to_{semantic_category}")
                reasons.extend(morphological_reasons)

                if final_score >= min_confidence and reasons:
                    results.append(IntensifierScore(
                        word=token.text,
                        confidence=round(final_score, 3),
                        reasons=reasons,
                        context=context.strip()
                    ))

        # Remove duplicates, keep highest confidence
        seen_words = {}
        for result in results:
            word_lower = result.word.lower()
            if word_lower not in seen_words or result.confidence > seen_words[word_lower].confidence:
                seen_words[word_lower] = result

        return sorted(seen_words.values(), key=lambda x: x.confidence, reverse=True)

    def analyze_text(self, text: str, label: str = "") -> Dict:
        """Comprehensive analysis of a single text."""
        doc = self.nlp(text)

        # Get all adjectives and nouns
        all_adjectives = [token.text for token in doc if token.pos_ == "ADJ"]
        all_nouns = [token.text for token in doc if token.pos_ == "NOUN"]

        # Get intensifying adjectives
        intensifying_adj = self.detect_intensifying_adjectives(text)

        # Find adjective-noun pairs
        intensified_pairs = []
        for token in doc:
            if token.pos_ == "NOUN":
                modifiers = [child for child in token.children if child.pos_ == "ADJ"]
                for modifier in modifiers:
                    if any(adj.word.lower() == modifier.text.lower() for adj in intensifying_adj):
                        intensified_pairs.append(f"{modifier.text} {token.text}")

        # Calculate rates
        total_adjectives = len(all_adjectives)
        intensifying_count = len(intensifying_adj)
        total_nouns = len(all_nouns)
        intensified_noun_count = len(intensified_pairs)

        adj_rate = (intensifying_count / total_adjectives * 100) if total_adjectives > 0 else 0
        noun_rate = (intensified_noun_count / total_nouns * 100) if total_nouns > 0 else 0

        return {
            'label': label,
            'word_count': len([token for token in doc if not token.is_space]),
            'sentence_count': len(list(doc.sents)),
            'total_adjectives': total_adjectives,
            'intensifying_adjectives': intensifying_count,
            'adj_intensification_rate': round(adj_rate, 1),
            'total_nouns': total_nouns,
            'intensified_nouns': intensified_noun_count,
            'noun_intensification_rate': round(noun_rate, 1),
            'intensifying_words': [adj.word for adj in intensifying_adj],
            'intensified_pairs': intensified_pairs,
            'confidence_scores': {adj.word: adj.confidence for adj in intensifying_adj},
            'detailed_intensifiers': intensifying_adj
        }

    def compare_texts(self, text1: str, text2: str, label1: str = "Text 1", label2: str = "Text 2"):
        """Compare two texts for intensification patterns."""
        print("\n" + "=" * 70)
        print("🔍 TEXT INTENSIFICATION COMPARISON ANALYSIS")
        print("=" * 70)

        # Analyze both texts
        analysis1 = self.analyze_text(text1, label1)
        analysis2 = self.analyze_text(text2, label2)

        # Print basic stats
        print(f"\n📊 BASIC STATISTICS")
        print("-" * 40)
        print(f"{label1:15} | Words: {analysis1['word_count']:4} | Sentences: {analysis1['sentence_count']:2}")
        print(f"{label2:15} | Words: {analysis2['word_count']:4} | Sentences: {analysis2['sentence_count']:2}")

        # Print intensification rates
        print(f"\n🎯 INTENSIFICATION RATES")
        print("-" * 40)
        print(f"{'Metric':<25} | {label1:<12} | {label2:<12} | Difference")
        print("-" * 65)

        adj_diff = analysis1['adj_intensification_rate'] - analysis2['adj_intensification_rate']
        noun_diff = analysis1['noun_intensification_rate'] - analysis2['noun_intensification_rate']

        print(
            f"{'Adjective Rate':<25} | {analysis1['adj_intensification_rate']:>10.1f}% | {analysis2['adj_intensification_rate']:>10.1f}% | {adj_diff:>+7.1f}%")
        print(
            f"{'Noun Intensification':<25} | {analysis1['noun_intensification_rate']:>10.1f}% | {analysis2['noun_intensification_rate']:>10.1f}% | {noun_diff:>+7.1f}%")

        # Detailed intensifier breakdown
        print(f"\n🔎 INTENSIFYING ADJECTIVES FOUND")
        print("-" * 40)

        max_intensifiers = max(len(analysis1['detailed_intensifiers']), len(analysis2['detailed_intensifiers']))

        if max_intensifiers > 0:
            print(f"{'Rank':<4} | {label1:<20} | {label2:<20}")
            print("-" * 50)

            for i in range(max_intensifiers):
                word1 = analysis1['detailed_intensifiers'][i].word if i < len(
                    analysis1['detailed_intensifiers']) else ""
                conf1 = f"({analysis1['detailed_intensifiers'][i].confidence:.2f})" if i < len(
                    analysis1['detailed_intensifiers']) else ""

                word2 = analysis2['detailed_intensifiers'][i].word if i < len(
                    analysis2['detailed_intensifiers']) else ""
                conf2 = f"({analysis2['detailed_intensifiers'][i].confidence:.2f})" if i < len(
                    analysis2['detailed_intensifiers']) else ""

                print(f"{i + 1:<4} | {word1:<12} {conf1:<8} | {word2:<12} {conf2:<8}")
        else:
            print("No intensifying adjectives detected in either text.")

        # Show intensified noun pairs
        print(f"\n🎭 INTENSIFIED NOUN PAIRS")
        print("-" * 40)

        if analysis1['intensified_pairs'] or analysis2['intensified_pairs']:
            print(
                f"{label1}: {', '.join(analysis1['intensified_pairs']) if analysis1['intensified_pairs'] else 'None'}")
            print(
                f"{label2}: {', '.join(analysis2['intensified_pairs']) if analysis2['intensified_pairs'] else 'None'}")
        else:
            print("No intensified noun pairs found in either text.")

        # Interpretation
        print(f"\n🤖 AI LIKELIHOOD ASSESSMENT")
        print("-" * 40)

        def get_ai_likelihood(noun_rate, adj_rate):
            combined_score = noun_rate * 2 + adj_rate
            if combined_score >= 50:
                return "VERY HIGH - Likely AI"
            elif combined_score >= 30:
                return "HIGH - Possibly AI"
            elif combined_score >= 15:
                return "MODERATE - Mixed signals"
            else:
                return "LOW - Likely human"

        likelihood1 = get_ai_likelihood(analysis1['noun_intensification_rate'], analysis1['adj_intensification_rate'])
        likelihood2 = get_ai_likelihood(analysis2['noun_intensification_rate'], analysis2['adj_intensification_rate'])

        print(f"{label1}: {likelihood1}")
        print(f"{label2}: {likelihood2}")

        # Winner declaration
        print(f"\n🏆 COMPARISON SUMMARY")
        print("-" * 40)

        if noun_diff > 5:
            print(f"'{label1}' shows significantly more intensification ({noun_diff:+.1f}% difference)")
        elif noun_diff < -5:
            print(f"'{label2}' shows significantly more intensification ({-noun_diff:+.1f}% difference)")
        else:
            print("Both texts show similar levels of intensification")

        return analysis1, analysis2


def get_multiline_input(prompt: str) -> str:
    """Get multiline input from user."""
    print(prompt)
    print("(Press Enter twice when finished, or Ctrl+C to quit)")
    lines = []
    empty_lines = 0

    try:
        while True:
            line = input()
            if line.strip() == "":
                empty_lines += 1
                if empty_lines >= 2:
                    break
            else:
                empty_lines = 0
                lines.append(line)
        return "\n".join(lines)
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
        exit()


def main():
    """Main interactive comparison tool."""
    print("🎯 TEXT INTENSIFICATION COMPARISON TOOL")
    print("=" * 50)
    print("This tool compares two texts for intensifying adjectives")
    print("- useful for detecting AI-generated content patterns")
    print("- analyzes semantic and morphological intensification")
    print()

    # Initialize detector
    print("Loading language model...")
    try:
        comparator = TextIntensificationComparator()
    except Exception as e:
        print(f"Error initializing: {e}")
        return

    print("✓ Ready for analysis!\n")

    while True:
        try:
            # Get first text
            text1 = get_multiline_input("\n📝 Enter TEXT 1:")
            if not text1.strip():
                print("Empty text entered. Please try again.")
                continue

            label1 = input("\n🏷️  Label for Text 1 (optional, press Enter for 'Text 1'): ").strip()
            if not label1:
                label1 = "Text 1"

            # Get second text
            text2 = get_multiline_input("\n📝 Enter TEXT 2:")
            if not text2.strip():
                print("Empty text entered. Please try again.")
                continue

            label2 = input("\n🏷️  Label for Text 2 (optional, press Enter for 'Text 2'): ").strip()
            if not label2:
                label2 = "Text 2"

            # Perform comparison
            print("\n🔄 Analyzing texts...")
            comparator.compare_texts(text1, text2, label1, label2)

            # Ask if user wants to continue
            print("\n" + "=" * 70)
            choice = input("\nAnalyze more texts? (y/n): ").strip().lower()
            if choice not in ['y', 'yes']:
                break

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error during analysis: {e}")
            print("Please try again with different text.")


if __name__ == "__main__":
    main()