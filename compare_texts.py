#!/usr/bin/env python3
"""
Text Intensification Comparison Tool
Compares two texts for intensifying adjectives and provides detailed analysis.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass
import math
import re
import subprocess
import sys
from typing import Dict, List, Set, Tuple

import numpy as np
import spacy
import subprocess


@dataclass
class IntensifierScore:
    word: str
    confidence: float
    reasons: List[str]
    context: str = ""
    part_of_speech: str = ""


class TextIntensificationComparator:
    def __init__(self):
        # Load the required spaCy model with word vectors
        self.nlp = self._load_or_download_spacy_model()

        # Seed words for each intensification type grouped by part of speech
        self.seed_vectors = {
            'ADJ': {
                'magnitude': ['enormous', 'massive', 'tremendous', 'substantial', 'significant'],
                'extremity': ['unprecedented', 'extraordinary', 'remarkable', 'outstanding'],
                'urgency': ['critical', 'crucial', 'vital', 'urgent', 'imperative'],
                'impact': ['groundbreaking', 'revolutionary', 'transformative', 'game-changing'],
                'emotion': ['alarming', 'stunning', 'devastating', 'compelling', 'dramatic'],
                'comprehensiveness': ['comprehensive', 'extensive', 'thorough', 'detailed', 'exhaustive']
            },
            'ADV': {
                'intensity': ['extremely', 'incredibly', 'exceptionally', 'immensely', 'tremendously'],
                'degree': ['highly', 'deeply', 'greatly', 'hugely', 'vastly'],
                'impact': ['significantly', 'dramatically', 'substantially', 'profoundly', 'severely'],
                'urgency': ['urgently', 'critically', 'desperately', 'pressingly', 'vitally'],
                'certainty': ['undoubtedly', 'certainly', 'definitely', 'unequivocally', 'absolutely']
            }
        }

        # Initialize category_vectors
        self.category_vectors = {'ADJ': {}, 'ADV': {}}

        # Build semantic vectors for each category
        self._build_semantic_vectors()

        # Thresholds
        self.semantic_threshold = 0.55  # Lowered for better recall

    def _load_or_download_spacy_model(self):
        """Load the required spaCy model, falling back to smaller models if needed."""
        try:
            nlp = spacy.load("en_core_web_lg")
            print("✓ Loaded en_core_web_lg model")
            return nlp
        except OSError:
            print("⚠️ The spaCy model 'en_core_web_lg' is not installed.")
            print("   Attempting to load the smaller 'en_core_web_sm' model instead.")
            print("   For best results, install the large model with: python -m spacy download en_core_web_lg")
            print("⚠️  en_core_web_lg not found, trying en_core_web_sm...")

        # Try to load the small model
        try:
            nlp = spacy.load("en_core_web_sm")
            print("✓ Loaded en_core_web_sm model (reduced accuracy)")
            print("💡 For better results, run: python -m spacy download en_core_web_lg")
            return nlp
        except OSError:
            print("📥 No spaCy model found. Downloading en_core_web_sm automatically...")

            try:
                # Download the small model automatically
                subprocess.check_call([
                    sys.executable, "-m", "spacy", "download", "en_core_web_sm"
                ], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

                print("✅ Successfully downloaded en_core_web_sm!")
                nlp = spacy.load("en_core_web_sm")
                print("✓ Model loaded and ready to use")
                return nlp

            except subprocess.CalledProcessError as e:
                print("❌ Failed to download spaCy model automatically.")
                print("Please install manually with:")
                print("  pip install spacy")
                print("  python -m spacy download en_core_web_sm")
                raise RuntimeError("Could not load or download spaCy model") from e
            except Exception as e:
                print(f"❌ Unexpected error: {e}")
                print("Please install spaCy model manually:")
                print("  pip install spacy")
                print("  python -m spacy download en_core_web_sm")
                raise

        # Seed words for each intensification type grouped by part of speech
        self.seed_vectors = {
            'ADJ': {
                'magnitude': ['enormous', 'massive', 'tremendous', 'substantial', 'significant'],
                'extremity': ['unprecedented', 'extraordinary', 'remarkable', 'outstanding'],
                'urgency': ['critical', 'crucial', 'vital', 'urgent', 'imperative'],
                'impact': ['groundbreaking', 'revolutionary', 'transformative', 'game-changing'],
                'emotion': ['alarming', 'stunning', 'devastating', 'compelling', 'dramatic'],
                'comprehensiveness': ['comprehensive', 'extensive', 'thorough', 'detailed', 'exhaustive']
            },
            'ADV': {
                'intensity': ['extremely', 'incredibly', 'exceptionally', 'immensely', 'tremendously'],
                'degree': ['highly', 'deeply', 'greatly', 'hugely', 'vastly'],
                'impact': ['significantly', 'dramatically', 'substantially', 'profoundly', 'severely'],
                'urgency': ['urgently', 'critically', 'desperately', 'pressingly', 'vitally'],
                'certainty': ['undoubtedly', 'certainly', 'definitely', 'unequivocally', 'absolutely']
            }
        }

        # Build semantic vectors for each category
        self._build_semantic_vectors()

        # Thresholds
        self.semantic_threshold = 0.55  # Lowered for better recall

    def _build_semantic_vectors(self):
        """Build average vectors for each intensification category."""
        self.category_vectors = {'ADJ': {}, 'ADV': {}}

        for pos_tag, categories in self.seed_vectors.items():
            for category, words in categories.items():
                vectors = []
                for word in words:
                    doc = self.nlp(word)
                    if doc[0].has_vector:
                        vectors.append(doc[0].vector)

                if vectors:
                    self.category_vectors[pos_tag][category] = np.mean(vectors, axis=0)

    def _semantic_similarity_score(self, word: str, pos_tag: str) -> Tuple[float, str]:
        """Calculate semantic similarity to intensifying categories."""
        doc = self.nlp(word)
        if not doc[0].has_vector:
            return 0.0, "no_vector"

        word_vector = doc[0].vector
        max_similarity = 0.0
        best_category = ""

        category_vectors = self.category_vectors.get(pos_tag, {})

        for category, category_vector in category_vectors.items():
            if category_vector is not None:
                similarity = np.dot(word_vector, category_vector) / (
                        np.linalg.norm(word_vector) * np.linalg.norm(category_vector)
                )
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_category = category

        return max_similarity, best_category

    def _morphological_analysis(self, word: str, pos_tag: str) -> Tuple[float, List[str]]:
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

        if pos_tag == "ADJ":
            # Superlative forms
            if word_lower.endswith('est') and len(word) > 4:
                score += 0.5
                reasons.append("superlative_form")

            # Intensifying suffixes
            if word_lower.endswith('ous') or word_lower.endswith('ful'):
                score += 0.2
                reasons.append("intensifying_suffix")

        if pos_tag == "ADV":
            intensifying_adverbs = {
                'extremely', 'incredibly', 'exceptionally', 'immensely', 'tremendously',
                'highly', 'deeply', 'greatly', 'hugely', 'vastly', 'remarkably',
                'severely', 'seriously', 'critically', 'urgently', 'vitally',
                'profoundly', 'drastically', 'overwhelmingly', 'strongly', 'absolutely',
                'certainly', 'undoubtedly', 'definitely'
            }

            if word_lower in intensifying_adverbs:
                score += 0.5
                reasons.append("known_intensifying_adverb")

            if word_lower.endswith('ly') and len(word) > 4:
                score += 0.2
                reasons.append("adverbial_intensifier_suffix")

        return min(score, 0.6), reasons

    def detect_intensifying_adjectives(self, text: str, min_confidence: float = 0.4) -> List[IntensifierScore]:
        """Detect intensifying adjectives and adverbs in text."""
        doc = self.nlp(text)
        results = []

        for token in doc:
            if token.pos_ in {"ADJ", "ADV"} and not token.is_stop and len(token.text) > 2:
                word = token.text.lower()
                context = token.sent.text

                # Calculate scores
                semantic_score, semantic_category = self._semantic_similarity_score(word, token.pos_)
                morphological_score, morphological_reasons = self._morphological_analysis(word, token.pos_)

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
                        context=context.strip(),
                        part_of_speech=token.pos_
                    ))

        # Remove duplicates, keep highest confidence
        seen_words = {}
        for result in results:
            word_key = (result.word.lower(), result.part_of_speech)
            if word_key not in seen_words or result.confidence > seen_words[word_key].confidence:
                seen_words[word_key] = result

        return sorted(seen_words.values(), key=lambda x: x.confidence, reverse=True)

    def analyze_text(self, text: str, label: str = "") -> Dict:
        """Comprehensive analysis of a single text."""
        doc = self.nlp(text)

        # Get all adjectives, adverbs, and nouns
        all_adjectives = [token.text for token in doc if token.pos_ == "ADJ"]
        all_adverbs = [token.text for token in doc if token.pos_ == "ADV"]
        all_nouns = [token.text for token in doc if token.pos_ == "NOUN"]

        # Get intensifying modifiers
        intensifying_scores = self.detect_intensifying_adjectives(text)
        intensifying_adjectives = [score for score in intensifying_scores if score.part_of_speech == "ADJ"]
        intensifying_adverbs = [score for score in intensifying_scores if score.part_of_speech == "ADV"]

        # Find adjective-noun pairs
        intensified_pairs = []
        for token in doc:
            if token.pos_ == "NOUN":
                modifiers = [child for child in token.children if child.pos_ == "ADJ"]
                for modifier in modifiers:
                    if any(adj.word.lower() == modifier.text.lower() for adj in intensifying_adjectives):
                        intensified_pairs.append(f"{modifier.text} {token.text}")

        # Calculate rates
        total_adjectives = len(all_adjectives)
        total_adverbs = len(all_adverbs)
        intensifying_adj_count = len(intensifying_adjectives)
        intensifying_adv_count = len(intensifying_adverbs)
        total_nouns = len(all_nouns)
        intensified_noun_count = len(intensified_pairs)

        adj_rate = (intensifying_adj_count / total_adjectives * 100) if total_adjectives > 0 else 0
        adv_rate = (intensifying_adv_count / total_adverbs * 100) if total_adverbs > 0 else 0
        noun_rate = (intensified_noun_count / total_nouns * 100) if total_nouns > 0 else 0

        return {
            'label': label,
            'word_count': len([token for token in doc if not token.is_space]),
            'sentence_count': len(list(doc.sents)),
            'total_adjectives': total_adjectives,
            'total_adverbs': total_adverbs,
            'intensifying_adjectives': intensifying_adj_count,
            'intensifying_adverbs': intensifying_adv_count,
            'adj_intensification_rate': round(adj_rate, 1),
            'adv_intensification_rate': round(adv_rate, 1),
            'total_nouns': total_nouns,
            'intensified_nouns': intensified_noun_count,
            'noun_intensification_rate': round(noun_rate, 1),
            'intensifying_words': [score.word for score in intensifying_scores],
            'intensified_pairs': intensified_pairs,
            'confidence_scores': {score.word: score.confidence for score in intensifying_scores},
            'detailed_intensifiers': intensifying_scores
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
        adv_diff = analysis1['adv_intensification_rate'] - analysis2['adv_intensification_rate']
        noun_diff = analysis1['noun_intensification_rate'] - analysis2['noun_intensification_rate']

        print(
            f"{'Adjective Rate':<25} | {analysis1['adj_intensification_rate']:>10.1f}% | {analysis2['adj_intensification_rate']:>10.1f}% | {adj_diff:>+7.1f}%")
        print(
            f"{'Adverb Rate':<25} | {analysis1['adv_intensification_rate']:>10.1f}% | {analysis2['adv_intensification_rate']:>10.1f}% | {adv_diff:>+7.1f}%")
        print(
            f"{'Noun Intensification':<25} | {analysis1['noun_intensification_rate']:>10.1f}% | {analysis2['noun_intensification_rate']:>10.1f}% | {noun_diff:>+7.1f}%")

        # Detailed intensifier breakdown
        print(f"\n🔎 INTENSIFYING MODIFIERS FOUND")
        print("-" * 40)

        max_intensifiers = max(len(analysis1['detailed_intensifiers']), len(analysis2['detailed_intensifiers']))

        if max_intensifiers > 0:
            print(f"{'Rank':<4} | {label1:<20} | {label2:<20}")
            print("-" * 50)

            for i in range(max_intensifiers):
                if i < len(analysis1['detailed_intensifiers']):
                    score1 = analysis1['detailed_intensifiers'][i]
                    word1 = f"{score1.word} [{score1.part_of_speech}]"
                    conf1 = f"({score1.confidence:.2f})"
                else:
                    word1 = ""
                    conf1 = ""

                if i < len(analysis2['detailed_intensifiers']):
                    score2 = analysis2['detailed_intensifiers'][i]
                    word2 = f"{score2.word} [{score2.part_of_speech}]"
                    conf2 = f"({score2.confidence:.2f})"
                else:
                    word2 = ""
                    conf2 = ""

                print(f"{i + 1:<4} | {word1:<20} {conf1:<8} | {word2:<20} {conf2:<8}")
        else:
            print("No intensifying modifiers detected in either text.")

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
    print("This tool compares two texts for intensifying adjectives and adverbs")
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
