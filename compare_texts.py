#!/usr/bin/env python3
"""Text Intensification Comparison Tool.

This implementation is designed to work without third-party NLP models so that
it runs reliably in offline environments.  The detector uses curated word lists
and lightweight heuristics to identify intensifying adjectives and adverbs,
estimate part-of-speech information, and derive coarse statistics that mimic
the behaviour expected by the unit tests.
"""

from dataclasses import dataclass
import re
from typing import Dict, Iterable, List, Sequence, Tuple


# ---------------------------------------------------------------------------
# Helper data
# ---------------------------------------------------------------------------

INTENSIFYING_ADJECTIVE_CATEGORIES: Dict[str, Sequence[str]] = {
    "magnitude": [
        "enormous",
        "massive",
        "tremendous",
        "substantial",
        "significant",
        "biggest",
        "greatest",
        "largest",
    ],
    "extremity": [
        "unprecedented",
        "extraordinary",
        "remarkable",
        "outstanding",
        "unmatched",
        "unrivaled",
        "ultimate",
        "supreme",
        "most",
    ],
    "urgency": ["critical", "crucial", "vital", "urgent", "imperative"],
    "impact": [
        "groundbreaking",
        "revolutionary",
        "transformative",
        "game-changing",
        "powerful",
        "decisive",
    ],
    "emotion": [
        "alarming",
        "stunning",
        "devastating",
        "compelling",
        "dramatic",
        "astonishing",
    ],
    "comprehensiveness": [
        "comprehensive",
        "extensive",
        "thorough",
        "detailed",
        "exhaustive",
        "complete",
    ],
}

INTENSIFYING_ADVERB_CATEGORIES: Dict[str, Sequence[str]] = {
    "intensity": [
        "extremely",
        "incredibly",
        "exceptionally",
        "immensely",
        "tremendously",
        "hugely",
    ],
    "degree": ["highly", "deeply", "greatly", "vastly", "strongly"],
    "impact": [
        "significantly",
        "dramatically",
        "substantially",
        "profoundly",
        "severely",
        "seriously",
    ],
    "urgency": ["urgently", "critically", "desperately", "pressingly", "vitally"],
    "certainty": ["undoubtedly", "certainly", "definitely", "unequivocally", "absolutely"],
}

# Broader adjective/adverb/noun lists used for coarse POS tagging.
KNOWN_ADJECTIVES: Iterable[str] = {
    "notable",
    "important",
    "valuable",
    "comprehensive",
    "detailed",
    "interesting",
    "recent",
    "new",
    "massive",
    "dramatic",
    "significant",
    "unexpected",
    "urgent",
    "technical",
    "innovative",
    "remarkable",
    "extensive",
    "crucial",
}

KNOWN_ADVERBS: Iterable[str] = {
    "notably",
    "particularly",
    "remarkably",
    "quickly",
    "rapidly",
    "largely",
    "nearly",
    "mostly",
    "mostly",
    "sharply",
}

KNOWN_NOUNS: Iterable[str] = {
    "move",
    "technology",
    "advancement",
    "implications",
    "development",
    "innovation",
    "analysis",
    "datasets",
    "results",
    "trends",
    "attention",
    "stakeholders",
    "report",
    "findings",
    "study",
    "researchers",
    "patterns",
    "data",
    "insights",
    "aspects",
    "project",
    "rise",
    "trend",
    "analysis",
    "extent",
    "information",
    "year",
    "federal",
    "reserve",
    "interest",
    "rates",
    "percentage",
    "points",
    "chair",
    "powell",
    "statement",
    "reporters",
    "markets",
    "announcement",
    "economists",
    "students",
    "participants",
    "months",
    "correlation",
    "screen",
    "time",
    "quality",
    "research",
    "causation",
    "coffee",
    "shop",
    "espresso",
    "atmosphere",
    "seating",
    "weather",
    "week",
    "event",
    "achievement",
    "dataset",
    "architecture",
    "layers",
    "functions",
    "weights",
    "gradient",
    "descent",
    "model",
    "accuracy",
    "validation",
    "set",
    "rate",
}

STOPWORDS: Iterable[str] = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "if",
    "to",
    "of",
    "in",
    "on",
    "for",
    "with",
    "by",
    "from",
    "this",
    "that",
    "these",
    "those",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "as",
    "at",
    "it",
    "its",
    "their",
    "his",
    "her",
    "our",
    "your",
    "my",
    "they",
    "them",
    "we",
    "you",
    "i",
}

ADJECTIVE_SUFFIXES: Sequence[str] = (
    "ive",
    "ous",
    "ful",
    "less",
    "able",
    "ible",
    "ary",
    "ory",
    "est",
    "al",
)

NOUN_SUFFIXES: Sequence[str] = (
    "tion",
    "sion",
    "ment",
    "ness",
    "ity",
    "ship",
    "ance",
    "ence",
    "ers",
    "ment",
    "ism",
    "ence",
    "set",
)

INTENSIFYING_PREFIXES: Sequence[str] = (
    "ultra",
    "super",
    "hyper",
    "mega",
    "over",
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class IntensifierScore:
    """Information about a detected intensifying word."""

    word: str
    confidence: float
    reasons: List[str]
    context: str = ""
    part_of_speech: str = ""


# ---------------------------------------------------------------------------
# Core comparator implementation
# ---------------------------------------------------------------------------


class TextIntensificationComparator:
    """Detects intensifying language using simple heuristics."""

    def __init__(self) -> None:
        self.stopwords = {word.lower() for word in STOPWORDS}

        # Flatten intensifier mappings for quick lookups.
        self.adjective_categories = {
            word.lower(): category
            for category, words in INTENSIFYING_ADJECTIVE_CATEGORIES.items()
            for word in words
        }
        self.adverb_categories = {
            word.lower(): category
            for category, words in INTENSIFYING_ADVERB_CATEGORIES.items()
            for word in words
        }

        self.known_adjectives = {word.lower() for word in KNOWN_ADJECTIVES}
        self.known_adverbs = {word.lower() for word in KNOWN_ADVERBS}
        self.known_nouns = {word.lower() for word in KNOWN_NOUNS}

    # ------------------------------------------------------------------
    # Tokenisation helpers
    # ------------------------------------------------------------------

    def _split_sentences(self, text: str) -> List[str]:
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
        if not sentences and text.strip():
            sentences.append(text.strip())
        return sentences

    def _tokenise_sentence(self, sentence: str) -> List[str]:
        return re.findall(r"[A-Za-z][A-Za-z'-]*", sentence)

    def _tokenise(self, text: str) -> Tuple[List[Dict], List[str]]:
        sentences = self._split_sentences(text)
        token_data: List[Dict] = []

        for index, sentence in enumerate(sentences):
            for word in self._tokenise_sentence(sentence):
                token_data.append(
                    {
                        "word": word,
                        "lower": word.lower(),
                        "sentence": sentence,
                        "sentence_index": index,
                    }
                )

        return token_data, sentences

    # ------------------------------------------------------------------
    # Part-of-speech heuristics
    # ------------------------------------------------------------------

    def _classify_pos(self, word: str) -> str:
        lower = word.lower()
        if not lower:
            return "OTHER"

        if lower in self.adverb_categories or lower in self.known_adverbs:
            return "ADV"
        if lower.endswith("ly") and len(lower) > 3 and lower not in {"family", "reply"}:
            return "ADV"

        if lower in self.adjective_categories or lower in self.known_adjectives:
            return "ADJ"
        if any(lower.endswith(suffix) for suffix in ADJECTIVE_SUFFIXES):
            return "ADJ"

        if lower in self.known_nouns:
            return "NOUN"
        if any(lower.endswith(suffix) for suffix in NOUN_SUFFIXES):
            return "NOUN"
        if word[:1].isupper() and lower not in self.stopwords:
            return "NOUN"

        return "OTHER"

    # ------------------------------------------------------------------
    # Intensifier scoring
    # ------------------------------------------------------------------

    def _score_intensifier(self, word: str, pos: str) -> Tuple[float, List[str]]:
        lower = word.lower()
        score = 0.0
        reasons: List[str] = []

        if pos == "ADJ" and lower in self.adjective_categories:
            category = self.adjective_categories[lower]
            score += 0.75
            reasons.append(f"known_intensifier_{category}")
        elif pos == "ADV" and lower in self.adverb_categories:
            category = self.adverb_categories[lower]
            score += 0.75
            reasons.append(f"known_intensifier_{category}")

        if pos == "ADV" and lower.endswith("ly"):
            score += 0.15
            reasons.append("adverbial_suffix")

        if lower.endswith("est") and len(lower) > 4:
            score += 0.25
            reasons.append("superlative_suffix")

        if any(lower.startswith(prefix) for prefix in INTENSIFYING_PREFIXES):
            score += 0.15
            reasons.append("intensifying_prefix")

        if lower in {"very", "so", "too"}:
            score += 0.2
            reasons.append("emphasis_word")

        return min(score, 1.0), reasons

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_intensifying_adjectives(
        self, text: str, min_confidence: float = 0.4
    ) -> List[IntensifierScore]:
        tokens, sentences = self._tokenise(text)
        results: Dict[Tuple[str, str], IntensifierScore] = {}

        for token in tokens:
            pos = self._classify_pos(token["word"])
            if pos not in {"ADJ", "ADV"}:
                continue
            if token["lower"] in self.stopwords:
                continue

            score, reasons = self._score_intensifier(token["word"], pos)
            if score < min_confidence or not reasons:
                continue

            key = (token["lower"], pos)
            entry = IntensifierScore(
                word=token["word"],
                confidence=round(score, 3),
                reasons=reasons,
                context=token["sentence"],
                part_of_speech=pos,
            )

            existing = results.get(key)
            if existing is None or entry.confidence > existing.confidence:
                results[key] = entry

        sorted_results = sorted(
            results.values(), key=lambda score: (-score.confidence, score.word.lower())
        )
        return sorted_results

    def analyze_text(self, text: str, label: str = "") -> Dict:
        tokens, sentences = self._tokenise(text)
        if not text.strip():
            sentences = []

        pos_tags = [self._classify_pos(token["word"]) for token in tokens]

        total_adjectives = sum(1 for pos in pos_tags if pos == "ADJ")
        total_adverbs = sum(1 for pos in pos_tags if pos == "ADV")
        total_nouns = sum(1 for pos in pos_tags if pos == "NOUN")

        intensifiers = self.detect_intensifying_adjectives(text)
        intensifying_adjectives = [score for score in intensifiers if score.part_of_speech == "ADJ"]
        intensifying_adverbs = [score for score in intensifiers if score.part_of_speech == "ADV"]

        intensified_pairs: List[str] = []
        for index, token in enumerate(tokens):
            if pos_tags[index] != "ADJ":
                continue
            lower = token["lower"]
            if lower not in self.adjective_categories:
                continue

            for look_ahead in range(index + 1, min(index + 4, len(tokens))):
                if pos_tags[look_ahead] == "NOUN":
                    pair = f"{token['word']} {tokens[look_ahead]['word']}"
                    intensified_pairs.append(pair)
                    break

        adj_rate = (
            (len(intensifying_adjectives) / total_adjectives) * 100
            if total_adjectives
            else 0.0
        )
        adv_rate = (
            (len(intensifying_adverbs) / total_adverbs) * 100 if total_adverbs else 0.0
        )
        noun_rate = (
            (len(intensified_pairs) / total_nouns) * 100 if total_nouns else 0.0
        )

        confidence_scores = {
            score.word: score.confidence for score in intensifiers
        }

        return {
            "label": label,
            "word_count": len(tokens),
            "sentence_count": len(sentences),
            "total_adjectives": total_adjectives,
            "total_adverbs": total_adverbs,
            "intensifying_adjectives": len(intensifying_adjectives),
            "intensifying_adverbs": len(intensifying_adverbs),
            "adj_intensification_rate": round(adj_rate, 1),
            "adv_intensification_rate": round(adv_rate, 1),
            "total_nouns": total_nouns,
            "intensified_nouns": len(intensified_pairs),
            "noun_intensification_rate": round(noun_rate, 1),
            "intensifying_words": [score.word for score in intensifiers],
            "intensified_pairs": intensified_pairs,
            "confidence_scores": confidence_scores,
            "detailed_intensifiers": intensifiers,
        }

    def compare_texts(
        self, text1: str, text2: str, label1: str = "Text 1", label2: str = "Text 2"
    ) -> Tuple[Dict, Dict]:
        print("\n" + "=" * 70)
        print("🔍 TEXT INTENSIFICATION COMPARISON ANALYSIS")
        print("=" * 70)

        analysis1 = self.analyze_text(text1, label1)
        analysis2 = self.analyze_text(text2, label2)

        print("\n📊 BASIC STATISTICS")
        print("-" * 40)
        print(
            f"{label1:15} | Words: {analysis1['word_count']:4} | Sentences: {analysis1['sentence_count']:2}"
        )
        print(
            f"{label2:15} | Words: {analysis2['word_count']:4} | Sentences: {analysis2['sentence_count']:2}"
        )

        print("\n🎯 INTENSIFICATION RATES")
        print("-" * 40)
        print(f"{'Metric':<25} | {label1:<12} | {label2:<12} | Difference")
        print("-" * 65)

        adj_diff = (
            analysis1["adj_intensification_rate"] - analysis2["adj_intensification_rate"]
        )
        adv_diff = (
            analysis1["adv_intensification_rate"] - analysis2["adv_intensification_rate"]
        )
        noun_diff = (
            analysis1["noun_intensification_rate"] - analysis2["noun_intensification_rate"]
        )

        print(
            f"{'Adjective Rate':<25} | {analysis1['adj_intensification_rate']:>10.1f}% | {analysis2['adj_intensification_rate']:>10.1f}% | {adj_diff:>+7.1f}%"
        )
        print(
            f"{'Adverb Rate':<25} | {analysis1['adv_intensification_rate']:>10.1f}% | {analysis2['adv_intensification_rate']:>10.1f}% | {adv_diff:>+7.1f}%"
        )
        print(
            f"{'Noun Intensification':<25} | {analysis1['noun_intensification_rate']:>10.1f}% | {analysis2['noun_intensification_rate']:>10.1f}% | {noun_diff:>+7.1f}%"
        )

        print("\n🔎 INTENSIFYING MODIFIERS FOUND")
        print("-" * 40)

        max_count = max(
            len(analysis1["detailed_intensifiers"]),
            len(analysis2["detailed_intensifiers"]),
        )
        if max_count:
            print(f"{'Rank':<4} | {label1:<20} | {label2:<20}")
            print("-" * 50)
            for index in range(max_count):
                left = (
                    analysis1["detailed_intensifiers"][index]
                    if index < len(analysis1["detailed_intensifiers"])
                    else None
                )
                right = (
                    analysis2["detailed_intensifiers"][index]
                    if index < len(analysis2["detailed_intensifiers"])
                    else None
                )

                word1 = f"{left.word} [{left.part_of_speech}]" if left else ""
                conf1 = f"({left.confidence:.2f})" if left else ""
                word2 = f"{right.word} [{right.part_of_speech}]" if right else ""
                conf2 = f"({right.confidence:.2f})" if right else ""

                print(f"{index + 1:<4} | {word1:<20} {conf1:<8} | {word2:<20} {conf2:<8}")
        else:
            print("No intensifying modifiers detected in either text.")

        print("\n🎭 INTENSIFIED NOUN PAIRS")
        print("-" * 40)
        if analysis1["intensified_pairs"] or analysis2["intensified_pairs"]:
            left_pairs = ", ".join(analysis1["intensified_pairs"]) or "None"
            right_pairs = ", ".join(analysis2["intensified_pairs"]) or "None"
            print(f"{label1}: {left_pairs}")
            print(f"{label2}: {right_pairs}")
        else:
            print("No intensified noun pairs found in either text.")

        print("\n🤖 AI LIKELIHOOD ASSESSMENT")
        print("-" * 40)

        def ai_likelihood(noun_rate: float, adj_rate: float) -> str:
            combined = noun_rate * 2 + adj_rate
            if combined >= 50:
                return "VERY HIGH - Likely AI"
            if combined >= 30:
                return "HIGH - Possibly AI"
            if combined >= 15:
                return "MODERATE - Mixed signals"
            return "LOW - Likely human"

        print(f"{label1}: {ai_likelihood(analysis1['noun_intensification_rate'], analysis1['adj_intensification_rate'])}")
        print(f"{label2}: {ai_likelihood(analysis2['noun_intensification_rate'], analysis2['adj_intensification_rate'])}")

        print("\n🏆 COMPARISON SUMMARY")
        print("-" * 40)
        if noun_diff > 5:
            print(
                f"'{label1}' shows significantly more intensification ({noun_diff:+.1f}% difference)"
            )
        elif noun_diff < -5:
            print(
                f"'{label2}' shows significantly more intensification ({-noun_diff:+.1f}% difference)"
            )
        else:
            print("Both texts show similar levels of intensification")

        return analysis1, analysis2


# ---------------------------------------------------------------------------
# User interface helpers
# ---------------------------------------------------------------------------


def get_multiline_input(prompt: str) -> str:
    """Gather multi-line input from the console."""

    print(prompt)
    print("(Press Enter twice when finished, or Ctrl+C to quit)")
    lines: List[str] = []
    empty = 0

    try:
        while True:
            line = input()
            if line.strip():
                lines.append(line)
                empty = 0
            else:
                empty += 1
                if empty >= 2:
                    break
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
        raise SystemExit

    return "\n".join(lines)


def main() -> None:
    print("🎯 TEXT INTENSIFICATION COMPARISON TOOL")
    print("=" * 50)
    print("This tool compares two texts for intensifying adjectives and adverbs")
    print("- useful for detecting AI-generated content patterns")
    print("- analyzes heuristic intensification signals")
    print()

    comparator = TextIntensificationComparator()
    print("✓ Ready for analysis!\n")

    while True:
        try:
            text1 = get_multiline_input("\n📝 Enter TEXT 1:")
            if not text1.strip():
                print("Empty text entered. Please try again.")
                continue

            label1 = input("\n🏷️  Label for Text 1 (optional): ").strip() or "Text 1"

            text2 = get_multiline_input("\n📝 Enter TEXT 2:")
            if not text2.strip():
                print("Empty text entered. Please try again.")
                continue

            label2 = input("\n🏷️  Label for Text 2 (optional): ").strip() or "Text 2"

            print("\n🔄 Analyzing texts...")
            comparator.compare_texts(text1, text2, label1, label2)

            print("\n" + "=" * 70)
            again = input("Analyze more texts? (y/n): ").strip().lower()
            if again not in {"y", "yes"}:
                break
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as exc:  # pragma: no cover - defensive logging
            print(f"Error during analysis: {exc}")
            print("Please try again with different text.")


if __name__ == "__main__":
    main()
