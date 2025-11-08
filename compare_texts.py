#!/usr/bin/env python3
"""Text Intensification Comparison Tool.

This implementation is designed to work without third-party NLP models so that
it runs reliably in offline environments.  The detector uses curated word lists
and lightweight heuristics to identify intensifying adjectives and adverbs,
estimate part-of-speech information, and derive coarse statistics that mimic
the behaviour expected by the unit tests.
"""

from collections import Counter
from dataclasses import dataclass
import re
import statistics
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

PRONOUNS = {
    "i",
    "you",
    "he",
    "she",
    "it",
    "we",
    "they",
    "me",
    "him",
    "her",
    "us",
    "them",
    "my",
    "our",
    "your",
    "their",
    "mine",
    "ours",
    "yours",
    "theirs",
}

MODAL_VERBS = {
    "can",
    "could",
    "may",
    "might",
    "must",
    "shall",
    "should",
    "will",
    "would",
}

DISCOURSE_MARKERS = [
    "overall",
    "in conclusion",
    "in summary",
    "additionally",
    "moreover",
    "furthermore",
    "on the other hand",
    "however",
    "therefore",
    "consequently",
    "firstly",
    "secondly",
    "finally",
]

HEDGE_WORDS = [
    "perhaps",
    "maybe",
    "likely",
    "seems",
    "appears",
    "roughly",
    "approximately",
    "somewhat",
    "arguably",
    "reportedly",
]

CERTAINTY_WORDS = [
    "definitely",
    "certainly",
    "undoubtedly",
    "clearly",
    "surely",
    "absolutely",
    "inevitably",
]

EMOTION_WORDS = {
    "amazing",
    "shocking",
    "tragic",
    "beautiful",
    "heartbreaking",
    "exciting",
    "terrifying",
    "delightful",
    "furious",
    "thrilled",
}

IDIOM_PATTERNS = [
    "at the end of the day",
    "the tip of the iceberg",
    "a double edged sword",
    "a blessing in disguise",
    "on the other hand",
    "the fact of the matter",
]

SELF_REFERENCE_PATTERNS = [
    r"\bas an ai\b",
    r"\bas a language model\b",
    r"\bi am an ai\b",
    r"\bi am a language model\b",
    r"\bas an ai language model\b",
]

IMPERATIVE_STARTERS = [
    "please",
    "consider",
    "imagine",
    "remember",
    "note",
    "ensure",
    "let's",
    "let us",
    "make sure",
    "avoid",
]

CURRENCY_SYMBOLS = "$€£¥₹"

PASSIVE_VOICE_PATTERN = re.compile(
    r"\b(?:is|are|was|were|be|been|being|am|has been|have been|had been)\s+\w+ed\b"
)
BULLET_LINE_PATTERN = re.compile(r"\s*(?:[-*•]|[0-9]+[.)])\s+")
HEADING_LETTER_PATTERN = re.compile(r"[A-Za-z]")
NUMBER_TOKEN_PATTERN = re.compile(r"\d+(?:[.,]\d+)?")
DOUBLE_PUNCT_PATTERN = re.compile(r"[!?]{2,}")
MULTI_SPACE_PATTERN = re.compile(r"[ \t]{2,}")

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
                position = len(token_data)
                token_data.append(
                    {
                        "word": word,
                        "lower": word.lower(),
                        "sentence": sentence,
                        "sentence_index": index,
                        "token_index": position,
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

    def _estimate_syllables(self, word: str) -> int:
        cleaned = re.sub(r"[^a-z]", "", word.lower())
        if not cleaned:
            return 0
        groups = re.findall(r"[aeiouy]+", cleaned)
        syllables = len(groups)
        if cleaned.endswith("e") and syllables > 1:
            syllables -= 1
        return max(syllables, 1)

    def _count_phrases(self, text_lower: str, phrases: Sequence[str]) -> int:
        total = 0
        for phrase in phrases:
            pattern = re.compile(rf"\b{re.escape(phrase)}\b", re.IGNORECASE)
            total += len(pattern.findall(text_lower))
        return total

    def _extract_text_features(self, text: str, tokens: List[Dict], sentences: List[str]) -> Dict[str, float]:
        features: Dict[str, float] = {}
        text_lower = text.lower()
        word_count = len(tokens)

        unique_words = {token["lower"] for token in tokens}
        features["unique_word_count"] = len(unique_words)
        features["type_token_ratio"] = (
            round(len(unique_words) / word_count, 4) if word_count else 0.0
        )

        sentence_lengths = Counter()
        for token in tokens:
            sentence_lengths[token["sentence_index"]] += 1
        length_values = list(sentence_lengths.values())
        avg_sentence_len = sum(length_values) / len(length_values) if length_values else 0.0
        sentence_length_std = (
            round(statistics.pstdev(length_values), 3) if len(length_values) > 1 else 0.0
        )
        features["avg_sentence_length"] = round(avg_sentence_len, 3)
        features["sentence_length_std"] = sentence_length_std

        paragraphs = [
            paragraph.strip()
            for paragraph in re.split(r"\n\s*\n", text.strip())
            if paragraph.strip()
        ]
        paragraph_lengths = [
            len(self._tokenise_sentence(paragraph)) for paragraph in paragraphs
        ]
        features["paragraph_count"] = len(paragraphs)
        features["paragraph_length_std"] = (
            round(statistics.pstdev(paragraph_lengths), 3)
            if len(paragraph_lengths) > 1
            else 0.0
        )

        stopword_count = sum(1 for token in tokens if token["lower"] in self.stopwords)
        features["stopword_ratio"] = (
            round(stopword_count / word_count, 4) if word_count else 0.0
        )

        pronoun_count = sum(1 for token in tokens if token["lower"] in PRONOUNS)
        features["pronoun_ratio"] = (
            round(pronoun_count / word_count, 4) if word_count else 0.0
        )
        modal_count = sum(1 for token in tokens if token["lower"] in MODAL_VERBS)
        features["modal_count"] = modal_count
        features["modal_density"] = (
            round(modal_count / max(1, len(sentences)), 4) if sentences else 0.0
        )

        passive_voice_count = len(PASSIVE_VOICE_PATTERN.findall(text_lower))
        features["passive_voice_count"] = passive_voice_count

        word_counter = Counter(token["lower"] for token in tokens if token["lower"])
        repeated_token_total = sum(count for count in word_counter.values() if count > 1)
        features["repeated_word_ratio"] = (
            round(repeated_token_total / word_count, 4) if word_count else 0.0
        )
        top_word_share = (
            sum(count for _, count in word_counter.most_common(5)) / word_count
            if word_count
            else 0.0
        )
        features["top_word_share"] = round(top_word_share, 4)

        lowers = [token["lower"] for token in tokens]
        bigrams = list(zip(lowers, lowers[1:]))
        trigrams = list(zip(lowers, lowers[1:], lowers[2:]))
        features["repeated_bigram_count"] = sum(
            1 for count in Counter(bigrams).values() if count > 1
        )
        features["repeated_trigram_count"] = sum(
            1 for count in Counter(trigrams).values() if count > 1
        )

        features["discourse_marker_count"] = self._count_phrases(text_lower, DISCOURSE_MARKERS)
        features["hedge_marker_count"] = self._count_phrases(text_lower, HEDGE_WORDS)
        features["certainty_marker_count"] = self._count_phrases(text_lower, CERTAINTY_WORDS)

        lines = text.splitlines()
        non_empty_lines = [line for line in lines if line.strip()]
        bullet_lines = sum(1 for line in lines if BULLET_LINE_PATTERN.match(line))
        heading_lines = 0
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            letters = "".join(HEADING_LETTER_PATTERN.findall(stripped))
            if letters and letters.isupper() and len(letters) >= 4:
                heading_lines += 1
        markdown_table_lines = sum(1 for line in lines if line.count("|") >= 2)

        features["bullet_line_count"] = bullet_lines
        features["heading_line_count"] = heading_lines
        features["bullet_ratio"] = (
            round(bullet_lines / max(1, len(non_empty_lines)), 4)
            if non_empty_lines
            else 0.0
        )
        features["markdown_table_count"] = markdown_table_lines

        sentence_total = len(sentences)
        features["comma_per_sentence"] = round(
            text.count(",") / max(1, sentence_total), 4
        )
        features["exclamation_count"] = text.count("!")
        features["question_mark_count"] = text.count("?")
        features["semicolon_count"] = text.count(";")
        features["colon_count"] = text.count(":")
        features["double_punct_count"] = len(DOUBLE_PUNCT_PATTERN.findall(text))
        features["quote_count"] = text.count('"')

        digit_count = sum(1 for ch in text if ch.isdigit())
        number_tokens = sum(1 for token in tokens if NUMBER_TOKEN_PATTERN.fullmatch(token["word"]))
        features["digit_count"] = digit_count
        features["number_token_ratio"] = (
            round(number_tokens / word_count, 4) if word_count else 0.0
        )
        percent_count = text.count("%") + len(re.findall(r"\bpercent\b", text_lower))
        features["percent_reference_count"] = percent_count
        currency_symbol_count = sum(text.count(symbol) for symbol in CURRENCY_SYMBOLS)
        features["currency_symbol_count"] = currency_symbol_count

        first_token_by_sentence: Dict[int, int] = {}
        for token in tokens:
            if token["sentence_index"] not in first_token_by_sentence:
                first_token_by_sentence[token["sentence_index"]] = token["token_index"]
        capitalized_mid_sentence = sum(
            1
            for token in tokens
            if token["word"][:1].isupper()
            and token["sentence_index"] in first_token_by_sentence
            and token["token_index"] != first_token_by_sentence[token["sentence_index"]]
        )
        features["capitalized_mid_sentence_count"] = capitalized_mid_sentence

        avg_word_length = (
            sum(len(token["word"]) for token in tokens) / word_count if word_count else 0.0
        )
        features["avg_word_length"] = round(avg_word_length, 3)
        long_word_count = sum(1 for token in tokens if len(token["word"]) >= 7)
        features["long_word_ratio"] = (
            round(long_word_count / word_count, 4) if word_count else 0.0
        )

        total_syllables = sum(self._estimate_syllables(token["word"]) for token in tokens)
        syllables_per_word = total_syllables / word_count if word_count else 0.0
        features["flesch_reading_ease"] = (
            round(
                206.835
                - 1.015 * avg_sentence_len
                - 84.6 * syllables_per_word,
                3,
            )
            if word_count and sentence_total
            else 0.0
        )

        self_reference_count = sum(
            len(re.findall(pattern, text_lower)) for pattern in SELF_REFERENCE_PATTERNS
        )
        features["self_reference_count"] = self_reference_count

        question_sentence_count = sum(
            1 for sentence in sentences if sentence.rstrip().endswith("?")
        )
        features["question_sentence_ratio"] = (
            round(question_sentence_count / max(1, sentence_total), 4)
            if sentence_total
            else 0.0
        )

        imperative_indicator_count = 0
        for sentence in sentences:
            stripped = sentence.strip().lower()
            if not stripped:
                continue
            if any(stripped.startswith(starter) for starter in IMPERATIVE_STARTERS):
                imperative_indicator_count += 1
        features["imperative_indicator_count"] = imperative_indicator_count

        emotion_word_count = sum(1 for token in tokens if token["lower"] in EMOTION_WORDS)
        features["emotion_word_count"] = emotion_word_count
        features["idiom_count"] = self._count_phrases(text_lower, IDIOM_PATTERNS)

        features["multi_space_count"] = len(MULTI_SPACE_PATTERN.findall(text))
        features["line_count"] = len(lines)

        return features

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

        feature_stats = self._extract_text_features(text, tokens, sentences)

        analysis = {
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
        analysis.update(feature_stats)
        analysis["feature_version"] = 1
        return analysis

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
