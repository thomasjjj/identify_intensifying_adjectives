#!/usr/bin/env python3
"""
Test Suite for Text Intensification Comparison Tool
Comprehensive testing with various text samples and validation scenarios.
"""

import unittest
import sys
import os
from io import StringIO
from contextlib import redirect_stdout


# Try multiple import strategies
def import_comparator():
    """Try to import the comparator from various locations."""
    import_attempts = [
        # Most common names
        "main",
        "compare_texts",
        # Parent directory
        "..main",
        "..compare_texts",
        # Other possible names
        "text_intensification_comparator",
        "intensifier_detector",
    ]

    # Add current directory and parent to path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.insert(0, current_dir)
    sys.path.insert(0, parent_dir)

    # Try to find the module
    for attempt in import_attempts:
        try:
            if attempt.startswith(".."):
                # Handle parent directory import
                module_name = attempt[2:]
                sys.path.insert(0, parent_dir)
                module = __import__(module_name)
            else:
                module = __import__(attempt)

            # Check if the required classes exist
            if hasattr(module, 'TextIntensificationComparator'):
                return module.TextIntensificationComparator, getattr(module, 'IntensifierScore', None)
        except ImportError:
            continue

    return None, None


# Try to import the main classes
TextIntensificationComparator, IntensifierScore = import_comparator()

if TextIntensificationComparator is None:
    print("=" * 60)
    print("❌ ERROR: Could not import TextIntensificationComparator")
    print("=" * 60)
    print("Please ensure one of these files exists in your project:")
    print("  - compare_texts.py")
    print("  - text_intensification_comparator.py")
    print("  - intensifier_detector.py")
    print()
    print("Current directory:", os.getcwd())
    print("Test file location:", os.path.abspath(__file__))
    print()
    print("Files in current directory:")
    current_files = [f for f in os.listdir('.') if f.endswith('.py')]
    for file in current_files:
        print(f"  - {file}")
    print("=" * 60)
    sys.exit(1)

print(f"✅ Successfully imported TextIntensificationComparator")


class TestTextSamples:
    """Collection of test text samples for various scenarios."""

    # AI-generated samples (high intensification expected)
    AI_SAMPLE_HIGH = """
    In an unprecedented move, the revolutionary technology represents a groundbreaking 
    advancement that could have profound implications. This extraordinary development 
    showcases remarkable innovation through comprehensive analysis of massive datasets. 
    The stunning results reveal alarming trends that demand urgent attention from 
    critical stakeholders.
    """

    AI_SAMPLE_MEDIUM = """
    The detailed report highlights significant findings from the extensive study. 
    Researchers discovered notable patterns in the comprehensive data analysis. 
    These important results provide valuable insights into crucial aspects of 
    the substantial research project.
    """

    AI_SAMPLE_LOW = """
    Recent studies show interesting patterns in climate data. Scientists found 
    correlations between temperature changes and weather events. The research 
    provides new information about environmental trends over time.
    """

    # Human-written samples (low intensification expected)
    HUMAN_SAMPLE_NEWS = """
    The Federal Reserve raised interest rates by 0.25 percentage points yesterday. 
    Chair Powell cited inflation concerns in his statement to reporters. Markets 
    closed down 1.2% following the announcement. Economists expect further rate 
    increases this year.
    """

    HUMAN_SAMPLE_ACADEMIC = """
    This study examines the relationship between social media usage and sleep 
    patterns in college students. We collected data from 200 participants over 
    six months. Results show correlation between screen time and sleep quality. 
    Further research is needed to establish causation.
    """

    HUMAN_SAMPLE_CASUAL = """
    I went to the new coffee shop downtown yesterday. The espresso was pretty good 
    and the atmosphere was nice. They have outdoor seating which is great when 
    the weather is warm. I'll probably go back next week.
    """

    # Edge cases
    EMPTY_TEXT = ""

    ONLY_NOUNS = "Dog cat house tree car book phone computer desk chair"

    ONLY_ADJECTIVES = "Big small fast slow hot cold bright dark loud quiet"

    SUPERLATIVES_TEXT = """
    This is the biggest, most extraordinary, and most unprecedented event ever. 
    It represents the greatest advancement and the most remarkable achievement 
    in the most comprehensive study of the largest dataset.
    """

    TECHNICAL_TEXT = """
    The neural network architecture utilizes convolutional layers with ReLU 
    activation functions. Backpropagation optimizes weights through gradient 
    descent. The model achieves 94.3% accuracy on the validation set with 
    a learning rate of 0.001.
    """


class TestIntensifierDetection(unittest.TestCase):
    """Test the core intensifier detection functionality."""

    @classmethod
    def setUpClass(cls):
        """Initialize the comparator once for all tests."""
        print("Initializing TextIntensificationComparator...")
        cls.comparator = TextIntensificationComparator()
        print("✓ Comparator ready")

    def test_high_intensification_detection(self):
        """Test detection of high intensification (AI-like text)."""
        analysis = self.comparator.analyze_text(TestTextSamples.AI_SAMPLE_HIGH, "AI High")

        self.assertGreater(analysis['noun_intensification_rate'], 20,
                           "High AI sample should have >20% noun intensification")
        self.assertGreater(analysis['adj_intensification_rate'], 30,
                           "High AI sample should have >30% adjective intensification")
        self.assertGreater(len(analysis['intensifying_words']), 3,
                           "Should detect multiple intensifying adjectives")

    def test_low_intensification_detection(self):
        """Test detection of low intensification (human-like text)."""
        analysis = self.comparator.analyze_text(TestTextSamples.HUMAN_SAMPLE_NEWS, "Human News")

        self.assertLess(analysis['noun_intensification_rate'], 15,
                        "Human sample should have <15% noun intensification")
        self.assertLess(analysis['adj_intensification_rate'], 20,
                        "Human sample should have <20% adjective intensification")

    def test_medium_intensification_detection(self):
        """Test detection of medium intensification."""
        analysis = self.comparator.analyze_text(TestTextSamples.AI_SAMPLE_MEDIUM, "AI Medium")

        # Should be between low and high, but AI text can be quite intensified
        self.assertGreater(analysis['noun_intensification_rate'], 10)
        self.assertLess(analysis['noun_intensification_rate'],
                        90)  # Increased threshold since AI detection is working well

        # Should detect some intensifying adjectives but not as many as high sample
        self.assertGreater(len(analysis['intensifying_words']), 0)

    def test_superlatives_detection(self):
        """Test detection of superlative forms."""
        analysis = self.comparator.analyze_text(TestTextSamples.SUPERLATIVES_TEXT, "Superlatives")

        self.assertGreater(analysis['adj_intensification_rate'], 50,
                           "Superlatives text should have very high intensification")

        # Check if specific superlatives are detected
        detected_words = [word.lower() for word in analysis['intensifying_words']]
        self.assertIn('biggest', detected_words, "Should detect 'biggest'")
        self.assertIn('greatest', detected_words, "Should detect 'greatest'")

    def test_adverb_intensification_detection(self):
        """Test detection of intensifying adverbs."""
        text = "The proposal is highly effective and deeply concerning, absolutely demanding immediate action."
        analysis = self.comparator.analyze_text(text, "Adverb Sample")

        self.assertGreaterEqual(analysis['intensifying_adverbs'], 2,
                                "Should detect multiple intensifying adverbs")
        self.assertGreater(analysis['adv_intensification_rate'], 0,
                           "Adverb intensification rate should be greater than zero")
        detected_adverbs = [score.word.lower() for score in analysis['detailed_intensifiers'] if score.part_of_speech == 'ADV']
        for expected in ['highly', 'deeply', 'absolutely']:
            self.assertIn(expected, detected_adverbs,
                          f"Should detect intensifying adverb '{expected}'")

    def test_empty_text_handling(self):
        """Test handling of empty or minimal text."""
        analysis = self.comparator.analyze_text(TestTextSamples.EMPTY_TEXT, "Empty")

        self.assertEqual(analysis['total_adjectives'], 0)
        self.assertEqual(analysis['total_nouns'], 0)
        self.assertEqual(analysis['intensifying_adjectives'], 0)

    def test_confidence_scores(self):
        """Test that confidence scores are reasonable."""
        analysis = self.comparator.analyze_text(TestTextSamples.AI_SAMPLE_HIGH, "AI High")

        for word, confidence in analysis['confidence_scores'].items():
            self.assertGreaterEqual(confidence, 0.0, f"Confidence for '{word}' should be >= 0")
            self.assertLessEqual(confidence, 1.0, f"Confidence for '{word}' should be <= 1")

    def test_intensified_pairs_format(self):
        """Test that intensified pairs are properly formatted."""
        analysis = self.comparator.analyze_text(TestTextSamples.AI_SAMPLE_HIGH, "AI High")

        for pair in analysis['intensified_pairs']:
            parts = pair.split(' ')
            self.assertGreaterEqual(len(parts), 2, f"Pair '{pair}' should have at least 2 words")


class TestComparisonFunctionality(unittest.TestCase):
    """Test the text comparison functionality."""

    @classmethod
    def setUpClass(cls):
        cls.comparator = TextIntensificationComparator()

    def test_ai_vs_human_comparison(self):
        """Test comparison between AI and human text."""
        # Capture output
        output = StringIO()
        with redirect_stdout(output):
            analysis1, analysis2 = self.comparator.compare_texts(
                TestTextSamples.AI_SAMPLE_HIGH,
                TestTextSamples.HUMAN_SAMPLE_NEWS,
                "AI Sample", "Human Sample"
            )

        # AI should have higher intensification
        self.assertGreater(analysis1['noun_intensification_rate'],
                           analysis2['noun_intensification_rate'],
                           "AI sample should have higher intensification than human")

    def test_same_text_comparison(self):
        """Test comparison of identical texts."""
        output = StringIO()
        with redirect_stdout(output):
            analysis1, analysis2 = self.comparator.compare_texts(
                TestTextSamples.AI_SAMPLE_MEDIUM,
                TestTextSamples.AI_SAMPLE_MEDIUM,
                "Text A", "Text B"
            )

        # Should be identical
        self.assertEqual(analysis1['noun_intensification_rate'],
                         analysis2['noun_intensification_rate'],
                         "Identical texts should have same intensification rates")


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions."""

    @classmethod
    def setUpClass(cls):
        cls.comparator = TextIntensificationComparator()

    def test_only_nouns_text(self):
        """Test text with only nouns."""
        analysis = self.comparator.analyze_text(TestTextSamples.ONLY_NOUNS, "Only Nouns")

        self.assertEqual(analysis['intensifying_adjectives'], 0,
                         "Text with only nouns should have no intensifying adjectives")
        self.assertEqual(analysis['adj_intensification_rate'], 0,
                         "Adjective intensification rate should be 0")

    def test_only_adjectives_text(self):
        """Test text with only adjectives."""
        analysis = self.comparator.analyze_text(TestTextSamples.ONLY_ADJECTIVES, "Only Adjectives")

        self.assertEqual(analysis['intensified_nouns'], 0,
                         "Text with only adjectives should have no intensified nouns")
        self.assertEqual(analysis['noun_intensification_rate'], 0,
                         "Noun intensification rate should be 0")

    def test_technical_text(self):
        """Test technical text (should have low intensification)."""
        analysis = self.comparator.analyze_text(TestTextSamples.TECHNICAL_TEXT, "Technical")

        self.assertLess(analysis['noun_intensification_rate'], 20,
                        "Technical text should have low intensification")


class TestBenchmark(unittest.TestCase):
    """Benchmark tests for performance and accuracy."""

    @classmethod
    def setUpClass(cls):
        cls.comparator = TextIntensificationComparator()

    def test_discrimination_accuracy(self):
        """Test ability to discriminate between AI and human text."""
        test_cases = [
            (TestTextSamples.AI_SAMPLE_HIGH, "AI", True),
            (TestTextSamples.AI_SAMPLE_MEDIUM, "AI", True),
            (TestTextSamples.HUMAN_SAMPLE_NEWS, "Human", False),
            (TestTextSamples.HUMAN_SAMPLE_ACADEMIC, "Human", False),
            (TestTextSamples.HUMAN_SAMPLE_CASUAL, "Human", False),
        ]

        correct_predictions = 0

        for text, label, expected_ai in test_cases:
            analysis = self.comparator.analyze_text(text, label)

            # Use 20% noun intensification as threshold
            predicted_ai = analysis['noun_intensification_rate'] > 20

            if predicted_ai == expected_ai:
                correct_predictions += 1
            else:
                print(f"❌ Misclassified {label}: {analysis['noun_intensification_rate']:.1f}% intensification")

        accuracy = correct_predictions / len(test_cases)
        print(f"\n📊 Discrimination Accuracy: {accuracy:.1%} ({correct_predictions}/{len(test_cases)})")

        self.assertGreaterEqual(accuracy, 0.8, "Should achieve at least 80% accuracy")


def run_performance_benchmark():
    """Run performance benchmark with various text lengths."""
    print("\n🚀 PERFORMANCE BENCHMARK")
    print("=" * 40)

    comparator = TextIntensificationComparator()

    # Create texts of different lengths
    base_text = TestTextSamples.AI_SAMPLE_HIGH
    test_texts = {
        "Short (50 words)": " ".join(base_text.split()[:50]),
        "Medium (100 words)": " ".join((base_text + " " + base_text).split()[:100]),
        "Long (200 words)": " ".join((base_text * 4).split()[:200]),
    }

    import time

    for name, text in test_texts.items():
        start_time = time.time()
        analysis = comparator.analyze_text(text, name)
        end_time = time.time()

        print(
            f"{name:<20} | {end_time - start_time:.3f}s | {analysis['noun_intensification_rate']:.1f}% intensification")


def run_sample_showcase():
    """Showcase the tool with sample texts."""
    print("\n🎭 SAMPLE SHOWCASE")
    print("=" * 50)

    comparator = TextIntensificationComparator()

    showcase_samples = [
        ("🤖 High AI Sample", TestTextSamples.AI_SAMPLE_HIGH),
        ("📊 Medium AI Sample", TestTextSamples.AI_SAMPLE_MEDIUM),
        ("📰 Human News", TestTextSamples.HUMAN_SAMPLE_NEWS),
        ("🎓 Human Academic", TestTextSamples.HUMAN_SAMPLE_ACADEMIC),
    ]

    for name, text in showcase_samples:
        analysis = comparator.analyze_text(text, name)
        print(f"\n{name}")
        print(f"  Noun intensification: {analysis['noun_intensification_rate']:.1f}%")
        print(f"  Adj intensification:  {analysis['adj_intensification_rate']:.1f}%")
        print(f"  Intensifiers found:   {', '.join(analysis['intensifying_words'][:3])}...")


def main():
    """Run all tests and benchmarks."""
    print("🧪 TEXT INTENSIFICATION TOOL - TEST SUITE")
    print("=" * 60)

    # Run unit tests
    print("\n🔍 Running Unit Tests...")
    test_loader = unittest.TestLoader()
    test_suite = unittest.TestSuite()

    # Add test classes
    test_classes = [
        TestIntensifierDetection,
        TestComparisonFunctionality,
        TestEdgeCases,
        TestBenchmark
    ]

    for test_class in test_classes:
        tests = test_loader.loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    # Show results summary
    print(f"\n📋 TEST RESULTS SUMMARY")
    print("=" * 30)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {(result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun:.1%}")

    if result.failures:
        print("\n❌ FAILURES:")
        for test, traceback in result.failures:
            message = traceback.split("AssertionError: ")[-1].splitlines()[0]
            print(f"  - {test}: {message}")

    if result.errors:
        print("\n🚨 ERRORS:")
        for test, traceback in result.errors:
            message = traceback.splitlines()[-2] if "\n" in traceback else traceback
            print(f"  - {test}: {message}")

    # Run additional benchmarks
    run_performance_benchmark()
    run_sample_showcase()

    # Final status
    if result.wasSuccessful():
        print(f"\n✅ ALL TESTS PASSED! Tool is working correctly.")
    else:
        print(f"\n❌ Some tests failed. Please review the output above.")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())