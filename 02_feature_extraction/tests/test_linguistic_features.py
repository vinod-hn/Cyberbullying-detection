# Test Linguistic Features
"""
Unit tests for linguistic features extraction module.
"""

import unittest
import sys
import os
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from linguistic_features import LinguisticFeatures


class TestLinguisticFeatures(unittest.TestCase):
    """Test cases for LinguisticFeatures class."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.extractor = LinguisticFeatures()
        
        cls.sample_texts = [
            "ninna face tumba ugly agi ide",
            "You are SO STUPID!!!",
            "This is a normal friendly message.",
            "bekagidya guru what is your problem",
            "I hate you!! You're terrible 😡😡"
        ]
    
    def test_initialization(self):
        """Test LinguisticFeatures initialization."""
        extractor = LinguisticFeatures()
        self.assertIsNotNone(extractor)
        self.assertIsNotNone(extractor.config)
    
    # =========================================================================
    # Lexical Features Tests
    # =========================================================================
    def test_lexical_word_count(self):
        """Test word count feature."""
        text = "hello world this is a test"
        features = self.extractor.extract_lexical_features(text)
        
        self.assertIn('word_count', features)
        self.assertEqual(features['word_count'], 6)
    
    def test_lexical_char_count(self):
        """Test character count feature."""
        text = "hello"
        features = self.extractor.extract_lexical_features(text)
        
        self.assertIn('char_count', features)
        self.assertEqual(features['char_count'], 5)
    
    def test_lexical_caps_ratio(self):
        """Test caps ratio for aggressive text."""
        text = "YOU ARE STUPID idiot"
        features = self.extractor.extract_lexical_features(text)
        
        self.assertIn('caps_ratio', features)
        self.assertGreater(features['caps_ratio'], 0.5)
    
    def test_lexical_punct_ratio(self):
        """Test punctuation ratio."""
        text = "What!!! Why??? No!!!"
        features = self.extractor.extract_lexical_features(text)
        
        self.assertIn('punct_ratio', features)
        self.assertGreater(features['punct_ratio'], 0.3)
    
    def test_lexical_empty_text(self):
        """Test lexical features with empty text."""
        features = self.extractor.extract_lexical_features("")
        
        self.assertEqual(features['word_count'], 0)
        self.assertEqual(features['char_count'], 0)
    
    # =========================================================================
    # Semantic Features Tests
    # =========================================================================
    def test_semantic_sentiment(self):
        """Test sentiment extraction."""
        positive_text = "I love you, you are wonderful!"
        negative_text = "I hate you, you are terrible!"
        
        pos_features = self.extractor.extract_semantic_features(positive_text)
        neg_features = self.extractor.extract_semantic_features(negative_text)
        
        self.assertIn('sentiment_polarity', pos_features)
        self.assertIn('sentiment_polarity', neg_features)
        
        # Positive text should have higher polarity
        self.assertGreater(
            pos_features['sentiment_polarity'],
            neg_features['sentiment_polarity']
        )
    
    def test_semantic_aggression(self):
        """Test aggression indicator detection."""
        aggressive_text = "I will kill you stupid idiot"
        neutral_text = "The weather is nice today"
        
        agg_features = self.extractor.extract_semantic_features(aggressive_text)
        neu_features = self.extractor.extract_semantic_features(neutral_text)
        
        self.assertIn('aggression_indicators', agg_features)
        self.assertGreater(
            agg_features['aggression_indicators'],
            neu_features['aggression_indicators']
        )
    
    # =========================================================================
    # Code-Mix Features Tests
    # =========================================================================
    def test_code_mix_index(self):
        """Test code-mix index calculation."""
        code_mixed = "ninna face tumba ugly agi ide"
        english_only = "this is a pure english sentence"
        
        mix_features = self.extractor.extract_code_mix_features(code_mixed)
        eng_features = self.extractor.extract_code_mix_features(english_only)
        
        self.assertIn('code_mix_index', mix_features)
        self.assertIn('code_mix_index', eng_features)
        
        # Code-mixed text should have higher index
        self.assertGreater(
            mix_features['code_mix_index'],
            eng_features['code_mix_index']
        )
    
    def test_kannada_ratio(self):
        """Test Kannada word ratio."""
        text = "ninna tumba guru bekagidya"
        features = self.extractor.extract_code_mix_features(text)
        
        self.assertIn('kannada_ratio', features)
        self.assertGreater(features['kannada_ratio'], 0)
    
    def test_switch_points(self):
        """Test language switch point detection."""
        text = "ninna face is ugly guru"
        features = self.extractor.extract_code_mix_features(text)
        
        self.assertIn('switch_points', features)
        self.assertGreater(features['switch_points'], 0)
    
    # =========================================================================
    # Readability Features Tests
    # =========================================================================
    def test_readability_features(self):
        """Test readability feature extraction."""
        text = "This is a simple sentence. It is easy to read."
        features = self.extractor.extract_readability_features(text)
        
        self.assertIn('syllable_count', features)
        self.assertIn('avg_syllables_per_word', features)
    
    def test_complex_text_readability(self):
        """Test readability for complex text."""
        simple_text = "I run. He runs. They run."
        complex_text = "The implementation of sophisticated algorithms necessitates comprehensive understanding."
        
        simple_features = self.extractor.extract_readability_features(simple_text)
        complex_features = self.extractor.extract_readability_features(complex_text)
        
        # Complex text should have higher avg syllables
        self.assertGreater(
            complex_features.get('avg_syllables_per_word', 0),
            simple_features.get('avg_syllables_per_word', 0)
        )
    
    # =========================================================================
    # Combined Features Tests
    # =========================================================================
    def test_all_features(self):
        """Test extracting all linguistic features."""
        text = self.sample_texts[0]
        features = self.extractor.extract_all_features(text)
        
        # Should have lexical features
        self.assertIn('word_count', features)
        
        # Should have semantic features
        self.assertIn('sentiment_polarity', features)
        
        # Should have code-mix features
        self.assertIn('code_mix_index', features)
    
    def test_numeric_features(self):
        """Test numeric feature extraction."""
        features = self.extractor.extract_numeric_features(self.sample_texts)
        
        self.assertIsInstance(features, np.ndarray)
        self.assertEqual(len(features), len(self.sample_texts))
    
    def test_feature_names(self):
        """Test feature names retrieval."""
        names = self.extractor.get_feature_names()
        
        self.assertIsInstance(names, list)
        self.assertTrue(len(names) > 0)
        self.assertIn('word_count', names)
    
    def test_batch_extraction(self):
        """Test batch extraction of features."""
        texts = self.sample_texts
        
        # Should handle list of texts
        all_features = self.extractor.extract_all_features(texts)
        
        if isinstance(all_features, list):
            self.assertEqual(len(all_features), len(texts))
        else:
            # Single dict with aggregated features
            self.assertIsInstance(all_features, dict)


class TestCodeMixedCyberbullying(unittest.TestCase):
    """Test linguistic features on Kannada-English cyberbullying samples."""
    
    @classmethod
    def setUpClass(cls):
        """Set up with cyberbullying samples."""
        cls.extractor = LinguisticFeatures()
        
        cls.bullying_texts = [
            "ninna thotha face yako guru",
            "bekagidya you are such a loser",
            "tumba ugly agi idiya singri",
        ]
        
        cls.neutral_texts = [
            "hello how are you today",
            "chennagide guru ella okay",
            "good morning everyone",
        ]
    
    def test_bullying_vs_neutral_aggression(self):
        """Test aggression detection difference."""
        bullying_agg = []
        for text in self.bullying_texts:
            features = self.extractor.extract_semantic_features(text)
            bullying_agg.append(features.get('aggression_indicators', 0))
        
        neutral_agg = []
        for text in self.neutral_texts:
            features = self.extractor.extract_semantic_features(text)
            neutral_agg.append(features.get('aggression_indicators', 0))
        
        # Bullying texts should have higher average aggression
        self.assertGreaterEqual(
            sum(bullying_agg) / len(bullying_agg),
            sum(neutral_agg) / len(neutral_agg)
        )
    
    def test_code_mix_in_bullying(self):
        """Test code-mix detection in bullying texts."""
        for text in self.bullying_texts:
            features = self.extractor.extract_code_mix_features(text)
            # Should detect Kannada elements
            self.assertGreater(features.get('kannada_ratio', 0), 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
