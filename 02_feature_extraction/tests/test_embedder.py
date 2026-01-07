# Test Embedder
"""
Unit tests for text embedder and transformer embedder modules.
"""

import unittest
import sys
import os
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from text_embedder import TextEmbedder


class TestTextEmbedder(unittest.TestCase):
    """Test cases for TextEmbedder class."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.sample_texts = [
            "ninna face tumba ugly agi ide",
            "you are very smart and kind",
            "yen guru helthiya ninge",
            "this is a normal message",
            "bekagidya thumba irritating fellow"
        ]
        cls.embedder = TextEmbedder()
    
    def test_initialization(self):
        """Test TextEmbedder initialization."""
        embedder = TextEmbedder()
        self.assertIsNotNone(embedder)
        self.assertIsNotNone(embedder.config)
    
    def test_tfidf_fit(self):
        """Test TF-IDF fitting."""
        self.embedder.fit_tfidf(self.sample_texts)
        self.assertTrue(self.embedder.tfidf_fitted)
    
    def test_tfidf_extract(self):
        """Test TF-IDF extraction."""
        self.embedder.fit_tfidf(self.sample_texts)
        features = self.embedder.extract_tfidf(self.sample_texts[:2])
        
        self.assertIsInstance(features, np.ndarray)
        self.assertEqual(len(features), 2)
    
    def test_tfidf_single_text(self):
        """Test TF-IDF with single text."""
        self.embedder.fit_tfidf(self.sample_texts)
        features = self.embedder.extract_tfidf("hello world test")
        
        self.assertIsInstance(features, np.ndarray)
        self.assertEqual(len(features), 1)
    
    def test_word2vec_fit(self):
        """Test Word2Vec fitting."""
        try:
            self.embedder.fit_word2vec(self.sample_texts)
            self.assertTrue(self.embedder.word2vec_fitted)
        except ImportError:
            self.skipTest("gensim not installed")
    
    def test_word2vec_extract(self):
        """Test Word2Vec extraction."""
        try:
            self.embedder.fit_word2vec(self.sample_texts)
            features = self.embedder.extract_word2vec(self.sample_texts[:2])
            
            self.assertIsInstance(features, np.ndarray)
            self.assertEqual(len(features), 2)
        except ImportError:
            self.skipTest("gensim not installed")
    
    def test_fasttext_fit(self):
        """Test FastText fitting."""
        try:
            self.embedder.fit_fasttext(self.sample_texts)
            self.assertTrue(self.embedder.fasttext_fitted)
        except ImportError:
            self.skipTest("gensim not installed")
    
    def test_fasttext_extract(self):
        """Test FastText extraction."""
        try:
            self.embedder.fit_fasttext(self.sample_texts)
            features = self.embedder.extract_fasttext(self.sample_texts[:2])
            
            self.assertIsInstance(features, np.ndarray)
            self.assertEqual(len(features), 2)
        except ImportError:
            self.skipTest("gensim not installed")
    
    def test_feature_names(self):
        """Test feature names retrieval."""
        self.embedder.fit_tfidf(self.sample_texts)
        names = self.embedder.get_feature_names('tfidf')
        
        self.assertIsInstance(names, list)
        self.assertTrue(len(names) > 0)
    
    def test_empty_text_handling(self):
        """Test handling of empty text."""
        self.embedder.fit_tfidf(self.sample_texts)
        features = self.embedder.extract_tfidf("")
        
        self.assertIsInstance(features, np.ndarray)
    
    def test_kannada_text(self):
        """Test Kannada romanized text handling."""
        kannada_texts = [
            "ninage gottu ninna problem",
            "heege maadibeda guru",
            "tumba kettadru neen"
        ]
        
        self.embedder.fit_tfidf(kannada_texts)
        features = self.embedder.extract_tfidf(kannada_texts)
        
        self.assertIsInstance(features, np.ndarray)
        self.assertEqual(len(features), 3)


class TestTransformerEmbedder(unittest.TestCase):
    """Test cases for TransformerEmbedder class."""
    
    def test_import(self):
        """Test transformer embedder import."""
        try:
            from transformer_embedder import TransformerEmbedder
            self.assertTrue(True)
        except ImportError:
            self.skipTest("transformers not installed")
    
    def test_model_configs(self):
        """Test available model configurations."""
        try:
            from transformer_embedder import TransformerEmbedder
            
            expected_models = ['mbert', 'indic-bert', 'bert-base', 'xlm-roberta', 'distilbert']
            
            for model in expected_models:
                self.assertIn(model, TransformerEmbedder.MODEL_CONFIGS)
        except ImportError:
            self.skipTest("transformers not installed")


if __name__ == '__main__':
    unittest.main(verbosity=2)
