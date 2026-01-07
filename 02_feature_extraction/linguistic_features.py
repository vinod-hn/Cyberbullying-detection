# Linguistic Features
"""
LinguisticFeatures: Linguistic analysis for cyberbullying detection.
Extracts syntax, semantics, sentiment, and readability features.
Optimized for Kannada-English code-mixed text.
"""

import re
import math
import logging
from typing import List, Dict, Optional, Any, Union, Tuple
from collections import Counter
import numpy as np

logger = logging.getLogger(__name__)

# Optional imports
try:
    import nltk
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.tag import pos_tag
    from nltk.corpus import stopwords
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    logger.warning("NLTK not available. Some features disabled.")

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False
    logger.warning("TextBlob not available. Sentiment analysis disabled.")


class LinguisticFeatures:
    """
    Linguistic feature extractor for cyberbullying detection.
    
    Extracts:
    - Lexical features (word count, char count, vocabulary richness)
    - Syntactic features (POS tags, sentence structure)
    - Semantic features (sentiment, subjectivity)
    - Readability features (complexity measures)
    - Code-mix specific features (language ratio, script mixing)
    
    Attributes:
        config: Configuration dictionary
        stopwords: Set of stopwords
    """
    
    # Kannada Unicode range for script detection
    KANNADA_RANGE = (0x0C80, 0x0CFF)
    
    # Common Romanized Kannada patterns
    KANNADA_PATTERNS = {
        'intensifiers': ['tumba', 'thumba', 'sakkat', 'sakkath', 'full', 'jaasti'],
        'pronouns': ['nee', 'neenu', 'naanu', 'avnu', 'avlu', 'ninna', 'nanna'],
        'address': ['maga', 'machaa', 'macha', 'guru', 'yaar', 're', 'anna', 'akka'],
        'negation': ['illa', 'beda', 'gotilla', 'gottilla'],
        'questions': ['yaake', 'yelli', 'hege', 'yenu', 'enu', 'yavaga'],
        'insults': ['thotha', 'thota', 'singri', 'moorkha', 'bewakoof', 'hucchadana'],
    }
    
    # Profanity/aggressive words for detection
    AGGRESSIVE_WORDS = {
        'english': [
            'stupid', 'idiot', 'dumb', 'fool', 'loser', 'hate', 'ugly', 'fat',
            'kill', 'die', 'useless', 'worthless', 'pathetic', 'disgusting',
            'shut up', 'get lost', 'go away', 'leave', 'stop', 'annoying',
            'irritating', 'toxic', 'fake', 'trash', 'garbage', 'waste'
        ],
        'kannada': [
            'thotha', 'thota', 'singri', 'dagarina', 'hucchadana', 'moorkha',
            'bewakoof', 'saayi', 'bekku', 'haavu', 'irritating', 'boring'
        ]
    }
    
    # Punctuation patterns for emphasis
    EMPHASIS_PATTERNS = {
        'exclamation': r'!+',
        'question': r'\?+',
        'ellipsis': r'\.{3,}',
        'caps': r'\b[A-Z]{2,}\b',
        'repeated_chars': r'(.)\1{2,}',
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize LinguisticFeatures.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or self._default_config()
        
        # Initialize NLTK resources
        self.stopwords_en = set()
        if NLTK_AVAILABLE:
            self._init_nltk()
    
    def _default_config(self) -> Dict[str, Any]:
        """Return default configuration."""
        return {
            'include_lexical': True,
            'include_syntactic': True,
            'include_semantic': True,
            'include_readability': True,
            'include_code_mix': True,
            'include_aggressive': True,
            'lowercase_for_analysis': True
        }
    
    def _init_nltk(self) -> None:
        """Initialize NLTK resources."""
        try:
            self.stopwords_en = set(stopwords.words('english'))
        except LookupError:
            try:
                nltk.download('stopwords', quiet=True)
                nltk.download('punkt', quiet=True)
                nltk.download('averaged_perceptron_tagger', quiet=True)
                self.stopwords_en = set(stopwords.words('english'))
            except Exception:
                self.stopwords_en = set()
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        if NLTK_AVAILABLE:
            try:
                return word_tokenize(text.lower())
            except Exception:
                pass
        # Fallback tokenization
        return re.findall(r'\b\w+\b', text.lower())
    
    def _sentence_tokenize(self, text: str) -> List[str]:
        """Tokenize text into sentences."""
        if NLTK_AVAILABLE:
            try:
                return sent_tokenize(text)
            except Exception:
                pass
        # Fallback sentence splitting
        return re.split(r'[.!?]+', text)
    
    # =========================================================================
    # Lexical Features
    # =========================================================================
    def extract_lexical_features(self, text: str) -> Dict[str, Any]:
        """
        Extract lexical features from text.
        
        Features:
        - Character count
        - Word count
        - Average word length
        - Vocabulary richness (unique/total)
        - Stopword ratio
        """
        tokens = self._tokenize(text)
        words = [t for t in tokens if t.isalpha()]
        
        char_count = len(text)
        word_count = len(words)
        unique_words = len(set(words))
        
        avg_word_length = (
            sum(len(w) for w in words) / word_count
            if word_count > 0 else 0
        )
        
        vocabulary_richness = (
            unique_words / word_count
            if word_count > 0 else 0
        )
        
        stopword_count = sum(1 for w in words if w in self.stopwords_en)
        stopword_ratio = stopword_count / word_count if word_count > 0 else 0
        
        return {
            'char_count': char_count,
            'word_count': word_count,
            'unique_word_count': unique_words,
            'avg_word_length': round(avg_word_length, 2),
            'vocabulary_richness': round(vocabulary_richness, 3),
            'stopword_ratio': round(stopword_ratio, 3),
            'uppercase_ratio': self._uppercase_ratio(text)
        }
    
    def _uppercase_ratio(self, text: str) -> float:
        """Calculate ratio of uppercase characters."""
        letters = [c for c in text if c.isalpha()]
        if not letters:
            return 0.0
        uppercase = sum(1 for c in letters if c.isupper())
        return round(uppercase / len(letters), 3)
    
    # =========================================================================
    # Syntactic Features
    # =========================================================================
    def extract_syntactic_features(self, text: str) -> Dict[str, Any]:
        """
        Extract syntactic features from text.
        
        Features:
        - POS tag distribution
        - Sentence count
        - Average sentence length
        - Punctuation patterns
        """
        sentences = self._sentence_tokenize(text)
        tokens = self._tokenize(text)
        
        # POS tagging
        pos_counts = {}
        if NLTK_AVAILABLE:
            try:
                tagged = pos_tag(tokens)
                pos_counts = Counter(tag for _, tag in tagged)
            except Exception:
                pass
        
        # Sentence features
        sentence_count = len([s for s in sentences if s.strip()])
        avg_sentence_length = (
            len(tokens) / sentence_count
            if sentence_count > 0 else 0
        )
        
        # Punctuation features
        punctuation = self._extract_punctuation_features(text)
        
        return {
            'sentence_count': sentence_count,
            'avg_sentence_length': round(avg_sentence_length, 2),
            'noun_count': pos_counts.get('NN', 0) + pos_counts.get('NNS', 0),
            'verb_count': pos_counts.get('VB', 0) + pos_counts.get('VBP', 0),
            'adjective_count': pos_counts.get('JJ', 0),
            'pronoun_count': pos_counts.get('PRP', 0) + pos_counts.get('PRP$', 0),
            **punctuation
        }
    
    def _extract_punctuation_features(self, text: str) -> Dict[str, int]:
        """Extract punctuation-based features."""
        return {
            'exclamation_count': len(re.findall(r'!', text)),
            'question_count': len(re.findall(r'\?', text)),
            'caps_word_count': len(re.findall(r'\b[A-Z]{2,}\b', text)),
            'repeated_char_count': len(re.findall(r'(.)\1{2,}', text)),
            'ellipsis_count': len(re.findall(r'\.{3,}', text))
        }
    
    # =========================================================================
    # Semantic Features
    # =========================================================================
    def extract_semantic_features(self, text: str) -> Dict[str, Any]:
        """
        Extract semantic features from text.
        
        Features:
        - Sentiment polarity
        - Subjectivity
        - Aggressive word count
        """
        features = {
            'sentiment_polarity': 0.0,
            'sentiment_subjectivity': 0.5,
            'is_positive': False,
            'is_negative': False,
            'is_neutral': True
        }
        
        # TextBlob sentiment
        if TEXTBLOB_AVAILABLE:
            try:
                blob = TextBlob(text)
                polarity = blob.sentiment.polarity
                subjectivity = blob.sentiment.subjectivity
                
                features['sentiment_polarity'] = round(polarity, 3)
                features['sentiment_subjectivity'] = round(subjectivity, 3)
                features['is_positive'] = polarity > 0.1
                features['is_negative'] = polarity < -0.1
                features['is_neutral'] = -0.1 <= polarity <= 0.1
            except Exception:
                pass
        
        # Aggressive word detection
        text_lower = text.lower()
        aggressive_count = 0
        aggressive_words_found = []
        
        for word in self.AGGRESSIVE_WORDS['english']:
            if word in text_lower:
                aggressive_count += 1
                aggressive_words_found.append(word)
        
        for word in self.AGGRESSIVE_WORDS['kannada']:
            if word in text_lower:
                aggressive_count += 1
                aggressive_words_found.append(word)
        
        features['aggressive_word_count'] = aggressive_count
        features['aggressive_words'] = aggressive_words_found
        features['has_aggression'] = aggressive_count > 0
        
        return features
    
    # =========================================================================
    # Readability Features
    # =========================================================================
    def extract_readability_features(self, text: str) -> Dict[str, Any]:
        """
        Extract readability features from text.
        
        Features:
        - Syllable count estimation
        - Flesch reading ease (adapted)
        - Text complexity score
        """
        tokens = self._tokenize(text)
        words = [t for t in tokens if t.isalpha()]
        sentences = self._sentence_tokenize(text)
        
        word_count = len(words)
        sentence_count = max(len([s for s in sentences if s.strip()]), 1)
        
        # Estimate syllables (simple heuristic)
        syllable_count = sum(self._count_syllables(w) for w in words)
        
        # Adapted Flesch reading ease
        if word_count > 0 and sentence_count > 0:
            avg_sentence_length = word_count / sentence_count
            avg_syllables_per_word = syllable_count / word_count
            flesch_score = 206.835 - 1.015 * avg_sentence_length - 84.6 * avg_syllables_per_word
            flesch_score = max(0, min(100, flesch_score))
        else:
            flesch_score = 50
        
        # Complexity score (higher = more complex)
        complexity = self._calculate_complexity(text, tokens)
        
        return {
            'syllable_count': syllable_count,
            'avg_syllables_per_word': round(syllable_count / word_count if word_count > 0 else 0, 2),
            'flesch_reading_ease': round(flesch_score, 2),
            'text_complexity': round(complexity, 3)
        }
    
    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word (heuristic)."""
        word = word.lower()
        count = 0
        vowels = 'aeiouy'
        prev_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_vowel:
                count += 1
            prev_vowel = is_vowel
        
        # Adjust for silent e
        if word.endswith('e'):
            count -= 1
        
        return max(count, 1)
    
    def _calculate_complexity(self, text: str, tokens: List[str]) -> float:
        """Calculate text complexity score."""
        if not tokens:
            return 0.0
        
        # Factors contributing to complexity
        long_words = sum(1 for t in tokens if len(t) > 6)
        unique_ratio = len(set(tokens)) / len(tokens) if tokens else 0
        
        # Check for code-mixing
        has_kannada = any(
            0x0C80 <= ord(c) <= 0x0CFF
            for c in text
        )
        
        complexity = (
            0.3 * (long_words / len(tokens)) +
            0.3 * unique_ratio +
            0.2 * (1 if has_kannada else 0) +
            0.2 * min(len(text) / 200, 1)
        )
        
        return complexity
    
    # =========================================================================
    # Code-Mix Features
    # =========================================================================
    def extract_code_mix_features(self, text: str) -> Dict[str, Any]:
        """
        Extract code-mixing features.
        
        Features:
        - Kannada script ratio
        - Romanized Kannada word count
        - English word count
        - Code-mix index
        """
        tokens = self._tokenize(text)
        
        # Script detection
        kannada_script_chars = 0
        roman_chars = 0
        total_chars = 0
        
        for char in text:
            if char.isalpha():
                total_chars += 1
                code = ord(char)
                if 0x0C80 <= code <= 0x0CFF:
                    kannada_script_chars += 1
                elif char.isascii():
                    roman_chars += 1
        
        kannada_script_ratio = kannada_script_chars / total_chars if total_chars > 0 else 0
        
        # Romanized Kannada detection
        romanized_kannada_count = 0
        english_count = 0
        
        all_kannada_words = set()
        for category in self.KANNADA_PATTERNS.values():
            all_kannada_words.update(category)
        
        for token in tokens:
            if token in all_kannada_words:
                romanized_kannada_count += 1
            elif token.isalpha() and token not in self.stopwords_en:
                english_count += 1
        
        # Code-mix index
        if romanized_kannada_count + english_count > 0:
            code_mix_index = min(romanized_kannada_count, english_count) / (romanized_kannada_count + english_count)
        else:
            code_mix_index = 0
        
        return {
            'kannada_script_ratio': round(kannada_script_ratio, 3),
            'romanized_kannada_count': romanized_kannada_count,
            'english_word_count': english_count,
            'code_mix_index': round(code_mix_index, 3),
            'is_code_mixed': (romanized_kannada_count > 0 and english_count > 0) or kannada_script_ratio > 0,
            'dominant_language': self._detect_dominant_language(
                kannada_script_ratio, romanized_kannada_count, english_count
            )
        }
    
    def _detect_dominant_language(
        self,
        kannada_ratio: float,
        kannada_word_count: int,
        english_count: int
    ) -> str:
        """Detect dominant language in text."""
        if kannada_ratio > 0.5:
            return 'kannada_script'
        elif kannada_word_count > english_count:
            return 'kannada_romanized'
        elif english_count > kannada_word_count:
            return 'english'
        else:
            return 'mixed'
    
    # =========================================================================
    # Combined Extraction
    # =========================================================================
    def extract(self, texts: Union[str, List[str]]) -> List[Dict[str, Any]]:
        """
        Extract all linguistic features from texts.
        
        Args:
            texts: Text or list of texts
            
        Returns:
            List of feature dictionaries
        """
        if isinstance(texts, str):
            texts = [texts]
        
        results = []
        
        for text in texts:
            features = {}
            
            if self.config.get('include_lexical', True):
                features.update(self.extract_lexical_features(text))
            
            if self.config.get('include_syntactic', True):
                features.update(self.extract_syntactic_features(text))
            
            if self.config.get('include_semantic', True):
                features.update(self.extract_semantic_features(text))
            
            if self.config.get('include_readability', True):
                features.update(self.extract_readability_features(text))
            
            if self.config.get('include_code_mix', True):
                features.update(self.extract_code_mix_features(text))
            
            results.append(features)
        
        return results
    
    def extract_numeric_features(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Extract numeric features as numpy array.
        
        Args:
            texts: Text or list of texts
            
        Returns:
            Feature matrix (n_samples, n_features)
        """
        all_features = self.extract(texts)
        
        # Select numeric features only
        numeric_keys = [
            'char_count', 'word_count', 'unique_word_count', 'avg_word_length',
            'vocabulary_richness', 'stopword_ratio', 'uppercase_ratio',
            'sentence_count', 'avg_sentence_length', 'exclamation_count',
            'question_count', 'caps_word_count', 'repeated_char_count',
            'sentiment_polarity', 'sentiment_subjectivity', 'aggressive_word_count',
            'syllable_count', 'flesch_reading_ease', 'text_complexity',
            'kannada_script_ratio', 'romanized_kannada_count', 'code_mix_index'
        ]
        
        matrix = []
        for features in all_features:
            row = [features.get(key, 0) for key in numeric_keys]
            matrix.append(row)
        
        return np.array(matrix)
    
    def get_feature_names(self) -> List[str]:
        """Get list of numeric feature names."""
        return [
            'char_count', 'word_count', 'unique_word_count', 'avg_word_length',
            'vocabulary_richness', 'stopword_ratio', 'uppercase_ratio',
            'sentence_count', 'avg_sentence_length', 'exclamation_count',
            'question_count', 'caps_word_count', 'repeated_char_count',
            'sentiment_polarity', 'sentiment_subjectivity', 'aggressive_word_count',
            'syllable_count', 'flesch_reading_ease', 'text_complexity',
            'kannada_script_ratio', 'romanized_kannada_count', 'code_mix_index'
        ]
    
    def __repr__(self) -> str:
        """String representation."""
        return f"LinguisticFeatures(nltk={NLTK_AVAILABLE}, textblob={TEXTBLOB_AVAILABLE})"


if __name__ == "__main__":
    print("LinguisticFeatures Test")
    print("=" * 50)
    
    extractor = LinguisticFeatures()
    
    test_texts = [
        "nee tumba irritating agthiya yaar!!!",
        "This is very annoying behavior!",
        "exam tumba tough aaytu, i hate it",
        "Why are you always watching me?",
    ]
    
    for text in test_texts:
        print(f"\nText: {text}")
        features = extractor.extract(text)[0]
        print(f"  Word count: {features.get('word_count')}")
        print(f"  Sentiment: {features.get('sentiment_polarity')}")
        print(f"  Code-mix index: {features.get('code_mix_index')}")
        print(f"  Aggressive words: {features.get('aggressive_words')}")
