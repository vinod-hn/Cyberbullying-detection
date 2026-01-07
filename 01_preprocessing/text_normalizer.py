# Text Normalizer
"""
TextNormalizer: Comprehensive text normalization for cyberbullying detection.
Optimized for Kannada-English code-mixed text.
Handles romanized Kannada, English, and mixed scripts.
"""

import re
import unicodedata
from typing import List, Dict, Optional, Any, Union
import os
import json


class TextNormalizer:
    """
    Text normalization class for preprocessing cyberbullying detection data.
    
    Features:
    - Whitespace normalization
    - Case normalization
    - Punctuation handling
    - URL/email removal
    - Hashtag ID removal (dataset-specific)
    - Character elongation reduction
    - Unicode normalization
    - Code-mixed text support (Kannada-English)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize TextNormalizer.
        
        Args:
            config: Configuration dictionary with normalization options.
        """
        self.config = config or self._default_config()
        self._compile_patterns()
    
    def _default_config(self) -> Dict[str, Any]:
        """Return default configuration."""
        return {
            'lowercase': True,
            'remove_urls': True,
            'remove_emails': True,
            'remove_hashtag_ids': True,  # For dataset format like #5a76
            'reduce_elongation': True,
            'max_char_repeat': 2,
            'normalize_unicode': True,
            'normalize_whitespace': True,
            'remove_zero_width': True,
            'preserve_kannada_script': True,
            'handle_contractions': True,
            'reduce_punctuation': True,
            'max_punct_repeat': 1,
            'remove_usernames': False,  # Keep anonymized usernames
            'strip_extra_spaces': True
        }
    
    def _compile_patterns(self) -> None:
        """Compile regex patterns for efficiency."""
        # URL pattern
        self.url_pattern = re.compile(
            r'https?://\S+|www\.\S+|ftp://\S+',
            re.IGNORECASE
        )
        
        # Email pattern
        self.email_pattern = re.compile(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        )
        
        # Hashtag ID pattern (dataset format: #abc123)
        self.hashtag_id_pattern = re.compile(r'#[a-f0-9]{4}')
        
        # Multiple whitespace
        self.whitespace_pattern = re.compile(r'\s+')
        
        # Character elongation (3+ repeated chars)
        self.elongation_pattern = re.compile(r'(.)\1{2,}')
        
        # Multiple punctuation
        self.punct_repeat_pattern = re.compile(r'([!?.,])\1+')
        
        # Zero-width characters
        self.zero_width_pattern = re.compile(
            r'[\u200b\u200c\u200d\u200e\u200f\u202a-\u202e\ufeff]'
        )
        
        # Kannada script range (for preservation)
        self.kannada_pattern = re.compile(r'[\u0C80-\u0CFF]+')
        
        # Phone number pattern
        self.phone_pattern = re.compile(r'\b\d{10,12}\b')
    
    def normalize(self, text: str) -> str:
        """
        Apply full normalization pipeline to text.
        
        Args:
            text: Input text to normalize.
            
        Returns:
            Normalized text string.
        """
        if text is None:
            return ""
        
        if not isinstance(text, str):
            text = str(text)
        
        if not text.strip():
            return ""
        
        result = text
        
        # Step 1: Unicode normalization (NFKC)
        if self.config.get('normalize_unicode', True):
            result = self.normalize_unicode(result)
        
        # Step 2: Remove zero-width characters
        if self.config.get('remove_zero_width', True):
            result = self.remove_zero_width(result)
        
        # Step 3: Remove URLs
        if self.config.get('remove_urls', True):
            result = self.remove_urls(result)
        
        # Step 4: Remove emails
        if self.config.get('remove_emails', True):
            result = self.remove_emails(result)
        
        # Step 5: Remove dataset hashtag IDs
        if self.config.get('remove_hashtag_ids', True):
            result = self.remove_hashtag_ids(result)
        
        # Step 6: Handle contractions
        if self.config.get('handle_contractions', True):
            result = self.expand_contractions(result)
        
        # Step 7: Reduce character elongation
        if self.config.get('reduce_elongation', True):
            result = self.reduce_elongation(result)
        
        # Step 8: Reduce punctuation repetition
        if self.config.get('reduce_punctuation', True):
            result = self.reduce_punctuation(result)
        
        # Step 9: Normalize whitespace
        if self.config.get('normalize_whitespace', True):
            result = self.normalize_whitespace(result)
        
        # Step 10: Lowercase (but preserve Kannada script handling)
        if self.config.get('lowercase', True):
            result = self.to_lowercase(result)
        
        # Step 11: Final cleanup
        if self.config.get('strip_extra_spaces', True):
            result = result.strip()
        
        return result
    
    def normalize_unicode(self, text: str) -> str:
        """
        Apply Unicode normalization (NFKC form).
        
        Args:
            text: Input text.
            
        Returns:
            Unicode normalized text.
        """
        return unicodedata.normalize('NFKC', text)
    
    def remove_zero_width(self, text: str) -> str:
        """
        Remove zero-width Unicode characters.
        
        Args:
            text: Input text.
            
        Returns:
            Text without zero-width characters.
        """
        return self.zero_width_pattern.sub('', text)
    
    def remove_urls(self, text: str) -> str:
        """
        Remove URLs from text.
        
        Args:
            text: Input text.
            
        Returns:
            Text with URLs removed.
        """
        return self.url_pattern.sub(' ', text)
    
    def remove_emails(self, text: str) -> str:
        """
        Remove email addresses from text.
        
        Args:
            text: Input text.
            
        Returns:
            Text with emails removed.
        """
        return self.email_pattern.sub(' ', text)
    
    def remove_hashtag_ids(self, text: str) -> str:
        """
        Remove dataset-specific hashtag IDs (format: #xxxx).
        
        Args:
            text: Input text.
            
        Returns:
            Text with hashtag IDs removed.
        """
        return self.hashtag_id_pattern.sub('', text)
    
    def reduce_elongation(self, text: str) -> str:
        """
        Reduce character elongation (e.g., 'noooooo' -> 'noo').
        
        Args:
            text: Input text.
            
        Returns:
            Text with reduced elongation.
        """
        max_repeat = self.config.get('max_char_repeat', 2)
        
        def replace_func(match):
            char = match.group(1)
            return char * max_repeat
        
        return self.elongation_pattern.sub(replace_func, text)
    
    def reduce_punctuation(self, text: str) -> str:
        """
        Reduce repeated punctuation (e.g., '!!!' -> '!').
        
        Args:
            text: Input text.
            
        Returns:
            Text with reduced punctuation.
        """
        max_repeat = self.config.get('max_punct_repeat', 1)
        
        def replace_func(match):
            char = match.group(1)
            return char * max_repeat
        
        return self.punct_repeat_pattern.sub(replace_func, text)
    
    def normalize_whitespace(self, text: str) -> str:
        """
        Normalize whitespace (multiple spaces to single, handle tabs/newlines).
        
        Args:
            text: Input text.
            
        Returns:
            Text with normalized whitespace.
        """
        # Replace all whitespace types with single space
        result = self.whitespace_pattern.sub(' ', text)
        return result
    
    def to_lowercase(self, text: str) -> str:
        """
        Convert text to lowercase (preserving Kannada script).
        
        Args:
            text: Input text.
            
        Returns:
            Lowercased text.
        """
        return text.lower()
    
    def expand_contractions(self, text: str) -> str:
        """
        Expand English contractions.
        
        Args:
            text: Input text.
            
        Returns:
            Text with expanded contractions.
        """
        contractions = {
            "don't": "do not",
            "doesn't": "does not",
            "didn't": "did not",
            "won't": "will not",
            "wouldn't": "would not",
            "couldn't": "could not",
            "shouldn't": "should not",
            "can't": "cannot",
            "isn't": "is not",
            "aren't": "are not",
            "wasn't": "was not",
            "weren't": "were not",
            "haven't": "have not",
            "hasn't": "has not",
            "hadn't": "had not",
            "i'm": "i am",
            "you're": "you are",
            "he's": "he is",
            "she's": "she is",
            "it's": "it is",
            "we're": "we are",
            "they're": "they are",
            "i've": "i have",
            "you've": "you have",
            "we've": "we have",
            "they've": "they have",
            "i'll": "i will",
            "you'll": "you will",
            "he'll": "he will",
            "she'll": "she will",
            "we'll": "we will",
            "they'll": "they will",
            "i'd": "i would",
            "you'd": "you would",
            "he'd": "he would",
            "she'd": "she would",
            "we'd": "we would",
            "they'd": "they would",
            "that's": "that is",
            "there's": "there is",
            "what's": "what is",
            "who's": "who is",
            "let's": "let us",
            "gonna": "going to",
            "wanna": "want to",
            "gotta": "got to",
            "hafta": "have to",
            "kinda": "kind of",
            "sorta": "sort of",
            "dunno": "do not know",
            "ain't": "is not",
        }
        
        result = text
        for contraction, expansion in contractions.items():
            # Case-insensitive replacement
            pattern = re.compile(re.escape(contraction), re.IGNORECASE)
            result = pattern.sub(expansion, result)
        
        return result
    
    # ==================== Additional Helper Methods ====================
    
    def clean_text(self, text: str) -> str:
        """
        Basic text cleaning (whitespace and strip).
        
        Args:
            text: Input text.
            
        Returns:
            Cleaned text.
        """
        if not text:
            return ""
        
        result = self.normalize_whitespace(text)
        return result.strip()
    
    def remove_noise(self, text: str) -> str:
        """
        Remove noise elements (URLs, emails, hashtags, mentions).
        
        Args:
            text: Input text.
            
        Returns:
            Text with noise removed.
        """
        result = text
        result = self.remove_urls(result)
        result = self.remove_emails(result)
        result = self.remove_hashtag_ids(result)
        
        # Remove @mentions
        result = re.sub(r'@\w+', '', result)
        
        return self.normalize_whitespace(result).strip()
    
    def is_kannada_text(self, text: str) -> bool:
        """
        Check if text contains Kannada script.
        
        Args:
            text: Input text.
            
        Returns:
            True if Kannada script present.
        """
        return bool(self.kannada_pattern.search(text))
    
    def get_script_ratio(self, text: str) -> Dict[str, float]:
        """
        Calculate ratio of different scripts in text.
        
        Args:
            text: Input text.
            
        Returns:
            Dictionary with script ratios.
        """
        if not text:
            return {'kannada': 0.0, 'latin': 0.0, 'other': 0.0}
        
        kannada_chars = len(self.kannada_pattern.findall(text))
        latin_chars = len(re.findall(r'[a-zA-Z]', text))
        total_chars = len(re.findall(r'\S', text))
        
        if total_chars == 0:
            return {'kannada': 0.0, 'latin': 0.0, 'other': 0.0}
        
        kannada_ratio = kannada_chars / total_chars
        latin_ratio = latin_chars / total_chars
        other_ratio = 1.0 - kannada_ratio - latin_ratio
        
        return {
            'kannada': round(kannada_ratio, 3),
            'latin': round(latin_ratio, 3),
            'other': round(max(0, other_ratio), 3)
        }
    
    def normalize_batch(self, texts: List[str]) -> List[str]:
        """
        Normalize a batch of texts.
        
        Args:
            texts: List of input texts.
            
        Returns:
            List of normalized texts.
        """
        return [self.normalize(text) for text in texts]
    
    def tokenize_basic(self, text: str) -> List[str]:
        """
        Basic whitespace tokenization.
        
        Args:
            text: Input text.
            
        Returns:
            List of tokens.
        """
        normalized = self.normalize(text)
        return normalized.split()
    
    def get_stats(self, text: str) -> Dict[str, Any]:
        """
        Get text statistics before and after normalization.
        
        Args:
            text: Input text.
            
        Returns:
            Dictionary with text statistics.
        """
        normalized = self.normalize(text)
        
        return {
            'original_length': len(text),
            'normalized_length': len(normalized),
            'original_words': len(text.split()),
            'normalized_words': len(normalized.split()),
            'has_kannada_script': self.is_kannada_text(text),
            'script_ratio': self.get_script_ratio(text),
            'has_urls': bool(self.url_pattern.search(text)),
            'has_emails': bool(self.email_pattern.search(text)),
        }


# Convenience function for quick normalization
def normalize_text(text: str) -> str:
    """Quick text normalization without creating instance."""
    normalizer = TextNormalizer()
    return normalizer.normalize(text)


if __name__ == "__main__":
    # Quick test
    normalizer = TextNormalizer()
    
    test_texts = [
        "Nee sakkat DUMB idiya yaar machaa!!! #5a76",
        "stop talking nonsense nin matu beka illa don't you think?",
        "  elli hogidaroo   ade tara   madthiya  ",
        "noooooo pleeeease stooooop!!!",
        "check https://example.com this link",
        "ನೀನು ತುಂಬಾ ಚೆನ್ನಾಗಿದ್ದೀಯ"
    ]
    
    for text in test_texts:
        print(f"\nOriginal: {text}")
        print(f"Normalized: {normalizer.normalize(text)}")
        print(f"Stats: {normalizer.get_stats(text)}")