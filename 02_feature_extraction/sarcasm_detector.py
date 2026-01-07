# Sarcasm Detector
"""
SarcasmDetector: Sarcasm and irony detection for cyberbullying analysis.
Detects hidden insults, passive-aggressive patterns, and disguised harassment.
Optimized for Kannada-English code-mixed text.
"""

import re
import logging
from typing import List, Dict, Optional, Any, Union, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class SarcasmDetector:
    """
    Sarcasm and irony detector for cyberbullying detection.
    
    Detects:
    - Verbal irony (saying opposite of meaning)
    - Passive-aggressive patterns
    - Backhanded compliments
    - Sarcastic praise
    - Hidden insults
    
    Important for cyberbullying as:
    - Bullies often disguise insults as compliments
    - Sarcasm can be more hurtful than direct attacks
    - Passive-aggressive behavior is a form of harassment
    
    Attributes:
        config: Configuration dictionary
        threshold: Detection threshold
    """
    
    # Sarcasm indicators
    SARCASM_MARKERS = {
        'punctuation': [
            r'\.{3,}',           # Ellipsis
            r'!+\?+',            # !? combinations
            r'\?+!+',            # ?! combinations
            r'~+',               # Tildes
        ],
        'phrases': [
            'oh really', 'yeah right', 'sure', 'of course', 'obviously',
            'totally', 'absolutely', 'definitely', 'wow', 'great job',
            'nice one', 'good for you', 'thanks a lot', 'how nice',
            'how wonderful', 'so smart', 'so clever', 'genius',
            'brilliant', 'amazing', 'fantastic', 'incredible',
        ],
        'kannada_markers': [
            'houdu houdu',  # "yes yes" sarcastically
            'tumba chennag', 'tumba smart',  # "very nice/smart"
            'super guru', 'super maga',  # Sarcastic praise
            'nodu nodu',  # "see see"
            'yaake illa',  # "why not" sarcastically
        ]
    }
    
    # Contrast patterns (positive + negative = sarcasm)
    CONTRAST_PATTERNS = {
        'positive_words': [
            'love', 'great', 'wonderful', 'amazing', 'brilliant', 'smart',
            'clever', 'nice', 'good', 'best', 'super', 'awesome',
            'beautiful', 'perfect', 'excellent', 'fantastic',
            'chennag', 'tumba', 'sakkat', 'super'  # Kannada positive
        ],
        'negative_indicators': [
            'but', 'though', 'however', 'except', 'only', 'just',
            'not', "n't", 'never', 'nothing', 'nobody',
            'illa', 'beda', 'alla'  # Kannada negation
        ]
    }
    
    # Passive-aggressive patterns
    PASSIVE_AGGRESSIVE = [
        "i'm fine", "it's fine", "whatever", "if you say so",
        "no offense but", "just saying", "i'm just being honest",
        "with all due respect", "i don't want to be mean but",
        "don't take this the wrong way", "no hard feelings",
        "i'm not angry", "it's okay i guess", "if that's what you want",
        "i thought you knew", "my bad", "sorry not sorry",
    ]
    
    # Backhanded compliment patterns
    BACKHANDED_PATTERNS = [
        r"you'?re\s+(?:so|pretty|actually)\s+\w+\s+for\s+(?:a|an)\s+\w+",  # "you're smart for a..."
        r"(?:good|nice|great)\s+(?:for|considering)\s+",  # "good for/considering..."
        r"(?:at least|at\s+least)\s+you",  # "at least you..."
        r"you\s+(?:almost|nearly)\s+(?:look|seem)",  # "you almost look..."
        r"i'?m\s+surprised\s+(?:you|that\s+you)",  # "i'm surprised you..."
    ]
    
    # Emoji patterns for sarcasm detection
    SARCASM_EMOJI_COMBOS = [
        ('positive_text', ['😏', '🙄', '😒']),  # Positive text with negative emoji
        ('negative_text', ['😊', '😇', '🥰']),  # Negative text with positive emoji
    ]
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize SarcasmDetector.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or self._default_config()
        self.threshold = self.config.get('threshold', 0.5)
        
        # Compile patterns
        self._compile_patterns()
    
    def _default_config(self) -> Dict[str, Any]:
        """Return default configuration."""
        return {
            'threshold': 0.5,
            'include_emoji_analysis': True,
            'include_contrast_detection': True,
            'include_passive_aggressive': True,
            'use_weighted_scoring': True
        }
    
    def _compile_patterns(self) -> None:
        """Compile regex patterns for efficiency."""
        # Punctuation patterns
        self.punct_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.SARCASM_MARKERS['punctuation']
        ]
        
        # Backhanded patterns
        self.backhanded_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.BACKHANDED_PATTERNS
        ]
        
        # Emoji pattern
        self.emoji_pattern = re.compile(
            r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F900-\U0001F9FF]+'
        )
        
        # Quote pattern (quoting something sarcastically)
        self.quote_pattern = re.compile(r'"([^"]+)"')
    
    # =========================================================================
    # Detection Methods
    # =========================================================================
    def detect_sarcasm_markers(self, text: str) -> Dict[str, Any]:
        """
        Detect explicit sarcasm markers.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with marker detection results
        """
        text_lower = text.lower()
        
        # Punctuation markers
        punct_count = sum(
            1 for p in self.punct_patterns if p.search(text)
        )
        
        # Phrase markers
        phrase_count = sum(
            1 for phrase in self.SARCASM_MARKERS['phrases']
            if phrase in text_lower
        )
        
        # Kannada markers
        kannada_count = sum(
            1 for marker in self.SARCASM_MARKERS['kannada_markers']
            if marker in text_lower
        )
        
        # Quotes (often used sarcastically)
        quotes = self.quote_pattern.findall(text)
        
        # All caps words (emphasis/sarcasm)
        caps_words = re.findall(r'\b[A-Z]{2,}\b', text)
        
        total_markers = punct_count + phrase_count + kannada_count + len(quotes)
        
        return {
            'punctuation_markers': punct_count,
            'phrase_markers': phrase_count,
            'kannada_markers': kannada_count,
            'quoted_phrases': quotes,
            'caps_emphasis': caps_words,
            'total_markers': total_markers,
            'has_sarcasm_markers': total_markers > 0
        }
    
    def detect_contrast(self, text: str) -> Dict[str, Any]:
        """
        Detect contrast patterns (positive + negative = potential sarcasm).
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with contrast detection results
        """
        text_lower = text.lower()
        words = text_lower.split()
        
        # Count positive and negative indicators
        positive_count = sum(
            1 for w in self.CONTRAST_PATTERNS['positive_words']
            if w in text_lower
        )
        
        negative_count = sum(
            1 for w in self.CONTRAST_PATTERNS['negative_indicators']
            if w in text_lower
        )
        
        # Check for but/however structure
        has_contrast_word = any(
            w in text_lower for w in ['but', 'however', 'though', 'except']
        )
        
        # Sentiment contrast score
        contrast_score = 0
        if positive_count > 0 and negative_count > 0:
            contrast_score = min(positive_count, negative_count) / max(positive_count, negative_count)
        
        # Check for positive words followed by negative
        has_positive_negative_sequence = False
        for i, word in enumerate(words[:-1]):
            if word in self.CONTRAST_PATTERNS['positive_words']:
                # Check next few words for negative
                for j in range(i+1, min(i+5, len(words))):
                    if words[j] in self.CONTRAST_PATTERNS['negative_indicators']:
                        has_positive_negative_sequence = True
                        break
        
        return {
            'positive_word_count': positive_count,
            'negative_indicator_count': negative_count,
            'has_contrast_word': has_contrast_word,
            'contrast_score': round(contrast_score, 3),
            'has_positive_negative_sequence': has_positive_negative_sequence,
            'is_contrasted': positive_count > 0 and (has_contrast_word or negative_count > 0)
        }
    
    def detect_passive_aggressive(self, text: str) -> Dict[str, Any]:
        """
        Detect passive-aggressive patterns.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with passive-aggressive detection results
        """
        text_lower = text.lower()
        
        matches = []
        for pattern in self.PASSIVE_AGGRESSIVE:
            if pattern in text_lower:
                matches.append(pattern)
        
        # Fine/okay with punctuation (dismissive)
        dismissive_pattern = re.search(r'\b(?:fine|okay|whatever)\.+\s*$', text_lower)
        
        # Backhanded compliments
        backhanded = []
        for pattern in self.backhanded_patterns:
            match = pattern.search(text_lower)
            if match:
                backhanded.append(match.group())
        
        return {
            'passive_aggressive_phrases': matches,
            'passive_aggressive_count': len(matches),
            'has_dismissive_ending': dismissive_pattern is not None,
            'backhanded_compliments': backhanded,
            'is_passive_aggressive': len(matches) > 0 or len(backhanded) > 0
        }
    
    def detect_emoji_contrast(self, text: str) -> Dict[str, Any]:
        """
        Detect emoji-text sentiment contrast.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with emoji contrast detection
        """
        # Extract emojis
        emojis = self.emoji_pattern.findall(text)
        text_without_emoji = self.emoji_pattern.sub('', text).strip()
        
        if not emojis or not text_without_emoji:
            return {
                'has_emoji_contrast': False,
                'emojis_found': emojis,
                'emoji_contrast_score': 0.0
            }
        
        # Analyze text sentiment (simple)
        text_lower = text_without_emoji.lower()
        
        positive_text = any(
            w in text_lower for w in self.CONTRAST_PATTERNS['positive_words']
        )
        negative_text = any(
            w in text_lower for w in ['hate', 'stupid', 'idiot', 'ugly', 'bad', 'terrible', 
                                       'irritating', 'annoying', 'thotha', 'singri']
        )
        
        # Check emoji sentiment
        positive_emojis = {'😊', '😃', '😄', '🥰', '😇', '👍', '❤️', '✨', '🎉'}
        negative_emojis = {'😡', '😠', '🙄', '😒', '😤', '💢', '👎', '😏'}
        
        has_positive_emoji = any(e in positive_emojis for e in ''.join(emojis))
        has_negative_emoji = any(e in negative_emojis for e in ''.join(emojis))
        
        # Contrast detection
        has_contrast = (positive_text and has_negative_emoji) or (negative_text and has_positive_emoji)
        
        contrast_score = 0.0
        if has_contrast:
            contrast_score = 0.7
            if positive_text and has_negative_emoji:
                contrast_score = 0.8  # More likely sarcasm
        
        return {
            'has_emoji_contrast': has_contrast,
            'emojis_found': emojis,
            'text_sentiment': 'positive' if positive_text else ('negative' if negative_text else 'neutral'),
            'emoji_sentiment': 'positive' if has_positive_emoji else ('negative' if has_negative_emoji else 'neutral'),
            'emoji_contrast_score': round(contrast_score, 3)
        }
    
    # =========================================================================
    # Combined Detection
    # =========================================================================
    def detect(self, texts: Union[str, List[str]]) -> List[Dict[str, Any]]:
        """
        Perform comprehensive sarcasm detection.
        
        Args:
            texts: Text or list of texts
            
        Returns:
            List of detection result dictionaries
        """
        if isinstance(texts, str):
            texts = [texts]
        
        results = []
        
        for text in texts:
            result = {}
            
            # Run all detectors
            markers = self.detect_sarcasm_markers(text)
            result.update(markers)
            
            if self.config.get('include_contrast_detection', True):
                contrast = self.detect_contrast(text)
                result.update(contrast)
            
            if self.config.get('include_passive_aggressive', True):
                passive = self.detect_passive_aggressive(text)
                result.update(passive)
            
            if self.config.get('include_emoji_analysis', True):
                emoji = self.detect_emoji_contrast(text)
                result.update(emoji)
            
            # Calculate overall sarcasm score
            sarcasm_score = self._calculate_sarcasm_score(result)
            result['sarcasm_score'] = round(sarcasm_score, 3)
            result['is_sarcastic'] = sarcasm_score >= self.threshold
            result['sarcasm_type'] = self._classify_sarcasm_type(result)
            
            results.append(result)
        
        return results
    
    def _calculate_sarcasm_score(self, features: Dict[str, Any]) -> float:
        """Calculate overall sarcasm probability score."""
        score = 0.0
        
        # Marker-based scoring
        markers = features.get('total_markers', 0)
        score += min(markers * 0.15, 0.3)
        
        # Contrast-based scoring
        if features.get('is_contrasted', False):
            score += 0.2
        if features.get('contrast_score', 0) > 0.5:
            score += 0.1
        
        # Passive-aggressive scoring
        if features.get('is_passive_aggressive', False):
            score += 0.25
        pa_count = features.get('passive_aggressive_count', 0)
        score += min(pa_count * 0.1, 0.2)
        
        # Backhanded compliment
        if features.get('backhanded_compliments'):
            score += 0.3
        
        # Emoji contrast
        if features.get('has_emoji_contrast', False):
            score += features.get('emoji_contrast_score', 0) * 0.3
        
        return min(score, 1.0)
    
    def _classify_sarcasm_type(self, features: Dict[str, Any]) -> str:
        """Classify the type of sarcasm detected."""
        if not features.get('is_sarcastic', False):
            return 'none'
        
        if features.get('backhanded_compliments'):
            return 'backhanded_compliment'
        
        if features.get('is_passive_aggressive', False):
            return 'passive_aggressive'
        
        if features.get('has_emoji_contrast', False):
            return 'emoji_contrast'
        
        if features.get('is_contrasted', False):
            return 'verbal_irony'
        
        if features.get('total_markers', 0) > 0:
            return 'marked_sarcasm'
        
        return 'subtle_sarcasm'
    
    def extract_numeric_features(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Extract numeric sarcasm features.
        
        Returns:
            Feature matrix (n_samples, n_features)
        """
        all_features = self.detect(texts)
        
        numeric_keys = [
            'total_markers', 'punctuation_markers', 'phrase_markers',
            'positive_word_count', 'negative_indicator_count', 'contrast_score',
            'passive_aggressive_count', 'emoji_contrast_score', 'sarcasm_score'
        ]
        
        matrix = []
        for features in all_features:
            row = [features.get(key, 0) for key in numeric_keys]
            matrix.append(row)
        
        return np.array(matrix)
    
    def get_feature_names(self) -> List[str]:
        """Get list of numeric feature names."""
        return [
            'total_markers', 'punctuation_markers', 'phrase_markers',
            'positive_word_count', 'negative_indicator_count', 'contrast_score',
            'passive_aggressive_count', 'emoji_contrast_score', 'sarcasm_score'
        ]
    
    def __repr__(self) -> str:
        """String representation."""
        return f"SarcasmDetector(threshold={self.threshold})"


if __name__ == "__main__":
    print("SarcasmDetector Test")
    print("=" * 50)
    
    detector = SarcasmDetector()
    
    test_texts = [
        "Oh yeah, you're SO smart 🙄",
        "Great job on the presentation... I guess",
        "Nice work for someone like you",
        "I'm not angry. It's fine. Whatever.",
        "You're actually pretty smart for a newbie",
        "tumba chennag maadidya... houdu houdu",
        "Thanks a lot for 'helping' me",
        "This is a normal message.",
        "You did well! 👍",
    ]
    
    for text in test_texts:
        result = detector.detect(text)[0]
        print(f"\nText: {text}")
        print(f"  Sarcasm score: {result.get('sarcasm_score')}")
        print(f"  Is sarcastic: {result.get('is_sarcastic')}")
        print(f"  Type: {result.get('sarcasm_type')}")
