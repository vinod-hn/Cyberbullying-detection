# Emoji Handler
"""
EmojiHandler: Comprehensive emoji processing for cyberbullying detection.
Handles emoji extraction, sentiment analysis, and bullying pattern detection.
Optimized for Kannada-English code-mixed text.
"""

import re
import json
import os
from typing import List, Dict, Optional, Tuple, Any, Union
from collections import Counter
import unicodedata


class EmojiHandler:
    """
    Comprehensive emoji handler for cyberbullying detection.
    
    Features:
    - Emoji detection and extraction
    - Sentiment analysis based on emoji semantics
    - Cyberbullying pattern detection
    - Support for emoji modifiers (skin tones, ZWJ sequences)
    - Integration with emoji_semantics.json lexicon
    """
    
    # Unicode ranges for emojis
    EMOJI_PATTERN = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # Emoticons
        "\U0001F300-\U0001F5FF"  # Symbols & Pictographs
        "\U0001F680-\U0001F6FF"  # Transport & Map
        "\U0001F700-\U0001F77F"  # Alchemical Symbols
        "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
        "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
        "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        "\U0001FA00-\U0001FA6F"  # Chess Symbols
        "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        "\U00002702-\U000027B0"  # Dingbats
        "\U000024C2-\U0001F251"  # Enclosed characters
        "\U0001F1E0-\U0001F1FF"  # Flags (iOS)
        "\U00002600-\U000026FF"  # Misc symbols
        "\U00002700-\U000027BF"  # Dingbats
        "\U0000FE00-\U0000FE0F"  # Variation Selectors
        "\U0001F000-\U0001F02F"  # Mahjong Tiles
        "\U0001F0A0-\U0001F0FF"  # Playing Cards
        "]+",
        flags=re.UNICODE
    )
    
    # Skin tone modifiers
    SKIN_TONE_MODIFIERS = [
        '\U0001F3FB',  # Light skin tone
        '\U0001F3FC',  # Medium-light skin tone
        '\U0001F3FD',  # Medium skin tone
        '\U0001F3FE',  # Medium-dark skin tone
        '\U0001F3FF',  # Dark skin tone
    ]
    
    # Zero Width Joiner
    ZWJ = '\u200D'
    
    # Offensive/Vulgar emojis commonly used in cyberbullying
    OFFENSIVE_EMOJIS = [
        '🖕',  # Middle finger
        '🖕🏻', '🖕🏼', '🖕🏽', '🖕🏾', '🖕🏿',  # Middle finger with skin tones
    ]
    
    # Aggressive/Angry emojis
    AGGRESSIVE_EMOJIS = [
        '😡',  # Angry face (pouting)
        '🤬',  # Face with symbols on mouth (swearing)
        '👊',  # Oncoming fist (punch)
        '👊🏻', '👊🏼', '👊🏽', '👊🏾', '👊🏿',  # Fist with skin tones
        '💢',  # Anger symbol
        '😤',  # Face with steam from nose
        '🔥',  # Fire (can indicate rage)
        '💀',  # Skull (death threat context)
        '☠️',  # Skull and crossbones
    ]
    
    # Threatening/Violent emojis
    THREATENING_EMOJIS = [
        '🔪',  # Kitchen knife
        '🗡️',  # Dagger
        '⚔️',  # Crossed swords
        '🔫',  # Pistol/Water gun
        '💣',  # Bomb
        '💥',  # Collision/Explosion
        '⚰️',  # Coffin
        '🪦',  # Headstone
        '☠️',  # Skull and crossbones
        '💀',  # Skull
        '👹',  # Ogre (Japanese demon)
        '👺',  # Goblin
        '😈',  # Smiling face with horns
        '👿',  # Angry face with horns
    ]
    
    # Mocking/Insulting emojis
    MOCKING_EMOJIS = [
        '🤡',  # Clown face (calling someone a clown)
        '🤪',  # Zany face (mocking crazy)
        '😜',  # Winking face with tongue
        '🙄',  # Face with rolling eyes
        '😏',  # Smirking face
        '💅',  # Nail polish (dismissive)
        '🤏',  # Pinching hand (size shaming)
        '🤣',  # Rolling on floor laughing (excessive mockery)
        '😂',  # Face with tears of joy (can be mocking)
        '🥴',  # Woozy face (mocking stupidity)
        '🤓',  # Nerd face (mocking intelligence)
        '🥱',  # Yawning face (dismissive/bored)
    ]
    
    # Body shaming emojis
    BODY_SHAMING_EMOJIS = [
        '🐷',  # Pig face (fat shaming)
        '🐖',  # Pig (fat shaming)
        '🐮',  # Cow face (fat shaming)
        '🐄',  # Cow (fat shaming)
        '🦛',  # Hippopotamus (fat shaming)
        '🐘',  # Elephant (fat shaming)
        '🤮',  # Vomiting face (disgust)
        '🤢',  # Nauseated face (disgust)
        '💩',  # Pile of poo (insult)
        '🦴',  # Bone (skinny shaming)
        '🦷',  # Tooth (appearance shaming)
        '👃',  # Nose (appearance shaming)
    ]
    
    # Sexual harassment emojis
    SEXUAL_HARASSMENT_EMOJIS = [
        '🍆',  # Eggplant (phallic symbol)
        '🍑',  # Peach (buttocks symbol)
        '💦',  # Sweat droplets (sexual context)
        '🍌',  # Banana (phallic symbol)
        '👅',  # Tongue
        '👀',  # Eyes (creepy staring)
        '😈',  # Smiling face with horns
        '🥵',  # Hot face (sexually suggestive)
        '😘',  # Face blowing a kiss (unwanted)
        '😍',  # Heart eyes (unwanted attention)
        '🤤',  # Drooling face
        '💋',  # Kiss mark
    ]
    
    # All bad emojis combined for quick lookup
    ALL_BAD_EMOJIS = set(
        OFFENSIVE_EMOJIS + AGGRESSIVE_EMOJIS + THREATENING_EMOJIS +
        MOCKING_EMOJIS + BODY_SHAMING_EMOJIS + SEXUAL_HARASSMENT_EMOJIS
    )
    
    def __init__(self, semantics_path: Optional[str] = None):
        """
        Initialize EmojiHandler.
        
        Args:
            semantics_path: Path to emoji_semantics.json file.
                          If None, attempts to load from default location.
        """
        self.semantics = {}
        self._load_semantics(semantics_path)
        self._build_emoji_lookup()
    
    def _load_semantics(self, semantics_path: Optional[str] = None) -> None:
        """Load emoji semantics from JSON file."""
        if semantics_path is None:
            # Try default locations
            possible_paths = [
                os.path.join(os.path.dirname(__file__), '..', '00_data', 'lexicon', 'emoji_semantics.json'),
                os.path.join(os.path.dirname(__file__), 'emoji_semantics.json'),
                'emoji_semantics.json',
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    semantics_path = path
                    break
        
        if semantics_path and os.path.exists(semantics_path):
            try:
                with open(semantics_path, 'r', encoding='utf-8') as f:
                    self.semantics = json.load(f)
            except (json.JSONDecodeError, IOError):
                self.semantics = {}
    
    def _build_emoji_lookup(self) -> None:
        """Build flat lookup dictionaries for quick access."""
        self._emoji_sentiment = {}
        self._emoji_names = {}
        self._emoji_cyberbullying = {}
        self._emoji_context = {}
        
        # Process negative emojis
        if 'negative_emojis' in self.semantics:
            for category, emojis in self.semantics['negative_emojis'].items():
                for emoji, data in emojis.items():
                    self._emoji_sentiment[emoji] = data.get('sentiment', -0.5)
                    self._emoji_names[emoji] = data.get('name', '')
                    self._emoji_cyberbullying[emoji] = data.get('cyberbullying_association', 'medium')
                    self._emoji_context[emoji] = data.get('common_context', '')
        
        # Process positive emojis
        if 'positive_emojis' in self.semantics:
            for category, emojis in self.semantics['positive_emojis'].items():
                for emoji, data in emojis.items():
                    self._emoji_sentiment[emoji] = data.get('sentiment', 0.5)
                    self._emoji_names[emoji] = data.get('name', '')
                    self._emoji_cyberbullying[emoji] = data.get('cyberbullying_association', 'none')
                    self._emoji_context[emoji] = data.get('common_context', '')
        
        # Process neutral emojis
        if 'neutral_emojis' in self.semantics:
            for emoji, data in self.semantics['neutral_emojis'].items():
                self._emoji_sentiment[emoji] = data.get('sentiment', 0)
                self._emoji_names[emoji] = data.get('name', '')
                self._emoji_cyberbullying[emoji] = data.get('cyberbullying_association', 'none')
                self._emoji_context[emoji] = data.get('common_context', '')
        
        # Process ambiguous emojis
        if 'ambiguous_emojis' in self.semantics:
            for emoji, data in self.semantics['ambiguous_emojis'].items():
                sentiment_range = data.get('sentiment_range', [0, 0])
                self._emoji_sentiment[emoji] = sum(sentiment_range) / 2  # Average
                self._emoji_names[emoji] = data.get('name', '')
                self._emoji_cyberbullying[emoji] = data.get('cyberbullying_association', 'context_dependent')
        
        # Build severity lookup from detection_weights
        self._severity_lookup = {}
        if 'detection_weights' in self.semantics:
            weights = self.semantics['detection_weights']
            for emoji in weights.get('critical_severity', []):
                self._severity_lookup[emoji] = 'critical'
            for emoji in weights.get('high_severity', []):
                self._severity_lookup[emoji] = 'high'
            for emoji in weights.get('medium_severity', []):
                self._severity_lookup[emoji] = 'medium'
            for emoji in weights.get('low_severity', []):
                self._severity_lookup[emoji] = 'low'
            for emoji in weights.get('positive_indicators', []):
                self._severity_lookup[emoji] = 'positive'
    
    # ==================== Core Detection Methods ====================
    
    def contains_emoji(self, text: str) -> bool:
        """
        Check if text contains any emoji.
        
        Args:
            text: Input text to check
            
        Returns:
            True if text contains at least one emoji
        """
        if not text:
            return False
        return bool(self.EMOJI_PATTERN.search(text))
    
    def extract_emojis(self, text: str) -> List[str]:
        """
        Extract all emojis from text.
        
        Args:
            text: Input text
            
        Returns:
            List of emojis found in text
        """
        if not text:
            return []
        
        emojis = []
        matches = self.EMOJI_PATTERN.findall(text)
        
        for match in matches:
            # Split compound matches into individual emojis
            for char in match:
                if self._is_emoji_char(char):
                    emojis.append(char)
        
        # Also handle ZWJ sequences as single emojis
        zwj_emojis = self._extract_zwj_sequences(text)
        if zwj_emojis:
            # Replace individual components with ZWJ sequence
            return zwj_emojis if zwj_emojis else emojis
        
        return emojis
    
    def _is_emoji_char(self, char: str) -> bool:
        """Check if a single character is an emoji."""
        if not char:
            return False
        # Check if it's in emoji unicode blocks
        code = ord(char)
        emoji_ranges = [
            (0x1F600, 0x1F64F),  # Emoticons
            (0x1F300, 0x1F5FF),  # Symbols & Pictographs
            (0x1F680, 0x1F6FF),  # Transport & Map
            (0x1F900, 0x1F9FF),  # Supplemental Symbols
            (0x2600, 0x26FF),    # Misc symbols
            (0x2700, 0x27BF),    # Dingbats
            (0x1F1E0, 0x1F1FF),  # Flags
        ]
        for start, end in emoji_ranges:
            if start <= code <= end:
                return True
        return False
    
    def _extract_zwj_sequences(self, text: str) -> List[str]:
        """Extract ZWJ (Zero Width Joiner) emoji sequences."""
        if self.ZWJ not in text:
            return []
        
        # Pattern for ZWJ sequences (e.g., family emojis)
        zwj_pattern = re.compile(
            r'(?:[\U0001F468-\U0001F469][\U0001F3FB-\U0001F3FF]?\u200D?)+'
            r'(?:[\U0001F466-\U0001F467\U0001F468-\U0001F469][\U0001F3FB-\U0001F3FF]?)*'
        )
        
        matches = zwj_pattern.findall(text)
        return [m for m in matches if m]
    
    def count_emojis(self, text: str) -> int:
        """
        Count total number of emojis in text.
        
        Args:
            text: Input text
            
        Returns:
            Total count of emojis
        """
        return len(self.extract_emojis(text))
    
    def count_unique_emojis(self, text: str) -> int:
        """
        Count unique emojis in text.
        
        Args:
            text: Input text
            
        Returns:
            Count of unique emojis
        """
        return len(set(self.extract_emojis(text)))
    
    def get_emoji_frequency(self, text: str) -> Dict[str, int]:
        """
        Get frequency of each emoji in text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary mapping emoji to its count
        """
        emojis = self.extract_emojis(text)
        return dict(Counter(emojis))
    
    # ==================== Sentiment Analysis Methods ====================
    
    def get_emoji_sentiment(self, emoji: str) -> float:
        """
        Get sentiment score for a single emoji.
        
        Args:
            emoji: Single emoji character
            
        Returns:
            Sentiment score (-1.0 to 1.0)
        """
        return self._emoji_sentiment.get(emoji, 0.0)
    
    def get_text_emoji_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Get aggregate emoji sentiment for text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with sentiment analysis results
        """
        emojis = self.extract_emojis(text)
        
        if not emojis:
            return {
                'emojis': [],
                'count': 0,
                'average_sentiment': 0.0,
                'overall': 'neutral'
            }
        
        sentiments = [self.get_emoji_sentiment(e) for e in emojis]
        avg_sentiment = sum(sentiments) / len(sentiments)
        
        if avg_sentiment > 0.3:
            overall = 'positive'
        elif avg_sentiment < -0.3:
            overall = 'negative'
        else:
            overall = 'neutral'
        
        return {
            'emojis': emojis,
            'count': len(emojis),
            'individual_sentiments': dict(zip(emojis, sentiments)),
            'average_sentiment': avg_sentiment,
            'overall': overall
        }
    
    # ==================== Cyberbullying Detection Methods ====================
    
    def is_aggressive_emoji(self, emoji: str) -> bool:
        """
        Check if emoji is associated with aggression.
        
        Args:
            emoji: Single emoji
            
        Returns:
            True if emoji is aggressive
        """
        # Check in predefined aggressive emoji lists
        if emoji in self.AGGRESSIVE_EMOJIS or emoji in self.THREATENING_EMOJIS:
            return True
        if emoji in self.OFFENSIVE_EMOJIS:
            return True
        # Also check from semantics
        association = self._emoji_cyberbullying.get(emoji, 'none')
        return association in ['high', 'critical']
    
    def is_body_shaming_emoji(self, emoji: str) -> bool:
        """
        Check if emoji is commonly used for body shaming.
        
        Args:
            emoji: Single emoji
            
        Returns:
            True if emoji is associated with body shaming
        """
        return emoji in self.BODY_SHAMING_EMOJIS
    
    def is_offensive_emoji(self, emoji: str) -> bool:
        """
        Check if emoji is offensive (middle finger, etc.).
        
        Args:
            emoji: Single emoji
            
        Returns:
            True if emoji is offensive
        """
        return emoji in self.OFFENSIVE_EMOJIS
    
    def is_sexual_harassment_emoji(self, emoji: str) -> bool:
        """
        Check if emoji is commonly used in sexual harassment.
        
        Args:
            emoji: Single emoji
            
        Returns:
            True if emoji is associated with sexual harassment
        """
        return emoji in self.SEXUAL_HARASSMENT_EMOJIS
    
    def is_mocking_emoji(self, emoji: str) -> bool:
        """
        Check if emoji is used for mocking/insulting.
        
        Args:
            emoji: Single emoji
            
        Returns:
            True if emoji is associated with mocking
        """
        return emoji in self.MOCKING_EMOJIS
    
    def is_bad_emoji(self, emoji: str) -> bool:
        """
        Check if emoji is any type of bad/harmful emoji.
        
        Args:
            emoji: Single emoji
            
        Returns:
            True if emoji is in any bad emoji category
        """
        return emoji in self.ALL_BAD_EMOJIS
    
    def detect_offensive_emojis(self, text: str) -> Dict[str, Any]:
        """
        Detect offensive emojis like middle finger in text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with offensive emoji detection results
        """
        emojis = self.extract_emojis(text)
        
        found_offensive = [e for e in emojis if e in self.OFFENSIVE_EMOJIS]
        
        if found_offensive:
            return {
                'has_offensive': True,
                'offensive_emojis': found_offensive,
                'count': len(found_offensive),
                'severity': 'high',
                'detail': f"Offensive emojis found: {', '.join(found_offensive)}"
            }
        
        return {
            'has_offensive': False,
            'offensive_emojis': [],
            'count': 0,
            'severity': 'none',
            'detail': None
        }
    
    def detect_sexual_harassment_emojis(self, text: str) -> Dict[str, Any]:
        """
        Detect sexual harassment emoji patterns.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with sexual harassment detection results
        """
        emojis = self.extract_emojis(text)
        freq = self.get_emoji_frequency(text)
        
        found_sexual = [e for e in emojis if e in self.SEXUAL_HARASSMENT_EMOJIS]
        
        if not found_sexual:
            return {
                'has_sexual_content': False,
                'emojis': [],
                'severity': 'none',
                'detail': None
            }
        
        # Check for explicit combinations
        explicit_combos = [
            ('🍆', '💦'), ('🍆', '🍑'), ('🍆', '👅'),
            ('🍌', '💦'), ('🍑', '💦'), ('👅', '💦')
        ]
        
        emoji_set = set(emojis)
        has_explicit_combo = any(
            e1 in emoji_set and e2 in emoji_set
            for e1, e2 in explicit_combos
        )
        
        # Determine severity
        if has_explicit_combo:
            severity = 'critical'
        elif len(found_sexual) >= 3:
            severity = 'high'
        elif len(found_sexual) >= 2:
            severity = 'medium'
        else:
            severity = 'low'
        
        return {
            'has_sexual_content': True,
            'emojis': found_sexual,
            'has_explicit_combo': has_explicit_combo,
            'severity': severity,
            'detail': f"Sexual harassment emojis found: {', '.join(found_sexual)}"
        }
    
    def detect_bullying_pattern(self, text: str) -> Dict[str, Any]:
        """
        Detect bullying patterns in emoji usage.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with pattern detection results
        """
        emojis = self.extract_emojis(text)
        
        result = {
            'has_pattern': False,
            'pattern_type': None,
            'severity': 'none',
            'details': []
        }
        
        if not emojis:
            return result
        
        # Check for repeated negative emojis
        freq = self.get_emoji_frequency(text)
        for emoji, count in freq.items():
            if count >= 3 and self.is_aggressive_emoji(emoji):
                result['has_pattern'] = True
                result['pattern_type'] = 'repeated_aggressive'
                result['severity'] = 'high'
                result['details'].append(f"Repeated aggressive emoji: {emoji} x{count}")
        
        # Check for threatening combinations
        threat_detected = self.detect_threatening_pattern(text)
        if threat_detected and threat_detected.get('is_threat'):
            result['has_pattern'] = True
            result['pattern_type'] = 'threat'
            result['severity'] = 'critical'
            result['details'].append(threat_detected.get('detail', ''))
        
        # Check for mocking pattern
        mock_detected = self.detect_mocking_pattern(text)
        if mock_detected and mock_detected.get('is_mocking'):
            result['has_pattern'] = True
            result['pattern_type'] = 'mockery'
            if result['severity'] != 'critical':
                result['severity'] = 'medium'
            result['details'].append(f"Mocking pattern: {mock_detected.get('pattern', '')}")
        
        # Check for offensive emojis (middle finger, etc.)
        offensive_detected = self.detect_offensive_emojis(text)
        if offensive_detected and offensive_detected.get('has_offensive'):
            result['has_pattern'] = True
            result['pattern_type'] = 'offensive'
            result['severity'] = 'high'
            result['details'].append(offensive_detected.get('detail', ''))
        
        # Check for sexual harassment emojis
        sexual_detected = self.detect_sexual_harassment_emojis(text)
        if sexual_detected and sexual_detected.get('has_sexual_content'):
            result['has_pattern'] = True
            result['pattern_type'] = 'sexual_harassment'
            if sexual_detected.get('severity') == 'critical':
                result['severity'] = 'critical'
            elif result['severity'] not in ['critical', 'high']:
                result['severity'] = 'high'
            result['details'].append(sexual_detected.get('detail', ''))
        
        # Check for body shaming emojis
        body_shaming = [e for e in emojis if self.is_body_shaming_emoji(e)]
        if body_shaming:
            result['has_pattern'] = True
            result['pattern_type'] = 'body_shaming'
            if result['severity'] not in ['critical', 'high']:
                result['severity'] = 'medium'
            result['details'].append(f"Body shaming emojis: {', '.join(body_shaming)}")
        
        return result
    
    def detect_emoji_spam(self, text: str) -> bool:
        """
        Detect emoji spam (repeated same emoji).
        
        Args:
            text: Input text
            
        Returns:
            True if emoji spam detected
        """
        freq = self.get_emoji_frequency(text)
        # Spam if any emoji repeated 4+ times
        return any(count >= 4 for count in freq.values())
    
    def detect_threatening_pattern(self, text: str) -> Dict[str, Any]:
        """
        Detect threatening emoji combinations.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with threat detection results
        """
        emojis = self.extract_emojis(text)
        
        found_threats = [e for e in emojis if e in self.THREATENING_EMOJIS]
        
        if found_threats:
            # Determine severity based on specific emojis
            critical_threats = ['🔪', '🔫', '💣', '🗡️', '⚔️']
            is_critical = any(e in critical_threats for e in found_threats)
            return {
                'is_threat': True,
                'threat_emojis': found_threats,
                'severity': 'critical' if is_critical else 'high',
                'detail': f"Threatening emojis found: {', '.join(found_threats)}"
            }
        
        return {'is_threat': False, 'threat_emojis': [], 'severity': 'none'}
    
    def detect_mocking_pattern(self, text: str) -> Dict[str, Any]:
        """
        Detect mocking emoji patterns (e.g., clown emoji).
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with mockery detection results
        """
        emojis = self.extract_emojis(text)
        freq = self.get_emoji_frequency(text)
        
        # Check for any mocking emoji from the constant
        found_mocking = [e for e in emojis if e in self.MOCKING_EMOJIS]
        
        # Check for clown emoji (strong mockery indicator)
        if '🤡' in emojis:
            clown_count = freq.get('🤡', 0)
            return {
                'is_mocking': True,
                'intensity': 'high' if clown_count >= 2 else 'medium',
                'pattern': 'clown_mockery',
                'mocking_emojis': found_mocking
            }
        
        # Check for rolling eyes
        if '🙄' in emojis:
            return {
                'is_mocking': True,
                'intensity': 'medium',
                'pattern': 'eye_roll_dismissal',
                'mocking_emojis': found_mocking
            }
        
        # Check for repeated laughing at someone
        if freq.get('😂', 0) >= 3 or freq.get('🤣', 0) >= 3:
            return {
                'is_mocking': True,
                'intensity': 'medium',
                'pattern': 'excessive_laughing',
                'mocking_emojis': found_mocking
            }
        
        # Check for nail polish (dismissive)
        if '💅' in emojis:
            return {
                'is_mocking': True,
                'intensity': 'low',
                'pattern': 'dismissive',
                'mocking_emojis': found_mocking
            }
        
        # Check for pinching hand (size shaming)
        if '🤏' in emojis:
            return {
                'is_mocking': True,
                'intensity': 'medium',
                'pattern': 'size_shaming',
                'mocking_emojis': found_mocking
            }
        
        # Check for any other mocking emojis
        if found_mocking:
            return {
                'is_mocking': True,
                'intensity': 'low',
                'pattern': 'general_mockery',
                'mocking_emojis': found_mocking
            }
        
        return {'is_mocking': False, 'intensity': 'none', 'pattern': None, 'mocking_emojis': []}
    
    def detect_sarcasm(self, text: str) -> Dict[str, Any]:
        """
        Detect potential sarcasm (positive emoji with negative text).
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with sarcasm detection results
        """
        emojis = self.extract_emojis(text)
        
        # Positive emojis that might indicate sarcasm
        sarcasm_indicators = ['👏', '😊', '👍', '🙌', '💯']
        
        # Negative words that might indicate sarcasm when paired with positive emojis
        negative_words = ['fail', 'stupid', 'dumb', 'idiot', 'loser', 'pathetic', 
                         'waste', 'useless', 'ugly', 'terrible', 'awful', 'worst']
        
        text_lower = text.lower()
        has_negative_words = any(word in text_lower for word in negative_words)
        has_positive_emojis = any(e in sarcasm_indicators for e in emojis)
        
        if has_negative_words and has_positive_emojis:
            return {
                'is_sarcastic': True,
                'confidence': 0.7,
                'reason': 'Positive emoji with negative text content'
            }
        
        return {'is_sarcastic': False, 'confidence': 0.0, 'reason': None}
    
    def analyze_emoji_context(self, text: str, emoji: str) -> Dict[str, Any]:
        """
        Analyze emoji meaning in context.
        
        Args:
            text: Full text containing emoji
            emoji: Specific emoji to analyze
            
        Returns:
            Dictionary with context analysis
        """
        # Skull emoji context analysis
        if emoji == '💀':
            laughing_indicators = ['lol', 'lmao', 'dead', 'dying', 'joke', 'funny', 'haha', 'im dead']
            text_lower = text.lower()
            
            is_laughing = any(ind in text_lower for ind in laughing_indicators)
            
            if is_laughing:
                return {
                    'emoji': emoji,
                    'interpreted_as': 'laughing/humor',
                    'sentiment': 0.3,
                    'is_threatening': False
                }
            else:
                return {
                    'emoji': emoji,
                    'interpreted_as': 'potential_threat',
                    'sentiment': -0.7,
                    'is_threatening': True
                }
        
        # Default context
        return {
            'emoji': emoji,
            'interpreted_as': self._emoji_context.get(emoji, 'unknown'),
            'sentiment': self.get_emoji_sentiment(emoji),
            'is_threatening': emoji in ['🔪', '🔫', '💣', '☠️']
        }
    
    # ==================== Replacement Methods ====================
    
    def replace_with_text(self, text: str) -> str:
        """
        Replace emojis with their text descriptions.
        
        Args:
            text: Input text
            
        Returns:
            Text with emojis replaced by descriptions
        """
        result = text
        emojis = self.extract_emojis(text)
        
        for emoji in set(emojis):
            name = self._emoji_names.get(emoji, 'emoji')
            result = result.replace(emoji, f' [{name}] ')
        
        return ' '.join(result.split())  # Clean up extra spaces
    
    def replace_with_placeholder(self, text: str, placeholder: str = "[EMOJI]") -> str:
        """
        Replace all emojis with a placeholder.
        
        Args:
            text: Input text
            placeholder: String to replace emojis with
            
        Returns:
            Text with emojis replaced by placeholder
        """
        result = self.EMOJI_PATTERN.sub(placeholder, text)
        return result
    
    def remove_emojis(self, text: str) -> str:
        """
        Remove all emojis from text.
        
        Args:
            text: Input text
            
        Returns:
            Text with emojis removed
        """
        result = self.EMOJI_PATTERN.sub(' ', text)
        return ' '.join(result.split())  # Clean up extra spaces
    
    # ==================== Meaning/Metadata Methods ====================
    
    def get_meaning(self, emoji: str) -> str:
        """
        Get meaning/description of emoji.
        
        Args:
            emoji: Single emoji
            
        Returns:
            Text description of emoji
        """
        name = self._emoji_names.get(emoji)
        if name:
            return name.replace('_', ' ')
        
        # Fallback to unicodedata
        try:
            return unicodedata.name(emoji, 'unknown emoji').lower()
        except (TypeError, ValueError):
            return 'unknown emoji'
    
    def get_category(self, emoji: str) -> Optional[str]:
        """
        Get category of emoji.
        
        Args:
            emoji: Single emoji
            
        Returns:
            Category string or None
        """
        # Check in semantics structure
        for category_type in ['negative_emojis', 'positive_emojis']:
            if category_type in self.semantics:
                for category, emojis in self.semantics[category_type].items():
                    if emoji in emojis:
                        return category
        
        if emoji in self.semantics.get('neutral_emojis', {}):
            return 'neutral'
        
        if emoji in self.semantics.get('ambiguous_emojis', {}):
            return 'ambiguous'
        
        return None
    
    def get_unicode_name(self, emoji: str) -> str:
        """
        Get official Unicode name of emoji.
        
        Args:
            emoji: Single emoji
            
        Returns:
            Unicode name string
        """
        try:
            return unicodedata.name(emoji, 'UNKNOWN')
        except (TypeError, ValueError):
            return 'UNKNOWN'
    
    def get_semantic(self, emoji: str) -> Optional[Dict[str, Any]]:
        """
        Get full semantic data for emoji.
        
        Args:
            emoji: Single emoji
            
        Returns:
            Semantic data dictionary or None
        """
        # Search in all categories
        for category_type in ['negative_emojis', 'positive_emojis']:
            if category_type in self.semantics:
                for category, emojis in self.semantics[category_type].items():
                    if emoji in emojis:
                        return emojis[emoji]
        
        if emoji in self.semantics.get('neutral_emojis', {}):
            return self.semantics['neutral_emojis'][emoji]
        
        if emoji in self.semantics.get('ambiguous_emojis', {}):
            return self.semantics['ambiguous_emojis'][emoji]
        
        return None
    
    # ==================== Normalization Methods ====================
    
    def normalize_skin_tones(self, emoji: str) -> str:
        """
        Remove skin tone modifiers from emoji.
        
        Args:
            emoji: Emoji possibly with skin tone modifier
            
        Returns:
            Base emoji without skin tone
        """
        result = emoji
        for modifier in self.SKIN_TONE_MODIFIERS:
            result = result.replace(modifier, '')
        return result
    
    def handle_zwj_sequences(self, text: str) -> str:
        """
        Handle ZWJ sequences (simplify complex emojis).
        
        Args:
            text: Text with potential ZWJ sequences
            
        Returns:
            Text with ZWJ sequences handled
        """
        # Simply return text with ZWJ removed for basic handling
        return text.replace(self.ZWJ, '')
    
    # ==================== Feature Extraction Methods ====================
    
    def extract_features(self, text: str) -> Dict[str, Any]:
        """
        Extract emoji-based features for ML.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of features
        """
        emojis = self.extract_emojis(text)
        freq = self.get_emoji_frequency(text)
        sentiment_data = self.get_text_emoji_sentiment(text)
        
        # Count by severity
        severity_counts = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0, 'positive': 0}
        for emoji in emojis:
            severity = self._severity_lookup.get(emoji, 'none')
            if severity in severity_counts:
                severity_counts[severity] += 1
        
        # Count by category
        category_counts = {
            'offensive': sum(1 for e in emojis if self.is_offensive_emoji(e)),
            'aggressive': sum(1 for e in emojis if self.is_aggressive_emoji(e)),
            'threatening': sum(1 for e in emojis if e in self.THREATENING_EMOJIS),
            'mocking': sum(1 for e in emojis if self.is_mocking_emoji(e)),
            'body_shaming': sum(1 for e in emojis if self.is_body_shaming_emoji(e)),
            'sexual': sum(1 for e in emojis if self.is_sexual_harassment_emoji(e)),
        }
        
        return {
            'emoji_count': len(emojis),
            'unique_emoji_count': len(set(emojis)),
            'emoji_density': len(emojis) / max(len(text.split()), 1),
            'average_sentiment': sentiment_data['average_sentiment'],
            'has_offensive': any(self.is_offensive_emoji(e) for e in emojis),
            'has_aggressive': any(self.is_aggressive_emoji(e) for e in emojis),
            'has_threatening': any(e in self.THREATENING_EMOJIS for e in emojis),
            'has_mocking': any(self.is_mocking_emoji(e) for e in emojis),
            'has_body_shaming': any(self.is_body_shaming_emoji(e) for e in emojis),
            'has_sexual': any(self.is_sexual_harassment_emoji(e) for e in emojis),
            'has_middle_finger': any(e in self.OFFENSIVE_EMOJIS for e in emojis),
            'has_bad_emoji': any(self.is_bad_emoji(e) for e in emojis),
            'is_spam': self.detect_emoji_spam(text),
            'severity_counts': severity_counts,
            'category_counts': category_counts,
            'bullying_pattern': self.detect_bullying_pattern(text)
        }
    
    # ==================== Batch Processing Methods ====================
    
    def process_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Process multiple texts in batch.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of analysis results
        """
        return [self.analyze(text) for text in texts]
    
    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Complete emoji analysis of text.
        
        Args:
            text: Input text
            
        Returns:
            Comprehensive analysis dictionary
        """
        emojis = self.extract_emojis(text)
        
        return {
            'text': text,
            'emojis': emojis,
            'count': len(emojis),
            'unique_count': len(set(emojis)),
            'frequency': self.get_emoji_frequency(text),
            'sentiment': self.get_text_emoji_sentiment(text),
            'bullying_analysis': self.detect_bullying_pattern(text),
            'threat_analysis': self.detect_threatening_pattern(text),
            'mockery_analysis': self.detect_mocking_pattern(text),
            'sarcasm_analysis': self.detect_sarcasm(text),
            'offensive_analysis': self.detect_offensive_emojis(text),
            'sexual_harassment_analysis': self.detect_sexual_harassment_emojis(text),
            'features': self.extract_features(text)
        }


# Convenience function for quick emoji extraction
def extract_emojis(text: str) -> List[str]:
    """Quick emoji extraction without creating handler instance."""
    handler = EmojiHandler()
    return handler.extract_emojis(text)


if __name__ == "__main__":
    # Quick test
    handler = EmojiHandler()
    
    test_texts = [
        "Hello 😀 how are you?",
        "You're so dumb 🤡🤡🤡",
        "nee tumba ugly 🤮 sakkat waste 😡",
        "Great job failing 👏👏👏",
        "That joke 💀💀💀 I'm dead",
        "🖕🖕 get lost 🖕",
        "You're such a 🐷🐷🐷",
        "Hey sexy 🍆💦🍑",
    ]
    
    for text in test_texts:
        print(f"\nText: {text}")
        result = handler.analyze(text)
        print(f"  Emojis: {result['emojis']}")
        print(f"  Sentiment: {result['sentiment']['overall']}")
        print(f"  Bullying: {result['bullying_analysis']['has_pattern']}")
        if result['offensive_analysis']['has_offensive']:
            print(f"  Offensive: {result['offensive_analysis']['detail']}")
        if result['sexual_harassment_analysis']['has_sexual_content']:
            print(f"  Sexual: {result['sexual_harassment_analysis']['detail']}")