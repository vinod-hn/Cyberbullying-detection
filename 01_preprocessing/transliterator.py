# Transliterator
"""
Transliterator: Bi-directional transliteration between Kannada script and Roman (Latin) script.
Handles Kannada-English code-mixed text for cyberbullying detection.
Supports phonetic mapping, conjunct consonants, and variable spellings.
"""

import re
import os
import json
import unicodedata
from typing import List, Dict, Optional, Tuple, Any, Union, Set


class Transliterator:
    """
    Bi-directional Kannada-Roman transliterator for cyberbullying detection.
    
    Features:
    - Kannada script to Roman (Latin) transliteration
    - Roman to Kannada script transliteration
    - Code-mixed text handling
    - Phonetic mapping support
    - Conjunct consonant handling
    - Variable spelling support
    - Auto-detection of script
    
    Attributes:
        kannada_to_roman_map: Mapping for Kannada to Roman conversion
        roman_to_kannada_map: Mapping for Roman to Kannada conversion
        config: Configuration settings
    """
    
    # Kannada Unicode range
    KANNADA_RANGE = (0x0C80, 0x0CFF)
    
    # Kannada vowels (Swaras) with Roman equivalents
    KANNADA_VOWELS = {
        'ಅ': 'a', 'ಆ': 'aa', 'ಇ': 'i', 'ಈ': 'ii',
        'ಉ': 'u', 'ಊ': 'uu', 'ಋ': 'ru', 'ೠ': 'ruu',
        'ಎ': 'e', 'ಏ': 'ee', 'ಐ': 'ai',
        'ಒ': 'o', 'ಓ': 'oo', 'ಔ': 'au',
        'ಅಂ': 'am', 'ಅಃ': 'ah',
    }
    
    # Kannada vowel signs (Matras) with Roman equivalents
    KANNADA_VOWEL_SIGNS = {
        'ಾ': 'aa', 'ಿ': 'i', 'ೀ': 'ii',
        'ು': 'u', 'ೂ': 'uu', 'ೃ': 'ru',
        'ೆ': 'e', 'ೇ': 'ee', 'ೈ': 'ai',
        'ೊ': 'o', 'ೋ': 'oo', 'ೌ': 'au',
        'ಂ': 'm', 'ಃ': 'h',
        '್': '',  # Virama (halant) - removes inherent vowel
    }
    
    # Kannada consonants (Vyanjanas) with Roman equivalents
    KANNADA_CONSONANTS = {
        # Velars (ka-varga)
        'ಕ': 'ka', 'ಖ': 'kha', 'ಗ': 'ga', 'ಘ': 'gha', 'ಙ': 'nga',
        # Palatals (cha-varga)
        'ಚ': 'cha', 'ಛ': 'chha', 'ಜ': 'ja', 'ಝ': 'jha', 'ಞ': 'nya',
        # Retroflexes (Ta-varga)
        'ಟ': 'ta', 'ಠ': 'tha', 'ಡ': 'da', 'ಢ': 'dha', 'ಣ': 'na',
        # Dentals (ta-varga)
        'ತ': 'ta', 'ಥ': 'tha', 'ದ': 'da', 'ಧ': 'dha', 'ನ': 'na',
        # Labials (pa-varga)
        'ಪ': 'pa', 'ಫ': 'pha', 'ಬ': 'ba', 'ಭ': 'bha', 'ಮ': 'ma',
        # Semi-vowels
        'ಯ': 'ya', 'ರ': 'ra', 'ಲ': 'la', 'ವ': 'va',
        # Sibilants
        'ಶ': 'sha', 'ಷ': 'sha', 'ಸ': 'sa',
        # Aspirate
        'ಹ': 'ha',
        # Additional
        'ಳ': 'la', 'ೞ': 'zha', 'ಱ': 'rra',
        # Nukta consonants (for borrowed sounds)
        'ಕ಼': 'qa', 'ಖ಼': 'kha', 'ಗ಼': 'gha', 'ಜ಼': 'za',
        'ಫ಼': 'fa',
    }
    
    # Kannada numerals
    KANNADA_NUMERALS = {
        '೦': '0', '೧': '1', '೨': '2', '೩': '3', '೪': '4',
        '೫': '5', '೬': '6', '೭': '7', '೮': '8', '೯': '9',
    }
    
    # Roman to Kannada vowels mapping (reverse)
    ROMAN_TO_KANNADA_VOWELS = {
        'a': 'ಅ', 'aa': 'ಆ', 'A': 'ಆ',
        'i': 'ಇ', 'ii': 'ಈ', 'ee': 'ಈ',
        'u': 'ಉ', 'uu': 'ಊ', 'oo': 'ಊ',
        'e': 'ಎ', 'ae': 'ಏ',
        'ai': 'ಐ',
        'o': 'ಒ', 'O': 'ಓ',
        'au': 'ಔ', 'ou': 'ಔ',
    }
    
    # Roman to Kannada consonants mapping
    ROMAN_TO_KANNADA_CONSONANTS = {
        # Basic consonants
        'k': 'ಕ', 'kh': 'ಖ', 'g': 'ಗ', 'gh': 'ಘ', 'ng': 'ಙ',
        'c': 'ಚ', 'ch': 'ಚ', 'chh': 'ಛ', 'j': 'ಜ', 'jh': 'ಝ', 'ny': 'ಞ',
        't': 'ತ', 'th': 'ಥ', 'd': 'ದ', 'dh': 'ಧ', 'n': 'ನ',
        'p': 'ಪ', 'ph': 'ಫ', 'f': 'ಫ', 'b': 'ಬ', 'bh': 'ಭ', 'm': 'ಮ',
        'y': 'ಯ', 'r': 'ರ', 'l': 'ಲ', 'v': 'ವ', 'w': 'ವ',
        'sh': 'ಶ', 's': 'ಸ', 'h': 'ಹ',
        'L': 'ಳ', 'zh': 'ೞ',
        # Nukta sounds
        'q': 'ಕ಼', 'z': 'ಜ಼',
    }
    
    # Roman to Kannada vowel signs (for syllable construction)
    ROMAN_TO_KANNADA_VOWEL_SIGNS = {
        'a': '', 'aa': 'ಾ', 'A': 'ಾ',
        'i': 'ಿ', 'ii': 'ೀ', 'ee': 'ೀ',
        'u': 'ು', 'uu': 'ೂ', 'oo': 'ೂ',
        'e': 'ೆ', 'ae': 'ೇ',
        'ai': 'ೈ',
        'o': 'ೊ', 'O': 'ೋ',
        'au': 'ೌ', 'ou': 'ೌ',
    }
    
    # Common Romanized Kannada words with standard Kannada spelling
    COMMON_WORDS_ROMAN_TO_KANNADA = {
        # Pronouns
        'nee': 'ನೀ', 'neenu': 'ನೀನು', 'naanu': 'ನಾನು', 'naan': 'ನಾನ್',
        'avnu': 'ಅವನು', 'avlu': 'ಅವಳು', 'avru': 'ಅವರು',
        'ninna': 'ನಿನ್ನ', 'nanna': 'ನನ್ನ', 'nimma': 'ನಿಮ್ಮ', 'namma': 'ನಮ್ಮ',
        'nin': 'ನಿನ್', 'nan': 'ನನ್',
        'ivan': 'ಇವನ', 'ivalu': 'ಇವಳು', 'ivru': 'ಇವರು',
        
        # Address terms
        'machaa': 'ಮಚ್ಚಾ', 'macha': 'ಮಚ್ಚ', 'maccha': 'ಮಚ್ಚ',
        'maga': 'ಮಗ', 'guru': 'ಗುರು', 'yaar': 'ಯಾರ್',
        'anna': 'ಅಣ್ಣ', 'akka': 'ಅಕ್ಕ', 'appa': 'ಅಪ್ಪ', 'amma': 'ಅಮ್ಮ',
        're': 'ರೇ', 'boss': 'ಬಾಸ್', 'bossu': 'ಬಾಸು', 'saar': 'ಸಾರ್',
        
        # Intensifiers
        'tumba': 'ತುಂಬಾ', 'thumba': 'ತುಂಬಾ',
        'sakkat': 'ಸಕ್ಕತ್', 'sakkath': 'ಸಕ್ಕತ್',
        'full': 'ಫುಲ್', 'swalpa': 'ಸ್ವಲ್ಪ',
        'jaasti': 'ಜಾಸ್ತಿ', 'chennag': 'ಚೆನ್ನಾಗ್', 'chennaag': 'ಚೆನ್ನಾಗ್',
        'chennagi': 'ಚೆನ್ನಾಗಿ',
        
        # Common verbs
        'maadu': 'ಮಾಡು', 'hogi': 'ಹೋಗಿ', 'baa': 'ಬಾ',
        'helu': 'ಹೇಳು', 'nodu': 'ನೋಡು', 'kelu': 'ಕೇಳು',
        'haaki': 'ಹಾಕಿ', 'tagond': 'ತಗೊಂಡ',
        'madthiya': 'ಮಾಡ್ತೀಯ', 'bartiya': 'ಬರ್ತೀಯ',
        'hogthiya': 'ಹೋಗ್ತೀಯ', 'madtiya': 'ಮಾಡ್ತೀಯ',
        'bartira': 'ಬರ್ತೀರಾ', 'barthini': 'ಬರ್ತೀನಿ',
        'hogthini': 'ಹೋಗ್ತೀನಿ', 'madthini': 'ಮಾಡ್ತೀನಿ',
        'aaytu': 'ಆಯ್ತು', 'agide': 'ಆಗಿದೆ',
        'hodbitta': 'ಹೋಡ್ಬಿಟ್ಟ', 'odi': 'ಓಡಿ',
        
        # Negation
        'illa': 'ಇಲ್ಲ', 'beda': 'ಬೇಡ', 'beka': 'ಬೇಕಾ',
        'gotilla': 'ಗೊತ್ತಿಲ್ಲ', 'gottilla': 'ಗೊತ್ತಿಲ್ಲ',
        'madbeda': 'ಮಾಡ್ಬೇಡ', 'hogbeda': 'ಹೋಗ್ಬೇಡ',
        'illade': 'ಇಲ್ಲದೆ', 'illaadre': 'ಇಲ್ಲಾದ್ರೆ',
        
        # Question words
        'yaake': 'ಯಾಕೆ', 'yelli': 'ಎಲ್ಲಿ', 'elli': 'ಎಲ್ಲಿ',
        'yavaga': 'ಯಾವಾಗ', 'yavag': 'ಯಾವಾಗ್',
        'hege': 'ಹೇಗೆ', 'yaavdu': 'ಯಾವ್ದು',
        'yenu': 'ಯೇನು', 'enu': 'ಏನು',
        'eshtu': 'ಎಷ್ಟು', 'yaaru': 'ಯಾರು',
        'hegidiya': 'ಹೇಗಿದ್ದೀಯ', 'hegiddiya': 'ಹೇಗಿದ್ದೀಯ',
        
        # Common words
        'ide': 'ಇದೆ', 'gottu': 'ಗೊತ್ತು', 'beku': 'ಬೇಕು',
        'andre': 'ಅಂದ್ರೆ', 'ashte': 'ಅಷ್ಟೇ',
        'kelsa': 'ಕೆಲ್ಸ', 'oota': 'ಊಟ', 'sari': 'ಸರಿ',
        'sumne': 'ಸುಮ್ನೆ', 'hogidaroo': 'ಹೋಗಿದ್ರೂ',
        'anta': 'ಅಂತ', 'ansta': 'ಅನ್ಸ್ತ',
        'tara': 'ತರ', 'kanstide': 'ಕಾಣಸ್ತಿದೆ',
        'houdu': 'ಹೌದು', 'alva': 'ಅಲ್ವಾ', 'aadre': 'ಆದ್ರೆ',
        
        # Time expressions
        'ivattu': 'ಇವತ್ತು', 'naale': 'ನಾಳೆ',
        'prati': 'ಪ್ರತಿ', 'yavaglu': 'ಯಾವಾಗ್ಲೂ',
        'dina': 'ದಿನ', 'nimisha': 'ನಿಮಿಷ',
        'gante': 'ಗಂಟೆ',
        
        # Postpositions
        'nalli': 'ನಲ್ಲಿ', 'ge': 'ಗೆ', 'inda': 'ಇಂದ',
        'jote': 'ಜೊತೆ', 'hatra': 'ಹತ್ರ',
        'mele': 'ಮೇಲೆ', 'kelage': 'ಕೆಳಗೆ',
        
        # Expressions
        'ayyo': 'ಅಯ್ಯೋ', 'ayyoo': 'ಅಯ್ಯೋ',
        'yappa': 'ಯಪ್ಪಾ', 'arey': 'ಅರೆ', 'abe': 'ಅಬೆ',
        
        # Insult words (for detection)
        'thotha': 'ತೋತಾ', 'thota': 'ತೋತಾ',
        'singri': 'ಸಿಂಗ್ರಿ', 'dagarina': 'ದಗರಿನ',
        'hucchadana': 'ಹುಚ್ಚದನ', 'hucchadanta': 'ಹುಚ್ಚದಂತ',
        'moorkha': 'ಮೂರ್ಖ', 'bewakoof': 'ಬೇವಕೂಫ್',
        'irritating': 'ಇರಿಟೇಟಿಂಗ್', 'agthiya': 'ಆಗ್ತಿಯ',
        'idiya': 'ಇದ್ದೀಯ',
        
        # Academic terms
        'exam': 'ಎಕ್ಸಾಮ್', 'lab': 'ಲ್ಯಾಬ್',
        'project': 'ಪ್ರಾಜೆಕ್ಟ್', 'class': 'ಕ್ಲಾಸ್',
        'assignment': 'ಅಸೈನ್‌ಮೆಂಟ್', 'presentation': 'ಪ್ರೆಸೆಂಟೇಶನ್',
        'submit': 'ಸಬ್ಮಿಟ್', 'group': 'ಗ್ರೂಪ್',
    }
    
    # Common Kannada words to Roman mapping
    COMMON_WORDS_KANNADA_TO_ROMAN = {v: k for k, v in COMMON_WORDS_ROMAN_TO_KANNADA.items()}
    
    # Common English words to preserve
    ENGLISH_WORDS = {
        'the', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has',
        'do', 'does', 'did', 'will', 'would', 'could', 'should', 'can',
        'a', 'an', 'and', 'but', 'or', 'not', 'no', 'yes',
        'you', 'your', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her',
        'my', 'his', 'its', 'our', 'their', 'this', 'that',
        'what', 'which', 'who', 'where', 'when', 'why', 'how',
        'i', 'am', 'to', 'of', 'in', 'on', 'at', 'by', 'for', 'with',
        'exam', 'lab', 'project', 'class', 'assignment', 'presentation',
        'stupid', 'dumb', 'idiot', 'fool', 'waste', 'useless', 'irritating',
        'fellow', 'dude', 'bro', 'guy', 'girl', 'friend',
        'super', 'nice', 'good', 'bad', 'late', 'time', 'full',
        'online', 'offline', 'group', 'chat', 'message', 'status',
        'complaint', 'remove', 'somehow', 'everyone', 'nobody',
        'tough', 'behaviour', 'behavior', 'attitude', 'problem',
    }
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        custom_mappings_path: Optional[str] = None
    ):
        """
        Initialize Transliterator.
        
        Args:
            config: Optional configuration dictionary.
            custom_mappings_path: Path to custom mappings file.
        """
        self.config = config or self._default_config()
        
        # Build complete mappings
        self._build_mappings()
        
        # Load custom mappings if provided
        if custom_mappings_path and os.path.exists(custom_mappings_path):
            self._load_custom_mappings(custom_mappings_path)
        
        # Compile patterns
        self._compile_patterns()
    
    def _default_config(self) -> Dict[str, Any]:
        """Return default configuration."""
        return {
            'preserve_english': True,
            'preserve_numbers': True,
            'preserve_punctuation': True,
            'preserve_emojis': True,
            'handle_conjuncts': True,
            'normalize_unicode': True,
            'lowercase_output': False,
            'word_level': True,  # Transliterate word by word
        }
    
    def _build_mappings(self) -> None:
        """Build complete transliteration mappings."""
        # Kannada to Roman complete mapping
        self.kannada_to_roman_map = {}
        self.kannada_to_roman_map.update(self.KANNADA_VOWELS)
        self.kannada_to_roman_map.update(self.KANNADA_VOWEL_SIGNS)
        self.kannada_to_roman_map.update(self.KANNADA_CONSONANTS)
        self.kannada_to_roman_map.update(self.KANNADA_NUMERALS)
        self.kannada_to_roman_map.update(self.COMMON_WORDS_KANNADA_TO_ROMAN)
        
        # Roman to Kannada complete mapping
        self.roman_to_kannada_map = {}
        self.roman_to_kannada_map.update(self.ROMAN_TO_KANNADA_VOWELS)
        self.roman_to_kannada_map.update(self.ROMAN_TO_KANNADA_CONSONANTS)
        self.roman_to_kannada_map.update(self.COMMON_WORDS_ROMAN_TO_KANNADA)
        
        # Build reverse numeral mapping
        self.roman_to_kannada_numerals = {v: k for k, v in self.KANNADA_NUMERALS.items()}
    
    def _load_custom_mappings(self, path: str) -> None:
        """Load custom mappings from file."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                custom = json.load(f)
            
            if 'kannada_to_roman' in custom:
                self.kannada_to_roman_map.update(custom['kannada_to_roman'])
            if 'roman_to_kannada' in custom:
                self.roman_to_kannada_map.update(custom['roman_to_kannada'])
        except Exception:
            pass
    
    def _compile_patterns(self) -> None:
        """Compile regex patterns for efficiency."""
        # Kannada script pattern
        self.kannada_pattern = re.compile(r'[\u0C80-\u0CFF]+')
        
        # Word pattern (handles mixed scripts)
        self.word_pattern = re.compile(r"[\w']+|[^\w\s]", re.UNICODE)
        
        # English word pattern
        self.english_word_pattern = re.compile(r'^[a-zA-Z]+$')
        
        # Emoji pattern
        self.emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"
            "\U0001F300-\U0001F5FF"
            "\U0001F680-\U0001F6FF"
            "\U0001F900-\U0001F9FF"
            "\U00002600-\U000026FF"
            "\U00002700-\U000027BF"
            "]+",
            flags=re.UNICODE
        )
        
        # Number pattern
        self.number_pattern = re.compile(r'\d+')
        
        # Hashtag ID pattern (dataset format)
        self.hashtag_pattern = re.compile(r'#[a-f0-9]{4}\b')
        
        # Build Roman syllable pattern (sorted by length for greedy matching)
        roman_keys = sorted(self.roman_to_kannada_map.keys(), key=len, reverse=True)
        escaped = [re.escape(k) for k in roman_keys if len(k) > 1]
        if escaped:
            self.roman_syllable_pattern = re.compile('|'.join(escaped), re.IGNORECASE)
        else:
            self.roman_syllable_pattern = None
    
    def is_kannada_script(self, text: str) -> bool:
        """
        Check if text contains Kannada script characters.
        
        Args:
            text: Input text to check
            
        Returns:
            True if text contains Kannada script
        """
        if not text:
            return False
        
        for char in text:
            code = ord(char)
            if self.KANNADA_RANGE[0] <= code <= self.KANNADA_RANGE[1]:
                return True
        
        return False
    
    def is_roman_script(self, text: str) -> bool:
        """
        Check if text is primarily Roman (Latin) script.
        
        Args:
            text: Input text to check
            
        Returns:
            True if text is primarily Roman script
        """
        if not text:
            return False
        
        # Check if mostly ASCII letters
        letters = [c for c in text if c.isalpha()]
        if not letters:
            return False
        
        ascii_count = sum(1 for c in letters if c.isascii())
        return ascii_count / len(letters) > 0.8
    
    def get_script_ratio(self, text: str) -> Dict[str, float]:
        """
        Get ratio of different scripts in text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with script ratios
        """
        if not text:
            return {'kannada': 0.0, 'roman': 0.0, 'other': 0.0}
        
        kannada_count = 0
        roman_count = 0
        other_count = 0
        total = 0
        
        for char in text:
            if char.isspace():
                continue
            
            total += 1
            code = ord(char)
            
            if self.KANNADA_RANGE[0] <= code <= self.KANNADA_RANGE[1]:
                kannada_count += 1
            elif char.isascii() and char.isalpha():
                roman_count += 1
            else:
                other_count += 1
        
        if total == 0:
            return {'kannada': 0.0, 'roman': 0.0, 'other': 0.0}
        
        return {
            'kannada': kannada_count / total,
            'roman': roman_count / total,
            'other': other_count / total
        }
    
    def kannada_to_roman(self, text: str) -> str:
        """
        Transliterate Kannada script to Roman (Latin) script.
        
        Args:
            text: Kannada text to transliterate
            
        Returns:
            Romanized text
        """
        if not text:
            return ""
        
        if self.config.get('normalize_unicode', True):
            text = unicodedata.normalize('NFC', text)
        
        result = []
        i = 0
        
        while i < len(text):
            char = text[i]
            
            # Check if character is Kannada
            code = ord(char)
            if self.KANNADA_RANGE[0] <= code <= self.KANNADA_RANGE[1]:
                # Try to match multi-char sequences first (words)
                matched = False
                
                # Try common words first (longest match)
                for length in range(min(15, len(text) - i), 0, -1):
                    substr = text[i:i + length]
                    if substr in self.COMMON_WORDS_KANNADA_TO_ROMAN:
                        result.append(self.COMMON_WORDS_KANNADA_TO_ROMAN[substr])
                        i += length
                        matched = True
                        break
                
                if not matched:
                    # Handle individual characters
                    if char in self.kannada_to_roman_map:
                        roman = self.kannada_to_roman_map[char]
                        result.append(roman)
                    else:
                        # Try to handle consonant + vowel combinations
                        if i + 1 < len(text):
                            next_char = text[i + 1]
                            if next_char in self.KANNADA_VOWEL_SIGNS:
                                # Get consonant base
                                if char in self.KANNADA_CONSONANTS:
                                    base = self.KANNADA_CONSONANTS[char]
                                    # Remove inherent 'a' and add vowel sign
                                    if base.endswith('a'):
                                        base = base[:-1]
                                    vowel = self.KANNADA_VOWEL_SIGNS.get(next_char, '')
                                    result.append(base + vowel)
                                    i += 2
                                    continue
                        
                        # Just add the character mapping or the character itself
                        if char in self.KANNADA_CONSONANTS:
                            result.append(self.KANNADA_CONSONANTS[char])
                        elif char in self.KANNADA_VOWELS:
                            result.append(self.KANNADA_VOWELS[char])
                        else:
                            result.append(char)
                    i += 1
            else:
                # Non-Kannada character - preserve it
                result.append(char)
                i += 1
        
        output = ''.join(result)
        
        if self.config.get('lowercase_output', False):
            output = output.lower()
        
        return output
    
    def roman_to_kannada(self, text: str) -> str:
        """
        Transliterate Roman (Latin) script to Kannada script.
        
        Args:
            text: Romanized text to transliterate
            
        Returns:
            Kannada script text
        """
        if not text:
            return ""
        
        # If already contains Kannada, return as-is
        if self.is_kannada_script(text) and not self.is_roman_script(text):
            return text
        
        if self.config.get('word_level', True):
            return self._transliterate_word_level(text, to_kannada=True)
        else:
            return self._transliterate_char_level(text, to_kannada=True)
    
    def _transliterate_word_level(self, text: str, to_kannada: bool = True) -> str:
        """Transliterate text word by word."""
        tokens = self.word_pattern.findall(text)
        result = []
        
        for token in tokens:
            transliterated = self._transliterate_token(token, to_kannada)
            result.append(transliterated)
        
        # Reconstruct with proper spacing
        output = []
        i = 0
        for char in text:
            if char.isspace():
                output.append(char)
            elif i < len(result):
                if not output or output[-1].isspace() or not result[i]:
                    output.append(result[i])
                else:
                    # Check if we need to add this token
                    if len(output) == 0 or output[-1] != result[i]:
                        output.append(result[i])
                i += 1
        
        # Simpler reconstruction
        return self._reconstruct_text(text, tokens, result)
    
    def _reconstruct_text(
        self,
        original: str,
        tokens: List[str],
        transliterated: List[str]
    ) -> str:
        """Reconstruct text with transliterated tokens preserving spacing."""
        result = original
        
        for orig, trans in zip(tokens, transliterated):
            if orig != trans:
                # Replace first occurrence
                result = result.replace(orig, trans, 1)
        
        return result
    
    def _transliterate_token(self, token: str, to_kannada: bool = True) -> str:
        """Transliterate a single token."""
        if not token:
            return token
        
        # Preserve punctuation
        if len(token) == 1 and not token.isalnum():
            return token
        
        # Preserve numbers
        if self.config.get('preserve_numbers', True) and token.isdigit():
            return token
        
        # Preserve emojis
        if self.config.get('preserve_emojis', True) and self.emoji_pattern.match(token):
            return token
        
        # Preserve hashtags
        if token.startswith('#'):
            return token
        
        token_lower = token.lower()
        
        if to_kannada:
            # Check if English word to preserve
            if self.config.get('preserve_english', True):
                if token_lower in self.ENGLISH_WORDS:
                    return token
            
            # Check common words mapping first
            if token_lower in self.COMMON_WORDS_ROMAN_TO_KANNADA:
                return self.COMMON_WORDS_ROMAN_TO_KANNADA[token_lower]
            
            # Transliterate character by character
            return self._roman_word_to_kannada(token_lower)
        else:
            # Kannada to Roman
            if token in self.COMMON_WORDS_KANNADA_TO_ROMAN:
                return self.COMMON_WORDS_KANNADA_TO_ROMAN[token]
            
            return self._kannada_word_to_roman(token)
    
    def _roman_word_to_kannada(self, word: str) -> str:
        """Convert a Roman word to Kannada script."""
        if not word:
            return word
        
        result = []
        i = 0
        
        while i < len(word):
            matched = False
            
            # Try longer matches first (3, 2, then 1 character)
            for length in [3, 2, 1]:
                if i + length <= len(word):
                    substr = word[i:i + length]
                    
                    # Check consonant mappings
                    if substr in self.ROMAN_TO_KANNADA_CONSONANTS:
                        consonant = self.ROMAN_TO_KANNADA_CONSONANTS[substr]
                        
                        # Check for following vowel
                        remaining = word[i + length:]
                        vowel_found = False
                        
                        for v_len in [2, 1]:
                            if len(remaining) >= v_len:
                                v_substr = remaining[:v_len]
                                if v_substr in self.ROMAN_TO_KANNADA_VOWEL_SIGNS:
                                    vowel_sign = self.ROMAN_TO_KANNADA_VOWEL_SIGNS[v_substr]
                                    result.append(consonant + vowel_sign)
                                    i += length + v_len
                                    vowel_found = True
                                    matched = True
                                    break
                        
                        if not vowel_found:
                            # Add consonant with inherent 'a'
                            result.append(consonant)
                            i += length
                            matched = True
                        break
                    
                    # Check vowel mappings (at word start)
                    if substr in self.ROMAN_TO_KANNADA_VOWELS:
                        result.append(self.ROMAN_TO_KANNADA_VOWELS[substr])
                        i += length
                        matched = True
                        break
            
            if not matched:
                # Keep character as-is
                result.append(word[i])
                i += 1
        
        return ''.join(result)
    
    def _kannada_word_to_roman(self, word: str) -> str:
        """Convert a Kannada word to Roman script."""
        return self.kannada_to_roman(word)
    
    def _transliterate_char_level(self, text: str, to_kannada: bool = True) -> str:
        """Transliterate text character by character."""
        if to_kannada:
            return self._roman_word_to_kannada(text.lower())
        else:
            return self.kannada_to_roman(text)
    
    def auto_transliterate(self, text: str) -> str:
        """
        Automatically detect script and transliterate to the other.
        
        Args:
            text: Input text
            
        Returns:
            Transliterated text
        """
        if not text:
            return ""
        
        ratios = self.get_script_ratio(text)
        
        if ratios['kannada'] > ratios['roman']:
            # Primarily Kannada - convert to Roman
            return self.kannada_to_roman(text)
        else:
            # Primarily Roman - convert to Kannada
            return self.roman_to_kannada(text)
    
    def transliterate(self, text: str, to_script: str = 'kannada') -> str:
        """
        Transliterate text to specified script.
        
        Args:
            text: Input text
            to_script: Target script ('kannada' or 'roman')
            
        Returns:
            Transliterated text
        """
        if to_script.lower() in ['kannada', 'kn', 'kan']:
            return self.roman_to_kannada(text)
        elif to_script.lower() in ['roman', 'en', 'latin', 'english']:
            return self.kannada_to_roman(text)
        else:
            return text
    
    def transliterate_preserve_english(self, text: str) -> str:
        """
        Transliterate text while preserving English words.
        
        Args:
            text: Input text (Roman with Kannada words)
            
        Returns:
            Text with Kannada words in Kannada script, English preserved
        """
        # Temporarily enable preserve_english
        original_config = self.config.get('preserve_english', True)
        self.config['preserve_english'] = True
        
        result = self.roman_to_kannada(text)
        
        self.config['preserve_english'] = original_config
        return result
    
    def batch_transliterate(
        self,
        texts: List[str],
        to_script: str = 'kannada'
    ) -> List[str]:
        """
        Transliterate multiple texts.
        
        Args:
            texts: List of input texts
            to_script: Target script
            
        Returns:
            List of transliterated texts
        """
        return [self.transliterate(text, to_script) for text in texts]
    
    def add_custom_mapping(self, roman: str, kannada: str) -> None:
        """
        Add a custom transliteration mapping.
        
        Args:
            roman: Roman representation
            kannada: Kannada representation
        """
        self.roman_to_kannada_map[roman.lower()] = kannada
        self.kannada_to_roman_map[kannada] = roman.lower()
        self.COMMON_WORDS_ROMAN_TO_KANNADA[roman.lower()] = kannada
        self.COMMON_WORDS_KANNADA_TO_ROMAN[kannada] = roman.lower()
    
    def remove_custom_mapping(self, roman: str) -> bool:
        """
        Remove a custom mapping.
        
        Args:
            roman: Roman representation to remove
            
        Returns:
            True if mapping was removed
        """
        roman_lower = roman.lower()
        removed = False
        
        if roman_lower in self.COMMON_WORDS_ROMAN_TO_KANNADA:
            kannada = self.COMMON_WORDS_ROMAN_TO_KANNADA[roman_lower]
            del self.COMMON_WORDS_ROMAN_TO_KANNADA[roman_lower]
            self.COMMON_WORDS_KANNADA_TO_ROMAN.pop(kannada, None)
            removed = True
        
        return removed
    
    def get_mapping(self) -> Dict[str, Dict[str, str]]:
        """
        Get current transliteration mappings.
        
        Returns:
            Dictionary with mappings
        """
        return {
            'kannada_to_roman': dict(self.kannada_to_roman_map),
            'roman_to_kannada': dict(self.roman_to_kannada_map),
            'common_words': dict(self.COMMON_WORDS_ROMAN_TO_KANNADA)
        }
    
    def get_stats(self) -> Dict[str, int]:
        """Get statistics about loaded mappings."""
        return {
            'kannada_to_roman_count': len(self.kannada_to_roman_map),
            'roman_to_kannada_count': len(self.roman_to_kannada_map),
            'common_words_count': len(self.COMMON_WORDS_ROMAN_TO_KANNADA),
            'vowels_count': len(self.KANNADA_VOWELS),
            'consonants_count': len(self.KANNADA_CONSONANTS),
        }
    
    def process(self, text: str) -> Dict[str, Any]:
        """
        Process text with full transliteration analysis.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with processing results
        """
        if not text:
            return {
                'original': '',
                'to_kannada': '',
                'to_roman': '',
                'script_ratio': {'kannada': 0, 'roman': 0, 'other': 0},
                'detected_script': 'unknown'
            }
        
        ratios = self.get_script_ratio(text)
        
        if ratios['kannada'] > ratios['roman']:
            detected = 'kannada'
        elif ratios['roman'] > ratios['kannada']:
            detected = 'roman'
        else:
            detected = 'mixed'
        
        return {
            'original': text,
            'to_kannada': self.roman_to_kannada(text),
            'to_roman': self.kannada_to_roman(text),
            'script_ratio': ratios,
            'detected_script': detected
        }
    
    def __repr__(self) -> str:
        """String representation."""
        stats = self.get_stats()
        return (
            f"Transliterator("
            f"common_words={stats['common_words_count']}, "
            f"vowels={stats['vowels_count']}, "
            f"consonants={stats['consonants_count']})"
        )


# Convenience functions
def kannada_to_roman(text: str) -> str:
    """Quick Kannada to Roman transliteration."""
    return Transliterator().kannada_to_roman(text)


def roman_to_kannada(text: str) -> str:
    """Quick Roman to Kannada transliteration."""
    return Transliterator().roman_to_kannada(text)


if __name__ == "__main__":
    # Quick test
    trans = Transliterator()
    
    test_texts = [
        ("neenu hegiddiya machaa", "roman_to_kannada"),
        ("ನೀನು ಹೇಗಿದ್ದೀಯ", "kannada_to_roman"),
        ("nee tumba irritating agthiya yaar", "roman_to_kannada"),
        ("exam tumba tough aaytu", "roman_to_kannada"),
        ("naale 9 gante lab ge bartira", "roman_to_kannada"),
    ]
    
    print(f"Transliterator Stats: {trans.get_stats()}")
    print()
    
    for text, direction in test_texts:
        if direction == "roman_to_kannada":
            result = trans.roman_to_kannada(text)
            print(f"Roman: {text}")
            print(f"Kannada: {result}")
        else:
            result = trans.kannada_to_roman(text)
            print(f"Kannada: {text}")
            print(f"Roman: {result}")
        print()
