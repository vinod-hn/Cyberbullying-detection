# Code Mix Processor
"""
CodeMixProcessor: Comprehensive handling of Kannada-English code-mixed text.
Optimized for cyberbullying detection in social media content.

Features:
- Language detection (Kannada script, Romanized Kannada, English, Code-mixed)
- Code-mix ratio calculation
- Token-level language classification
- Text normalization for mixed scripts
- Pattern detection for cyberbullying contexts
- Profanity detection across languages
"""

import re
import os
import csv
import json
from typing import List, Dict, Optional, Tuple, Any, Union, Set
from collections import Counter


class CodeMixProcessor:
    """
    Processor for Kannada-English code-mixed text.
    
    Handles language detection, normalization, and analysis of code-mixed
    content commonly found in cyberbullying messages.
    
    Attributes:
        kannada_vocab: Set of known Kannada/Romanized Kannada words
        english_vocab: Set of common English words
        profanity_kannada: Set of Kannada profanity terms
        profanity_english: Set of English profanity terms
    """
    
    # Kannada Unicode range
    KANNADA_RANGE = (0x0C80, 0x0CFF)
    
    # Common Romanized Kannada words (from dataset analysis)
    DEFAULT_KANNADA_VOCAB = {
        # Pronouns
        'nee', 'neenu', 'naanu', 'naan', 'avnu', 'avlu', 'avru', 'navu',
        'ivan', 'ivalu', 'ivru', 'ninna', 'nanna', 'avna', 'avla', 'nimma',
        'namma', 'ivana', 'ivala', 'yaaru', 'yenu', 'enu',
        
        # Address terms
        'maga', 'machaa', 'macha', 'maccha', 'guru', 'yaar', 're', 'anna',
        'akka', 'appa', 'amma', 'boss', 'bossu', 'saar',
        
        # Intensifiers
        'tumba', 'thumba', 'sakkat', 'sakkath', 'swalpa', 'jaasti', 'full',
        'chennaag', 'chennag', 'bekku', 'thika',
        
        # Verbs & Verb endings
        'madthiya', 'bartiya', 'hogthiya', 'nodthiya', 'kelthiya', 'heltiya',
        'idiya', 'agthiya', 'maadidya', 'bartidya', 'hogidya', 'aaytu',
        'aagide', 'madkond', 'bartini', 'hogthini', 'madthini', 'maadu',
        'hogi', 'baa', 'helu', 'nodu', 'kelu', 'tagond', 'haaki', 'hodbitta',
        'odi', 'barthini', 'madtiya', 'bartira', 'hegidiya', 'hegiddiya',
        
        # Negation
        'illa', 'beda', 'agalla', 'baralla', 'gotilla', 'gottilla', 'bekagilla',
        'madbeda', 'hogbeda', 'helbeda', 'illaadre', 'illade',
        
        # Question words
        'yaake', 'yelli', 'yavaga', 'hege', 'yaavdu', 'yenu', 'eshtu', 'yavag',
        
        # Common words
        'ide', 'gottu', 'beku', 'andre', 'ashte', 'kelsa', 'oota', 'sari',
        'sumne', 'elli', 'hogidaroo', 'anta', 'ansta', 'tara', 'kanstide',
        'nodoke', 'togondiya', 'ivananna', 'irboda', 'besseru', 'houdu',
        
        # Postpositions
        'nalli', 'ge', 'inda', 'jote', 'hatra', 'mele', 'kelage', 'mundhe', 'hinde',
        
        # Particles & Expressions
        'alva', 'alla', 'aadre', 'nodappa', 'kelappa', 'ayyoo', 'ayyo', 'yappa',
        'arey', 'abe', 'eno', 'yeno', 'haa',
        
        # Time expressions
        'ivattu', 'naale', 'prati', 'yavaglu', 'yavattu',
        
        # Borrowed but Kannada-ized
        'feel', 'agutte', 'agthiya', 'ade', 'ee', 'aa',
    }
    
    # Common English words (high frequency)
    DEFAULT_ENGLISH_VOCAB = {
        # Common words
        'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has',
        'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may',
        'might', 'must', 'can', 'a', 'an', 'and', 'but', 'or', 'not', 'no', 'yes',
        'you', 'your', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us',
        'them', 'my', 'his', 'its', 'our', 'their', 'this', 'that', 'these', 'those',
        'what', 'which', 'who', 'whom', 'where', 'when', 'why', 'how',
        'i', 'am', 'to', 'of', 'in', 'on', 'at', 'by', 'for', 'with', 'about',
        'from', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
        
        # Adjectives commonly used in cyberbullying
        'stupid', 'dumb', 'idiot', 'fool', 'ugly', 'fat', 'loser', 'waste',
        'fake', 'toxic', 'annoying', 'irritating', 'boring', 'useless',
        'pathetic', 'disgusting', 'terrible', 'horrible', 'awful',
        'complete', 'total', 'absolute', 'such',
        
        # Verbs
        'go', 'come', 'see', 'look', 'talk', 'speak', 'say', 'tell', 'ask',
        'stop', 'start', 'leave', 'stay', 'get', 'take', 'give', 'make',
        'like', 'want', 'need', 'know', 'think', 'feel', 'understand',
        
        # Social media / Tech
        'message', 'reply', 'status', 'check', 'online', 'spam', 'block',
        'follow', 'unfollow', 'post', 'comment', 'share', 'group', 'chat',
        'ping', 'dm', 'text', 'call',
        
        # Common in mixed text
        'just', 'only', 'also', 'too', 'very', 'really', 'so', 'much',
        'more', 'less', 'enough', 'always', 'never', 'ever', 'again',
        'now', 'then', 'here', 'there', 'everywhere', 'nowhere',
        'everyone', 'nobody', 'someone', 'anyone',
        'time', 'day', 'today', 'tomorrow', 'yesterday',
        
        # Academic context
        'class', 'exam', 'project', 'assignment', 'lecture', 'lab',
        'presentation', 'notes', 'submit', 'deadline',
        
        # Borrowed in Kannada context but English
        'fellow', 'dude', 'bro', 'guy', 'girl', 'friend', 'behaviour',
        'behavior', 'attitude', 'problem', 'issue', 'reason',
    }
    
    # Default Kannada profanity (Romanized) - from bad_words.csv and profanity_kannada.csv
    DEFAULT_PROFANITY_KANNADA = {
        # Basic insults
        'thotha', 'thota', 'thotha maga', 'singri', 'singri maga',
        'dagarina', 'dagarina maga', 'hucchadana', 'hucchadanta', 'hucchadanthe',
        'moorkha', 'bewakoof', 'bekku',
        
        # Vulgar/Sexual terms
        'gaandu', 'gandu', 'soolya', 'sulya', 'bosodina', 'bosodike',
        'soole', 'sooley', 'thullu', 'keya', 'tullna keya', 'thullu keya',
        'hendruna keya', 'ninna thullu keya', 'nin thullu keya',
        
        # Family insults
        'nimavvan', 'nimmavvana', 'nimmavvana maga', 'nim amman', 'ninna amma',
        'ninna ammanakeya', 'ninna gandu maga',
        
        # Animal/Object insults
        'haavu', 'haavu bro', 'haavu bro neenu', 'haavu maga',
        'mangyan maga', 'monkey thara', 'kallu', 'kalladantha', 'kall maga',
        'kall nan magane', 'gaanchali', 'kembatti',
        
        # Mental health stigma
        'thale kettide', 'muklyappa',
        
        # Threat-related
        'saayi', 'saayi maga', 'sucide', 'die mad',
        
        # Compound insults from dataset
        'thotha maga tara', 'dagarina maga tara', 'bosodina tara',
        'hucchadanta mode', 'singri maga baribeda',
    }
    
    # Default English profanity - from bad_words.csv and profanity_english.csv
    DEFAULT_PROFANITY_ENGLISH = {
        # Common insults
        'stupid', 'idiot', 'moron', 'dumb', 'dumbass', 'dimwit', 'fool',
        'loser', 'jerk', 'bozo', 'duffer', 'rubbish', 'trash', 'worthless',
        'pathetic', 'disgusting', 'ugly', 'fat', 'freak', 'creep', 'psycho',
        
        # Vulgar terms
        'asshole', 'asswipe', 'ass', 'bastard', 'bitch', 'slut', 'whore',
        'dick', 'dickhead', 'dickhole', 'choad', 'cock', 'cockfoam',
        'cock-socker', 'butthead', 'bellend', 'twat', 'pissflaps', 'piss off',
        
        # Sexual harassment terms
        'sexy', 'sexy bro', 'sexy baby', 'naked', 'sex', 'asexual',
        'fucking balls', 'suck his dick', 'son of bitch', '69',
        
        # Slurs and hate speech
        'retard', 'nigga', 'nigger', 'fag',
        
        # Mild profanity
        'crap', 'shit', 'damn', 'hell', 'bloody hell', 'bloddy hell', 'wtf',
        
        # Behavioral insults
        'cry baby', 'fake character', 'not loyal', 'not a loyal',
        'joker', 'joker thara', 'dog', 'monkey',
        
        # Threats
        'die', 'die mad', 'sucide', 'suicide', 'kill',
        
        # Compound phrases from dataset
        'taking the piss', 'piss off attitude',
    }
    
    # Bad words extracted from bad_words.csv patterns
    DEFAULT_BAD_WORD_PATTERNS = {
        # Kannada patterns
        'thotha maga tara idiya',
        'dagarina maga tara idiya',
        'bosodina tara idiya',
        'hucchadanta mode nalli idiya',
        'singri maga baribeda',
        'haavu bro neenu maga baribeda',
        'ninna thullu keya tara idiya',
        'saayi maga tara idiya',
        'mangyan maga tara idiya',
        'kall nan magane mode nalli',
        'kembatti mode nalli',
        'nimavvan mode nalli',
        'muklyappa nataka',
        'gaandu tara mathadbeda',
        'thotha tara mathadbeda',
        
        # English patterns in Kannada context
        'you are such a',
        'stop acting like a total',
        'this behaviour is not okay',
        'people are tired of your',
        'only a will spam',
        'ninna behavior nodi everyone frustrate',
        'yaaru kuda ninna nature ishta padalla',
        'attitude beda',
    }
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        lexicon_path: Optional[str] = None
    ):
        """
        Initialize CodeMixProcessor.
        
        Args:
            config: Optional configuration dictionary.
            lexicon_path: Path to lexicon directory.
        """
        self.config = config or self._default_config()
        self.lexicon_path = lexicon_path or self._get_default_lexicon_path()
        
        # Initialize vocabularies
        self.kannada_vocab: Set[str] = set(self.DEFAULT_KANNADA_VOCAB)
        self.english_vocab: Set[str] = set(self.DEFAULT_ENGLISH_VOCAB)
        self.profanity_kannada: Set[str] = set(self.DEFAULT_PROFANITY_KANNADA)
        self.profanity_english: Set[str] = set(self.DEFAULT_PROFANITY_ENGLISH)
        self.bad_word_patterns: Set[str] = set(self.DEFAULT_BAD_WORD_PATTERNS)
        
        # Combined bad words set for quick lookup
        self.all_bad_words: Set[str] = set()
        self._build_bad_words_set()
        
        # Load lexicons if available
        self._load_lexicons()
        self._load_bad_words()
        
        # Compile patterns
        self._compile_patterns()
    
    def _default_config(self) -> Dict[str, Any]:
        """Return default configuration."""
        return {
            'lowercase': True,
            'remove_extra_spaces': True,
            'preserve_emojis': True,
            'preserve_numbers': True,
            'reduce_punctuation': True,
            'max_punct_repeat': 1,
            'detect_profanity': True,
            'expand_slang': False,
        }
    
    def _get_default_lexicon_path(self) -> str:
        """Get default lexicon directory path."""
        return os.path.join(
            os.path.dirname(__file__),
            '..', '00_data', 'lexicon'
        )
    
    def _load_lexicons(self) -> None:
        """Load vocabulary from lexicon files."""
        if not os.path.exists(self.lexicon_path):
            return
        
        # Load Kannada slang
        kannada_slang_path = os.path.join(self.lexicon_path, 'kannada_slang.csv')
        if os.path.exists(kannada_slang_path):
            try:
                with open(kannada_slang_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        term = row.get('term', '').strip().lower()
                        if term:
                            self.kannada_vocab.add(term)
            except Exception:
                pass
        
        # Load English slang
        english_slang_path = os.path.join(self.lexicon_path, 'english_slang.csv')
        if os.path.exists(english_slang_path):
            try:
                with open(english_slang_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        term = row.get('term', '').strip().lower()
                        if term:
                            self.english_vocab.add(term)
            except Exception:
                pass
        
        # Load Kannada profanity
        profanity_kn_path = os.path.join(self.lexicon_path, 'profanity_kannada.csv')
        if os.path.exists(profanity_kn_path):
            try:
                with open(profanity_kn_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        # Get transliteration (romanized form)
                        trans = row.get('transliteration', '').strip().lower()
                        term = row.get('term', '').strip().lower()
                        if trans:
                            self.profanity_kannada.add(trans)
                        if term and not self._is_kannada_script(term):
                            self.profanity_kannada.add(term)
            except Exception:
                pass
        
        # Load English profanity
        profanity_en_path = os.path.join(self.lexicon_path, 'profanity_english.csv')
        if os.path.exists(profanity_en_path):
            try:
                with open(profanity_en_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        term = row.get('term', '').strip().lower()
                        if term:
                            self.profanity_english.add(term)
            except Exception:
                pass
        
        # Rebuild bad words set after loading
        self._build_bad_words_set()
    
    def _build_bad_words_set(self) -> None:
        """Build combined set of all bad words for quick lookup."""
        self.all_bad_words = set()
        self.all_bad_words.update(self.profanity_kannada)
        self.all_bad_words.update(self.profanity_english)
        # Add individual words from patterns
        for pattern in self.bad_word_patterns:
            words = pattern.lower().split()
            self.all_bad_words.update(words)
    
    def _load_bad_words(self) -> None:
        """Load bad words from bad_words.csv dataset."""
        # Try to find bad_words.csv
        possible_paths = [
            os.path.join(os.path.dirname(__file__), '..', '00_data', 'raw', 'bad_words.csv'),
            os.path.join(self.lexicon_path, '..', 'raw', 'bad_words.csv'),
        ]
        
        bad_words_path = None
        for path in possible_paths:
            if os.path.exists(path):
                bad_words_path = path
                break
        
        if not bad_words_path:
            return
        
        try:
            with open(bad_words_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    message = row.get('message', '').lower()
                    label = row.get('label', '').lower()
                    
                    # Extract potential bad words from messages
                    if label in ['insult', 'harassment', 'threat', 'sexual harassment']:
                        # Extract words that appear in patterns
                        self._extract_bad_words_from_message(message)
        except Exception:
            pass
        
        # Rebuild the combined set
        self._build_bad_words_set()
    
    def _extract_bad_words_from_message(self, message: str) -> None:
        """Extract potential bad words from a cyberbullying message."""
        if not message:
            return
        
        # Known bad word markers in dataset
        bad_word_markers = [
            # Kannada profanity patterns
            'thotha', 'gaandu', 'gandu', 'bosodina', 'singri', 'dagarina',
            'hucchadana', 'hucchadanta', 'haavu', 'thullu', 'keya', 'soolya',
            'nimavvan', 'mangyan', 'kallu', 'kembatti', 'muklyappa', 'saayi',
            
            # English profanity patterns  
            'dickhole', 'pissflaps', 'asswipe', 'bellend', 'twat', 'choad',
            'cockfoam', 'cock-socker', 'butthead', 'dimwit', 'bozo', 'duffer',
            'nigga', 'nigger', 'fag', 'retard',
            
            # Sexual terms
            'sexy bro', 'sexy baby', 'naked', 'fucking balls',
        ]
        
        for marker in bad_word_markers:
            if marker in message:
                # Add to appropriate set
                if any(c in marker for c in ['a', 'e', 'i', 'o', 'u']) and marker.isascii():
                    # Check if it's likely Kannada (romanized) or English
                    if marker in self.DEFAULT_PROFANITY_KANNADA or marker in ['thotha', 'gaandu', 'bosodina', 'singri', 'dagarina', 'haavu', 'thullu', 'soolya']:
                        self.profanity_kannada.add(marker)
                    else:
                        self.profanity_english.add(marker)
    
    def _compile_patterns(self) -> None:
        """Compile regex patterns for efficiency."""
        # Kannada script pattern
        self.kannada_script_pattern = re.compile(r'[\u0C80-\u0CFF]+')
        
        # Whitespace normalization
        self.whitespace_pattern = re.compile(r'\s+')
        
        # Punctuation repetition
        self.punct_repeat_pattern = re.compile(r'([!?.,])\1+')
        
        # Word tokenizer (handles mixed scripts)
        self.word_pattern = re.compile(r"[\w']+|[^\w\s]", re.UNICODE)
        
        # Emoji pattern
        self.emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"
            "\U0001F300-\U0001F5FF"
            "\U0001F680-\U0001F6FF"
            "\U0001F900-\U0001F9FF"
            "\U0001FA00-\U0001FAFF"
            "\U00002600-\U000026FF"
            "\U00002700-\U000027BF"
            "]+",
            flags=re.UNICODE
        )
        
        # Hashtag ID pattern (dataset format)
        self.hashtag_id_pattern = re.compile(r'#[a-f0-9]{4}\b')
        
        # URL pattern
        self.url_pattern = re.compile(r'https?://\S+|www\.\S+')
        
        # Mention pattern
        self.mention_pattern = re.compile(r'@\w+')
    
    def _is_kannada_script(self, text: str) -> bool:
        """Check if text contains Kannada script characters."""
        for char in text:
            code = ord(char)
            if self.KANNADA_RANGE[0] <= code <= self.KANNADA_RANGE[1]:
                return True
        return False
    
    def _get_kannada_ratio(self, text: str) -> float:
        """Calculate ratio of Kannada script characters in text."""
        if not text:
            return 0.0
        
        total_chars = 0
        kannada_chars = 0
        
        for char in text:
            if char.isalpha():
                total_chars += 1
                code = ord(char)
                if self.KANNADA_RANGE[0] <= code <= self.KANNADA_RANGE[1]:
                    kannada_chars += 1
        
        return kannada_chars / total_chars if total_chars > 0 else 0.0
    
    def detect_language(self, text: str) -> str:
        """
        Detect the primary language of the text.
        
        Args:
            text: Input text to analyze.
            
        Returns:
            Language code: 'kannada', 'english', 'code-mixed', 
                          'romanized-kannada', or 'unknown'
        """
        if not text or not text.strip():
            return "unknown"
        
        text = text.strip()
        
        # Check for Kannada script
        kannada_ratio = self._get_kannada_ratio(text)
        
        if kannada_ratio > 0.8:
            return "kannada"
        elif kannada_ratio > 0.3:
            return "code-mixed"
        
        # For Roman script text, check vocabulary
        tokens = self._tokenize(text.lower())
        
        if not tokens:
            return "unknown"
        
        kannada_count = 0
        english_count = 0
        total_words = 0
        
        for token in tokens:
            if not token.isalpha():
                continue
            total_words += 1
            
            if token in self.kannada_vocab or token in self.profanity_kannada:
                kannada_count += 1
            elif token in self.english_vocab or token in self.profanity_english:
                english_count += 1
        
        if total_words == 0:
            return "unknown"
        
        kannada_word_ratio = kannada_count / total_words
        english_word_ratio = english_count / total_words
        
        # Classification logic
        if kannada_word_ratio > 0.7:
            return "romanized-kannada"
        elif english_word_ratio > 0.8:
            return "english"
        elif kannada_word_ratio > 0.3 or (kannada_count > 0 and english_count > 0):
            return "code-mixed"
        elif english_word_ratio > 0.5:
            return "english"
        else:
            return "code-mixed"
    
    def calculate_code_mix_ratio(self, text: str) -> float:
        """
        Calculate the code-mixing ratio of the text.
        
        The ratio indicates the proportion of Kannada/Romanized-Kannada
        words in the text. Higher ratio = more Kannada content.
        
        Args:
            text: Input text to analyze.
            
        Returns:
            Float between 0.0 and 1.0 representing Kannada content ratio.
        """
        if not text or not text.strip():
            return 0.0
        
        tokens = self._tokenize(text.lower())
        
        if not tokens:
            return 0.0
        
        kannada_count = 0
        total_words = 0
        
        for token in tokens:
            if not token.isalpha():
                continue
            total_words += 1
            
            # Check if Kannada script
            if self._is_kannada_script(token):
                kannada_count += 1
            # Check if Romanized Kannada
            elif token in self.kannada_vocab or token in self.profanity_kannada:
                kannada_count += 1
        
        return kannada_count / total_words if total_words > 0 else 0.0
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        if not text:
            return []
        return self.word_pattern.findall(text)
    
    def classify_tokens(self, text: str) -> List[Dict[str, str]]:
        """
        Classify each token in the text by language.
        
        Args:
            text: Input text to classify.
            
        Returns:
            List of dictionaries with 'word' and 'language' keys.
        """
        if not text:
            return []
        
        tokens = self._tokenize(text)
        classified = []
        
        for token in tokens:
            token_lower = token.lower()
            
            # Determine language
            if self._is_kannada_script(token):
                lang = "kannada"
            elif not token.isalpha():
                if token.isdigit():
                    lang = "number"
                else:
                    lang = "punctuation"
            elif token_lower in self.kannada_vocab or token_lower in self.profanity_kannada:
                lang = "romanized-kannada"
            elif token_lower in self.english_vocab or token_lower in self.profanity_english:
                lang = "english"
            else:
                # Unknown - could be either language
                lang = "unknown"
            
            classified.append({
                "word": token,
                "token": token,
                "language": lang,
                "lang": lang
            })
        
        return classified
    
    def normalize(self, text: str) -> str:
        """
        Normalize code-mixed text.
        
        Applies:
        - Lowercase conversion
        - Whitespace normalization
        - Punctuation reduction
        - Hashtag ID removal
        
        Args:
            text: Input text to normalize.
            
        Returns:
            Normalized text string.
        """
        if not text:
            return ""
        
        result = text
        
        # Remove hashtag IDs (dataset format like #5a76)
        result = self.hashtag_id_pattern.sub('', result)
        
        # Lowercase if configured
        if self.config.get('lowercase', True):
            result = result.lower()
        
        # Remove URLs
        result = self.url_pattern.sub('', result)
        
        # Normalize whitespace
        if self.config.get('remove_extra_spaces', True):
            result = self.whitespace_pattern.sub(' ', result)
        
        # Reduce punctuation repetition
        if self.config.get('reduce_punctuation', True):
            max_repeat = self.config.get('max_punct_repeat', 1)
            result = self.punct_repeat_pattern.sub(r'\1' * max_repeat, result)
        
        # Strip leading/trailing whitespace
        result = result.strip()
        
        return result
    
    def detect_profanity(self, text: str) -> Dict[str, Any]:
        """
        Detect profanity in code-mixed text.
        
        Args:
            text: Input text to analyze.
            
        Returns:
            Dictionary with:
                - 'has_profanity': bool
                - 'kannada_profanity': list of found Kannada profanity
                - 'english_profanity': list of found English profanity
                - 'bad_patterns': list of found bad word patterns
                - 'count': total count
                - 'severity': estimated severity level
        """
        if not text:
            return {
                'has_profanity': False,
                'kannada_profanity': [],
                'english_profanity': [],
                'bad_patterns': [],
                'count': 0,
                'severity': 'none'
            }
        
        text_lower = text.lower()
        tokens = self._tokenize(text_lower)
        
        found_kannada = []
        found_english = []
        found_patterns = []
        
        # Check individual tokens
        for token in tokens:
            if token in self.profanity_kannada:
                found_kannada.append(token)
            if token in self.profanity_english:
                found_english.append(token)
        
        # Check for multi-word bad patterns
        for pattern in self.bad_word_patterns:
            if pattern.lower() in text_lower:
                found_patterns.append(pattern)
        
        # Check for compound profanity (bigrams/trigrams)
        compound_profanity = self._detect_compound_profanity(text_lower)
        found_kannada.extend(compound_profanity.get('kannada', []))
        found_english.extend(compound_profanity.get('english', []))
        
        # Calculate severity
        total_count = len(found_kannada) + len(found_english) + len(found_patterns)
        severity = self._calculate_profanity_severity(
            found_kannada, found_english, found_patterns
        )
        
        return {
            'has_profanity': total_count > 0,
            'kannada_profanity': list(set(found_kannada)),
            'english_profanity': list(set(found_english)),
            'bad_patterns': found_patterns,
            'count': total_count,
            'severity': severity
        }
    
    def _detect_compound_profanity(self, text: str) -> Dict[str, List[str]]:
        """Detect compound/multi-word profanity phrases."""
        found = {'kannada': [], 'english': []}
        
        # Kannada compound profanity
        kannada_compounds = [
            'thotha maga', 'dagarina maga', 'singri maga', 'soolya maga',
            'haavu bro', 'mangyan maga', 'kall maga', 'ninna amma',
            'nim amman', 'thullu keya', 'hendruna keya', 'nimmavvana maga',
            'thotha tara', 'gaandu tara', 'bosodina tara', 'hucchadanta mode',
            'saayi maga', 'kembatti mode', 'nimavvan mode',
        ]
        
        for compound in kannada_compounds:
            if compound in text:
                found['kannada'].append(compound)
        
        # English compound profanity
        english_compounds = [
            'son of bitch', 'suck his dick', 'fucking balls', 'sexy bro',
            'sexy baby', 'cock-socker', 'bloody hell', 'die mad',
            'cry baby', 'piss off', 'taking the piss',
        ]
        
        for compound in english_compounds:
            if compound in text:
                found['english'].append(compound)
        
        return found
    
    def _calculate_profanity_severity(
        self,
        kannada: List[str],
        english: List[str],
        patterns: List[str]
    ) -> str:
        """Calculate overall severity of profanity found."""
        # High severity terms
        high_severity_kn = {'gaandu', 'soolya', 'soole', 'bosodina', 'thullu', 
                           'keya', 'ninna amma', 'nimmavvana maga', 'saayi'}
        high_severity_en = {'nigga', 'nigger', 'retard', 'fag', 'cock-socker',
                           'suck his dick', 'fucking balls', 'die', 'sucide',
                           'suicide', 'kill'}
        
        # Check for high severity
        for term in kannada:
            if term in high_severity_kn:
                return 'high'
        for term in english:
            if term in high_severity_en:
                return 'high'
        
        # Medium severity - multiple profanity or certain terms
        if len(kannada) + len(english) >= 2:
            return 'medium'
        
        if kannada or english or patterns:
            return 'low'
        
        return 'none'
    
    def detect_bad_words(self, text: str) -> Dict[str, Any]:
        """
        Detect all bad words (alias for detect_profanity with extended info).
        
        Args:
            text: Input text to analyze.
            
        Returns:
            Dictionary with comprehensive bad word detection results.
        """
        result = self.detect_profanity(text)
        
        # Add additional context
        text_lower = text.lower()
        
        # Check for threat indicators
        threat_words = ['sucide', 'suicide', 'die', 'kill', 'saayi', 'die mad']
        result['has_threat'] = any(w in text_lower for w in threat_words)
        
        # Check for sexual content
        sexual_words = ['thullu', 'keya', 'sexy', 'naked', 'sex', 'dick', 
                       'cock', 'fucking', 'hendruna keya']
        result['has_sexual_content'] = any(w in text_lower for w in sexual_words)
        
        # Check for slurs
        slur_words = ['nigga', 'nigger', 'fag', 'retard']
        result['has_slurs'] = any(w in text_lower for w in slur_words)
        
        return result
    
    def identify_switch_points(self, text: str) -> List[Dict[str, Any]]:
        """
        Identify points where language switches occur.
        
        Args:
            text: Input text to analyze.
            
        Returns:
            List of switch points with position and languages.
        """
        classified = self.classify_tokens(text)
        switch_points = []
        
        prev_lang = None
        for i, token_info in enumerate(classified):
            curr_lang = token_info['language']
            
            # Skip punctuation and numbers
            if curr_lang in ['punctuation', 'number']:
                continue
            
            if prev_lang and prev_lang != curr_lang and curr_lang != 'unknown':
                switch_points.append({
                    'position': i,
                    'from_lang': prev_lang,
                    'to_lang': curr_lang,
                    'word': token_info['word']
                })
            
            if curr_lang != 'unknown':
                prev_lang = curr_lang
        
        return switch_points
    
    def identify_patterns(self, text: str) -> Dict[str, Any]:
        """
        Identify code-mixing patterns in the text.
        
        Patterns detected:
        - Kannada base + English insult
        - English base + Kannada fillers
        - Intensifier patterns
        - Address term usage
        
        Args:
            text: Input text to analyze.
            
        Returns:
            Dictionary with identified patterns.
        """
        if not text:
            return {'patterns': []}
        
        text_lower = text.lower()
        classified = self.classify_tokens(text)
        patterns = []
        
        # Pattern: Kannada intensifier + English adjective
        intensifiers = ['tumba', 'sakkat', 'full', 'swalpa', 'jaasti']
        adjectives = ['irritating', 'annoying', 'boring', 'stupid', 'dumb', 
                      'toxic', 'fake', 'waste', 'useless', 'pathetic']
        
        for intensifier in intensifiers:
            for adj in adjectives:
                if f"{intensifier} {adj}" in text_lower:
                    patterns.append({
                        'type': 'intensifier_adjective',
                        'pattern': f"{intensifier} {adj}",
                        'severity': 'medium'
                    })
        
        # Pattern: Address terms (maga, machaa, re, yaar)
        address_terms = ['maga', 'machaa', 'macha', 're', 'yaar', 'guru']
        for term in address_terms:
            if term in text_lower.split():
                patterns.append({
                    'type': 'address_term',
                    'pattern': term,
                    'severity': 'low'
                })
        
        # Pattern: Profanity with "thara" (like a)
        if ' tara ' in text_lower or ' thara ' in text_lower:
            patterns.append({
                'type': 'comparison_insult',
                'pattern': 'X tara/thara',
                'severity': 'medium'
            })
        
        # Pattern: Profanity detected
        profanity = self.detect_profanity(text)
        if profanity['has_profanity']:
            patterns.append({
                'type': 'profanity',
                'kannada': profanity['kannada_profanity'],
                'english': profanity['english_profanity'],
                'severity': 'high' if profanity['count'] > 1 else 'medium'
            })
        
        return {
            'patterns': patterns,
            'pattern_count': len(patterns),
            'has_high_severity': any(p.get('severity') == 'high' for p in patterns)
        }
    
    def process(self, text: str) -> Dict[str, Any]:
        """
        Full processing pipeline for code-mixed text.
        
        Args:
            text: Input text to process.
            
        Returns:
            Dictionary with all analysis results.
        """
        if not text:
            return {
                'original': '',
                'normalized': '',
                'language': 'unknown',
                'code_mix_ratio': 0.0,
                'tokens': [],
                'patterns': {},
                'profanity': {}
            }
        
        normalized = self.normalize(text)
        
        return {
            'original': text,
            'normalized': normalized,
            'language': self.detect_language(text),
            'code_mix_ratio': self.calculate_code_mix_ratio(text),
            'tokens': self.classify_tokens(text),
            'switch_points': self.identify_switch_points(text),
            'patterns': self.identify_patterns(text),
            'profanity': self.detect_profanity(text)
        }
    
    def preprocess(self, text: str) -> str:
        """
        Alias for normalize(). Preprocess text for downstream tasks.
        
        Args:
            text: Input text.
            
        Returns:
            Preprocessed text.
        """
        return self.normalize(text)
    
    def process_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Process multiple texts.
        
        Args:
            texts: List of texts to process.
            
        Returns:
            List of processing results.
        """
        return [self.process(text) for text in texts]
    
    def get_vocabulary_stats(self) -> Dict[str, int]:
        """
        Get vocabulary statistics.
        
        Returns:
            Dictionary with vocabulary sizes.
        """
        return {
            'kannada_vocab_size': len(self.kannada_vocab),
            'english_vocab_size': len(self.english_vocab),
            'kannada_profanity_size': len(self.profanity_kannada),
            'english_profanity_size': len(self.profanity_english)
        }
    
    def add_to_vocabulary(
        self,
        words: List[str],
        vocab_type: str = 'kannada'
    ) -> None:
        """
        Add words to vocabulary.
        
        Args:
            words: List of words to add.
            vocab_type: 'kannada', 'english', 'profanity_kannada', 'profanity_english'
        """
        vocab_map = {
            'kannada': self.kannada_vocab,
            'english': self.english_vocab,
            'profanity_kannada': self.profanity_kannada,
            'profanity_english': self.profanity_english
        }
        
        if vocab_type in vocab_map:
            vocab_map[vocab_type].update(w.lower() for w in words)
    
    def expand_slang(self, text: str) -> str:
        """
        Expand slang terms in text (placeholder for future implementation).
        
        Args:
            text: Input text with slang.
            
        Returns:
            Text with expanded slang.
        """
        # This is a placeholder - actual implementation would use slang_expander
        return text
    
    def transliterate_to_kannada(self, text: str) -> str:
        """
        Transliterate Romanized Kannada to Kannada script (placeholder).
        
        Args:
            text: Romanized Kannada text.
            
        Returns:
            Kannada script text.
        """
        # Placeholder - actual implementation in transliterator module
        return text
    
    def transliterate_to_roman(self, text: str) -> str:
        """
        Transliterate Kannada script to Roman (placeholder).
        
        Args:
            text: Kannada script text.
            
        Returns:
            Romanized text.
        """
        # Placeholder - actual implementation in transliterator module
        return text
