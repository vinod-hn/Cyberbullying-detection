# Slang Expander
"""
SlangExpander: Comprehensive slang and abbreviation expansion for cyberbullying detection.
Handles Kannada slang, English slang, internet abbreviations, and code-mixed expressions.
Optimized for Kannada-English code-mixed text in social media contexts.
"""

import re
import os
import csv
import json
from typing import List, Dict, Optional, Tuple, Any, Union, Set
from collections import defaultdict


class SlangExpander:
    """
    Comprehensive slang expander for cyberbullying detection.
    
    Features:
    - Kannada slang expansion (Romanized)
    - English slang and internet abbreviations
    - Code-mixed slang patterns
    - Context-aware expansion
    - Profanity-aware expansion
    - Support for lexicon files
    
    Attributes:
        kannada_slang: Dictionary of Kannada slang terms
        english_slang: Dictionary of English slang terms
        abbreviations: Dictionary of abbreviations
        config: Configuration settings
    """
    
    # Default Kannada slang dictionary (from dataset analysis)
    DEFAULT_KANNADA_SLANG = {
        # Address terms
        'machaa': 'friend',
        'macha': 'friend',
        'maccha': 'friend',
        'maga': 'buddy',
        'yaar': 'friend',
        're': 'hey',
        'guru': 'friend',
        'boss': 'friend',
        'bossu': 'friend',
        'anna': 'elder brother',
        'akka': 'elder sister',
        'appa': 'father',
        'amma': 'mother',
        'saar': 'sir',
        
        # Intensifiers
        'tumba': 'very',
        'thumba': 'very',
        'sakkat': 'very',
        'sakkath': 'very',
        'full': 'very',
        'swalpa': 'little',
        'jaasti': 'more',
        'chennag': 'nicely',
        'chennaag': 'nicely',
        
        # Common verbs (informal -> formal)
        'madthiya': 'are you doing',
        'bartiya': 'will you come',
        'hogthiya': 'are you going',
        'nodthiya': 'are you seeing',
        'kelthiya': 'are you listening',
        'heltiya': 'are you telling',
        'madtiya': 'are you doing',
        'bartira': 'will you come',
        'barthini': 'I will come',
        'hogthini': 'I will go',
        'madthini': 'I will do',
        'maadu': 'do',
        'hogi': 'go',
        'baa': 'come',
        'helu': 'tell',
        'nodu': 'see',
        'kelu': 'listen',
        'haaki': 'put',
        'tagond': 'take',
        'hodbitta': 'went away',
        'odi': 'run',
        'aaytu': 'done',
        'agide': 'has become',
        
        # Negation
        'illa': 'no',
        'beda': 'do not want',
        'beka': 'want',
        'agalla': 'cannot happen',
        'baralla': 'will not come',
        'gotilla': 'do not know',
        'gottilla': 'do not know',
        'madbeda': 'do not do',
        'hogbeda': 'do not go',
        'helbeda': 'do not tell',
        'illade': 'without',
        'illaadre': 'if not',
        
        # Question words
        'yaake': 'why',
        'yelli': 'where',
        'yavaga': 'when',
        'yavag': 'when',
        'hege': 'how',
        'yaavdu': 'which',
        'yenu': 'what',
        'enu': 'what',
        'eshtu': 'how much',
        'yaaru': 'who',
        
        # Common words
        'ide': 'is there',
        'gottu': 'know',
        'beku': 'want',
        'andre': 'then',
        'ashte': 'that is all',
        'kelsa': 'work',
        'oota': 'food',
        'sari': 'okay',
        'sumne': 'simply',
        'elli': 'where',
        'hogidaroo': 'wherever you go',
        'anta': 'that',
        'ansta': 'feels like',
        'tara': 'like',
        'kanstide': 'seems',
        'nodoke': 'to see',
        'togondiya': 'take it',
        'ivananna': 'this person',
        'houdu': 'yes',
        'alva': 'right',
        'aadre': 'but',
        
        # Time expressions
        'ivattu': 'today',
        'naale': 'tomorrow',
        'prati': 'every',
        'yavaglu': 'always',
        'dina': 'day',
        'nimisha': 'minute',
        
        # Postpositions
        'nalli': 'in',
        'ge': 'to',
        'inda': 'from',
        'jote': 'with',
        'hatra': 'near',
        'mele': 'on',
        'kelage': 'below',
        
        # Pronouns
        'nee': 'you',
        'neenu': 'you',
        'naanu': 'I',
        'naan': 'I',
        'avnu': 'he',
        'avlu': 'she',
        'avru': 'they',
        'navu': 'we',
        'ivan': 'this man',
        'ivalu': 'this woman',
        'ivru': 'these people',
        'ninna': 'your',
        'nanna': 'my',
        'nimma': 'your (formal)',
        'namma': 'our',
        'nin': 'your',
        
        # Expressions
        'ayyo': 'oh no',
        'ayyoo': 'oh no',
        'yappa': 'father exclamation',
        'arey': 'hey',
        'abe': 'hey',
        'eno': 'something',
        'yeno': 'something',
        
        # Academic slang (from dataset)
        'ppt': 'presentation',
        'lec': 'lecture',
        'lab': 'laboratory',
        'fest': 'festival',
        'grp': 'group',
        
        # Insults (need to preserve for detection)
        'thotha': 'fool',
        'thota': 'fool',
        'singri': 'idiot',
        'dagarina': 'scoundrel',
        'hucchadana': 'mad person',
        'hucchadanta': 'like mad',
        'moorkha': 'stupid',
        'bewakoof': 'fool',
        'joker': 'clown',
        'waste': 'useless',
        'useless': 'useless',
        'irritating': 'annoying',
        'disgusting': 'disgusting',
        'boring': 'boring',
        'hopeless': 'hopeless',
        'dumb': 'stupid',
        'lazy': 'lazy',
        'headache': 'problem',
        'noovu': 'pain',
        'thale': 'head',
    }
    
    # Default English slang dictionary
    DEFAULT_ENGLISH_SLANG = {
        # Address terms
        'dude': 'friend',
        'bro': 'brother',
        'man': 'person',
        'fam': 'family',
        'squad': 'friend group',
        'homie': 'friend',
        'bestie': 'best friend',
        
        # Greetings
        'yo': 'hey',
        'sup': 'what is up',
        'wassup': 'what is up',
        'heya': 'hello',
        'hiya': 'hello',
        
        # Reactions
        'lol': 'laughing out loud',
        'lmao': 'laughing hard',
        'lmfao': 'laughing very hard',
        'rofl': 'rolling on floor laughing',
        'omg': 'oh my god',
        'omfg': 'oh my god',
        'wtf': 'what the hell',
        'wth': 'what the hell',
        'smh': 'shaking my head',
        'facepalm': 'disappointment',
        'ikr': 'I know right',
        
        # Discourse markers
        'tbh': 'to be honest',
        'ngl': 'not going to lie',
        'imo': 'in my opinion',
        'imho': 'in my honest opinion',
        'fyi': 'for your information',
        'btw': 'by the way',
        'afaik': 'as far as I know',
        'iirc': 'if I recall correctly',
        
        # Responses
        'idk': 'I do not know',
        'idc': 'I do not care',
        'idgaf': 'I do not care',
        'whatever': 'I do not care',
        'k': 'okay',
        'kk': 'okay',
        'ok': 'okay',
        'okie': 'okay',
        'bet': 'okay',
        'aight': 'alright',
        'ight': 'alright',
        'yep': 'yes',
        'yup': 'yes',
        'nope': 'no',
        'nah': 'no',
        
        # Time
        'rn': 'right now',
        'atm': 'at the moment',
        'asap': 'as soon as possible',
        'l8r': 'later',
        'tmrw': 'tomorrow',
        '2day': 'today',
        '2nite': 'tonight',
        
        # Status
        'brb': 'be right back',
        'gtg': 'got to go',
        'g2g': 'got to go',
        'ttyl': 'talk to you later',
        'cya': 'see you',
        'bbl': 'be back later',
        'afk': 'away from keyboard',
        
        # Modifiers
        'lowkey': 'somewhat',
        'highkey': 'very much',
        'hella': 'very',
        'v': 'very',
        'kinda': 'kind of',
        'sorta': 'sort of',
        'prolly': 'probably',
        'probs': 'probably',
        'def': 'definitely',
        'totes': 'totally',
        'legit': 'legitimately',
        'literally': 'actually',
        'basically': 'essentially',
        
        # Judgments
        'sus': 'suspicious',
        'sketchy': 'suspicious',
        'shady': 'suspicious',
        'salty': 'bitter',
        'toxic': 'harmful',
        'cringe': 'embarrassing',
        'cringey': 'embarrassing',
        'lame': 'uncool',
        'wack': 'bad',
        'bogus': 'fake',
        'cap': 'lie',
        'nocap': 'no lie',
        
        # Praise
        'fire': 'excellent',
        'lit': 'exciting',
        'dope': 'cool',
        'sick': 'cool',
        'epic': 'awesome',
        'goat': 'greatest of all time',
        'slay': 'do excellently',
        'based': 'admirable',
        'bussin': 'really good',
        'slaps': 'is really good',
        
        # Actions
        'flex': 'show off',
        'ghosted': 'ignored',
        'ghosting': 'ignoring',
        'spam': 'send repeatedly',
        'spamming': 'sending repeatedly',
        'troll': 'provoke',
        'trolling': 'provoking',
        'dm': 'direct message',
        'ping': 'message',
        
        # Emotional
        'mood': 'relatable',
        'vibe': 'atmosphere',
        'vibes': 'feelings',
        'feels': 'emotions',
        'triggered': 'upset',
        'butthurt': 'offended',
        'salty': 'bitter',
        
        # Emphasis
        'period': 'end of discussion',
        'periodt': 'end of discussion',
        'fr': 'for real',
        'frfr': 'for real for real',
        'deadass': 'seriously',
        'ong': 'on god',
        'nbs': 'no bullshit',
        
        # Identities (judgmental)
        'simp': 'overly devoted person',
        'karen': 'entitled complainer',
        'boomer': 'old fashioned person',
        'noob': 'newcomer',
        'newbie': 'newcomer',
        
        # Phrases
        'im dead': 'that is hilarious',
        'dying': 'laughing hard',
        'cant even': 'overwhelmed',
        'no offense': 'do not be offended',
        'just saying': 'stating opinion',
        'just my opinion': 'personal opinion',
        'dont you think': 'seeking agreement',
    }
    
    # Internet abbreviations
    DEFAULT_ABBREVIATIONS = {
        # Common
        'u': 'you',
        'ur': 'your',
        'r': 'are',
        'y': 'why',
        'n': 'and',
        'b': 'be',
        'c': 'see',
        '4': 'for',
        '2': 'to',
        'w': 'with',
        'w/': 'with',
        'w/o': 'without',
        'b4': 'before',
        'gr8': 'great',
        'l8': 'late',
        'h8': 'hate',
        'm8': 'mate',
        'str8': 'straight',
        'pls': 'please',
        'plz': 'please',
        'thx': 'thanks',
        'thnx': 'thanks',
        'ty': 'thank you',
        'tysm': 'thank you so much',
        'np': 'no problem',
        'nvm': 'never mind',
        'nvr': 'never',
        'evr': 'ever',
        'msg': 'message',
        'txt': 'text',
        'pic': 'picture',
        'pics': 'pictures',
        'vid': 'video',
        'info': 'information',
        'convo': 'conversation',
        'prob': 'problem',
        'probs': 'problems',
        'diff': 'different',
        'obv': 'obviously',
        'obvs': 'obviously',
        'esp': 'especially',
        'tho': 'though',
        'altho': 'although',
        'bc': 'because',
        'cuz': 'because',
        'coz': 'because',
        'cos': 'because',
        'gonna': 'going to',
        'gotta': 'got to',
        'wanna': 'want to',
        'lemme': 'let me',
        'gimme': 'give me',
        'dunno': 'do not know',
        'aint': 'is not',
        'isnt': 'is not',
        'doesnt': 'does not',
        'dont': 'do not',
        'didnt': 'did not',
        'wont': 'will not',
        'cant': 'cannot',
        'couldnt': 'could not',
        'shouldnt': 'should not',
        'wouldnt': 'would not',
        'havent': 'have not',
        'hasnt': 'has not',
        'hadnt': 'had not',
        'im': 'I am',
        'ive': 'I have',
        'ill': 'I will',
        'id': 'I would',
        'youre': 'you are',
        'youve': 'you have',
        'youll': 'you will',
        'youd': 'you would',
        'hes': 'he is',
        'shes': 'she is',
        'its': 'it is',
        'were': 'we are',
        'weve': 'we have',
        'theyre': 'they are',
        'theyve': 'they have',
        'theyll': 'they will',
        'whats': 'what is',
        'whos': 'who is',
        'wheres': 'where is',
        'hows': 'how is',
        'thats': 'that is',
    }
    
    # Code-mixed patterns (Kannada-English hybrid)
    DEFAULT_CODE_MIX_PATTERNS = {
        # Verb patterns
        'madthiya': 'are you doing',
        'bartiya': 'are you coming',
        'check maadu': 'check it',
        'submit madbeku': 'have to submit',
        'share madidakke': 'for sharing',
        'join agthini': 'I will join',
        'wait maadu': 'wait',
        'repeat maadu': 'repeat',
        'decide maadi': 'decide',
        'ping madbedi': 'do not ping',
        'text madthiya': 'are you texting',
        'follow madthiya': 'are you following',
        'remove maadona': 'let us remove',
        'complaint haakthini': 'I will complain',
        'behave madthiya': 'are you behaving',
        
        # Noun patterns
        'time gothu': 'know the time',
        'online time': 'online time',
        'group nalli': 'in the group',
        'class nalli': 'in the class',
        'mode nalli': 'in mode',
        'lesson sigutte': 'will get lesson',
        
        # Adjective patterns
        'full lazy': 'very lazy',
        'full dumb': 'very dumb',
        'full time': 'always',
        'waste fellow': 'useless person',
        'complete waste': 'totally useless',
        'tumba irritating': 'very irritating',
        'tumba disgusting': 'very disgusting',
        'sakkat dumb': 'very stupid',
        
        # Expression patterns
        'feel agutte': 'feels like',
        'stalking tara': 'like stalking',
        'joker ne': 'is a joker',
        'better agide': 'is better',
        'better ansta': 'feels better',
    }
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        lexicon_path: Optional[str] = None
    ):
        """
        Initialize SlangExpander.
        
        Args:
            config: Optional configuration dictionary.
            lexicon_path: Path to lexicon directory.
        """
        self.config = config or self._default_config()
        self.lexicon_path = lexicon_path or self._get_default_lexicon_path()
        
        # Initialize dictionaries
        self.kannada_slang: Dict[str, str] = dict(self.DEFAULT_KANNADA_SLANG)
        self.english_slang: Dict[str, str] = dict(self.DEFAULT_ENGLISH_SLANG)
        self.abbreviations: Dict[str, str] = dict(self.DEFAULT_ABBREVIATIONS)
        self.code_mix_patterns: Dict[str, str] = dict(self.DEFAULT_CODE_MIX_PATTERNS)
        
        # Additional metadata
        self.kannada_metadata: Dict[str, Dict[str, str]] = {}
        self.english_metadata: Dict[str, Dict[str, str]] = {}
        
        # Load lexicons if available
        self._load_lexicons()
        
        # Compile patterns
        self._compile_patterns()
    
    def _default_config(self) -> Dict[str, Any]:
        """Return default configuration."""
        return {
            'expand_kannada_slang': True,
            'expand_english_slang': True,
            'expand_abbreviations': True,
            'expand_code_mix': True,
            'preserve_original': False,
            'case_sensitive': False,
            'preserve_profanity': True,
            'max_expansions': 3,
            'add_brackets': False,
            'word_boundary': True,
        }
    
    def _get_default_lexicon_path(self) -> str:
        """Get default lexicon directory path."""
        return os.path.join(
            os.path.dirname(__file__),
            '..', '00_data', 'lexicon'
        )
    
    def _load_lexicons(self) -> None:
        """Load slang dictionaries from lexicon files."""
        if not os.path.exists(self.lexicon_path):
            return
        
        # Load Kannada slang
        kannada_path = os.path.join(self.lexicon_path, 'kannada_slang.csv')
        if os.path.exists(kannada_path):
            try:
                with open(kannada_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        term = row.get('term', '').strip().lower()
                        meaning = row.get('meaning', '').strip()
                        if term and meaning:
                            self.kannada_slang[term] = meaning
                            self.kannada_metadata[term] = {
                                'usage_context': row.get('usage_context', ''),
                                'formality': row.get('formality', ''),
                                'region': row.get('region', '')
                            }
            except Exception:
                pass
        
        # Load English slang
        english_path = os.path.join(self.lexicon_path, 'english_slang.csv')
        if os.path.exists(english_path):
            try:
                with open(english_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        term = row.get('term', '').strip().lower()
                        meaning = row.get('meaning', '').strip()
                        if term and meaning:
                            self.english_slang[term] = meaning
                            self.english_metadata[term] = {
                                'usage_context': row.get('usage_context', ''),
                                'formality': row.get('formality', ''),
                                'category': row.get('category', '')
                            }
            except Exception:
                pass
    
    def _compile_patterns(self) -> None:
        """Compile regex patterns for efficiency."""
        # Word boundary pattern
        if self.config.get('word_boundary', True):
            self.boundary = r'\b'
        else:
            self.boundary = ''
        
        # Build combined patterns for each dictionary
        self._kannada_pattern = self._build_pattern(self.kannada_slang)
        self._english_pattern = self._build_pattern(self.english_slang)
        self._abbrev_pattern = self._build_pattern(self.abbreviations)
        
        # Code-mix patterns (phrase-level, no word boundary)
        self._code_mix_patterns_compiled = {}
        for phrase, expansion in self.code_mix_patterns.items():
            pattern = re.compile(
                re.escape(phrase),
                re.IGNORECASE if not self.config.get('case_sensitive', False) else 0
            )
            self._code_mix_patterns_compiled[pattern] = expansion
    
    def _build_pattern(self, dictionary: Dict[str, str]) -> Optional[re.Pattern]:
        """Build compiled regex pattern from dictionary keys."""
        if not dictionary:
            return None
        
        # Sort by length (longer first) to match longer terms first
        sorted_terms = sorted(dictionary.keys(), key=len, reverse=True)
        
        # Escape and join
        escaped = [re.escape(term) for term in sorted_terms]
        pattern_str = f"{self.boundary}({'|'.join(escaped)}){self.boundary}"
        
        flags = re.IGNORECASE if not self.config.get('case_sensitive', False) else 0
        return re.compile(pattern_str, flags)
    
    def expand_kannada(self, text: str) -> str:
        """
        Expand Kannada slang terms in text.
        
        Args:
            text: Input text with Kannada slang
            
        Returns:
            Text with Kannada slang expanded
        """
        if not text or not self.config.get('expand_kannada_slang', True):
            return text
        
        if self._kannada_pattern is None:
            return text
        
        def replace_match(match):
            term = match.group(0).lower()
            expansion = self.kannada_slang.get(term)
            if expansion:
                if self.config.get('preserve_original', False):
                    return f"{match.group(0)} ({expansion})"
                elif self.config.get('add_brackets', False):
                    return f"[{expansion}]"
                return expansion
            return match.group(0)
        
        return self._kannada_pattern.sub(replace_match, text)
    
    def expand_english(self, text: str) -> str:
        """
        Expand English slang terms in text.
        
        Args:
            text: Input text with English slang
            
        Returns:
            Text with English slang expanded
        """
        if not text or not self.config.get('expand_english_slang', True):
            return text
        
        if self._english_pattern is None:
            return text
        
        def replace_match(match):
            term = match.group(0).lower()
            expansion = self.english_slang.get(term)
            if expansion:
                if self.config.get('preserve_original', False):
                    return f"{match.group(0)} ({expansion})"
                elif self.config.get('add_brackets', False):
                    return f"[{expansion}]"
                return expansion
            return match.group(0)
        
        return self._english_pattern.sub(replace_match, text)
    
    def expand_abbreviations(self, text: str) -> str:
        """
        Expand internet abbreviations in text.
        
        Args:
            text: Input text with abbreviations
            
        Returns:
            Text with abbreviations expanded
        """
        if not text or not self.config.get('expand_abbreviations', True):
            return text
        
        if self._abbrev_pattern is None:
            return text
        
        def replace_match(match):
            term = match.group(0).lower()
            expansion = self.abbreviations.get(term)
            if expansion:
                if self.config.get('preserve_original', False):
                    return f"{match.group(0)} ({expansion})"
                elif self.config.get('add_brackets', False):
                    return f"[{expansion}]"
                return expansion
            return match.group(0)
        
        return self._abbrev_pattern.sub(replace_match, text)
    
    def expand_code_mix(self, text: str) -> str:
        """
        Expand code-mixed patterns (Kannada-English hybrid).
        
        Args:
            text: Input text with code-mixed patterns
            
        Returns:
            Text with code-mixed patterns expanded
        """
        if not text or not self.config.get('expand_code_mix', True):
            return text
        
        result = text
        for pattern, expansion in self._code_mix_patterns_compiled.items():
            if self.config.get('preserve_original', False):
                result = pattern.sub(f"\\g<0> ({expansion})", result)
            elif self.config.get('add_brackets', False):
                result = pattern.sub(f"[{expansion}]", result)
            else:
                result = pattern.sub(expansion, result)
        
        return result
    
    def expand(self, text: str) -> str:
        """
        Expand all slang, abbreviations, and code-mix patterns.
        
        Args:
            text: Input text
            
        Returns:
            Text with all expansions applied
        """
        if not text:
            return text
        
        result = text
        
        # Apply expansions in order (code-mix first as it's phrase-level)
        if self.config.get('expand_code_mix', True):
            result = self.expand_code_mix(result)
        
        if self.config.get('expand_kannada_slang', True):
            result = self.expand_kannada(result)
        
        if self.config.get('expand_english_slang', True):
            result = self.expand_english(result)
        
        if self.config.get('expand_abbreviations', True):
            result = self.expand_abbreviations(result)
        
        # Clean up extra spaces
        result = re.sub(r'\s+', ' ', result).strip()
        
        return result
    
    def get_slang_info(self, term: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a slang term.
        
        Args:
            term: Slang term to look up
            
        Returns:
            Dictionary with term information or None
        """
        term_lower = term.lower()
        
        # Check Kannada slang
        if term_lower in self.kannada_slang:
            return {
                'term': term,
                'expansion': self.kannada_slang[term_lower],
                'language': 'kannada',
                'metadata': self.kannada_metadata.get(term_lower, {})
            }
        
        # Check English slang
        if term_lower in self.english_slang:
            return {
                'term': term,
                'expansion': self.english_slang[term_lower],
                'language': 'english',
                'metadata': self.english_metadata.get(term_lower, {})
            }
        
        # Check abbreviations
        if term_lower in self.abbreviations:
            return {
                'term': term,
                'expansion': self.abbreviations[term_lower],
                'language': 'abbreviation',
                'metadata': {}
            }
        
        return None
    
    def detect_slang(self, text: str) -> Dict[str, Any]:
        """
        Detect all slang terms in text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with detected slang information
        """
        if not text:
            return {
                'kannada_slang': [],
                'english_slang': [],
                'abbreviations': [],
                'code_mix': [],
                'total_count': 0
            }
        
        text_lower = text.lower()
        
        # Detect Kannada slang
        kannada_found = []
        if self._kannada_pattern:
            matches = self._kannada_pattern.findall(text_lower)
            kannada_found = list(set(matches))
        
        # Detect English slang
        english_found = []
        if self._english_pattern:
            matches = self._english_pattern.findall(text_lower)
            english_found = list(set(matches))
        
        # Detect abbreviations
        abbrev_found = []
        if self._abbrev_pattern:
            matches = self._abbrev_pattern.findall(text_lower)
            abbrev_found = list(set(matches))
        
        # Detect code-mix patterns
        code_mix_found = []
        for pattern in self._code_mix_patterns_compiled:
            if pattern.search(text):
                code_mix_found.append(pattern.pattern)
        
        total = len(kannada_found) + len(english_found) + len(abbrev_found) + len(code_mix_found)
        
        return {
            'kannada_slang': kannada_found,
            'english_slang': english_found,
            'abbreviations': abbrev_found,
            'code_mix': code_mix_found,
            'total_count': total
        }
    
    def get_expansion_report(self, text: str) -> Dict[str, Any]:
        """
        Get detailed report of expansions in text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with expansion details
        """
        if not text:
            return {
                'original': text,
                'expanded': text,
                'expansions': [],
                'expansion_count': 0
            }
        
        expansions = []
        detected = self.detect_slang(text)
        
        # Build expansion list
        for term in detected['kannada_slang']:
            if term in self.kannada_slang:
                expansions.append({
                    'term': term,
                    'expansion': self.kannada_slang[term],
                    'type': 'kannada_slang'
                })
        
        for term in detected['english_slang']:
            if term in self.english_slang:
                expansions.append({
                    'term': term,
                    'expansion': self.english_slang[term],
                    'type': 'english_slang'
                })
        
        for term in detected['abbreviations']:
            if term in self.abbreviations:
                expansions.append({
                    'term': term,
                    'expansion': self.abbreviations[term],
                    'type': 'abbreviation'
                })
        
        for pattern_str in detected['code_mix']:
            for phrase, exp in self.code_mix_patterns.items():
                if re.escape(phrase) == pattern_str:
                    expansions.append({
                        'term': phrase,
                        'expansion': exp,
                        'type': 'code_mix'
                    })
                    break
        
        return {
            'original': text,
            'expanded': self.expand(text),
            'expansions': expansions,
            'expansion_count': len(expansions)
        }
    
    def add_slang(
        self,
        term: str,
        expansion: str,
        slang_type: str = 'kannada',
        metadata: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Add a new slang term to the dictionary.
        
        Args:
            term: Slang term
            expansion: Expanded meaning
            slang_type: Type of slang ('kannada', 'english', 'abbreviation', 'code_mix')
            metadata: Optional metadata dictionary
        """
        term_lower = term.lower()
        
        if slang_type == 'kannada':
            self.kannada_slang[term_lower] = expansion
            if metadata:
                self.kannada_metadata[term_lower] = metadata
        elif slang_type == 'english':
            self.english_slang[term_lower] = expansion
            if metadata:
                self.english_metadata[term_lower] = metadata
        elif slang_type == 'abbreviation':
            self.abbreviations[term_lower] = expansion
        elif slang_type == 'code_mix':
            self.code_mix_patterns[term_lower] = expansion
        
        # Recompile patterns
        self._compile_patterns()
    
    def remove_slang(self, term: str, slang_type: str = 'kannada') -> bool:
        """
        Remove a slang term from the dictionary.
        
        Args:
            term: Slang term to remove
            slang_type: Type of slang
            
        Returns:
            True if term was removed
        """
        term_lower = term.lower()
        removed = False
        
        if slang_type == 'kannada' and term_lower in self.kannada_slang:
            del self.kannada_slang[term_lower]
            self.kannada_metadata.pop(term_lower, None)
            removed = True
        elif slang_type == 'english' and term_lower in self.english_slang:
            del self.english_slang[term_lower]
            self.english_metadata.pop(term_lower, None)
            removed = True
        elif slang_type == 'abbreviation' and term_lower in self.abbreviations:
            del self.abbreviations[term_lower]
            removed = True
        elif slang_type == 'code_mix' and term_lower in self.code_mix_patterns:
            del self.code_mix_patterns[term_lower]
            removed = True
        
        if removed:
            self._compile_patterns()
        
        return removed
    
    def get_vocabulary_stats(self) -> Dict[str, int]:
        """Get statistics about loaded vocabularies."""
        return {
            'kannada_slang_count': len(self.kannada_slang),
            'english_slang_count': len(self.english_slang),
            'abbreviation_count': len(self.abbreviations),
            'code_mix_count': len(self.code_mix_patterns),
            'total': (
                len(self.kannada_slang) +
                len(self.english_slang) +
                len(self.abbreviations) +
                len(self.code_mix_patterns)
            )
        }
    
    def process(self, text: str) -> Dict[str, Any]:
        """
        Process text with full slang analysis and expansion.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with processing results
        """
        return {
            'original': text,
            'expanded': self.expand(text),
            'detected': self.detect_slang(text),
            'report': self.get_expansion_report(text)
        }
    
    def process_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Process multiple texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of processing results
        """
        return [self.process(text) for text in texts]
    
    def __repr__(self) -> str:
        """String representation."""
        stats = self.get_vocabulary_stats()
        return (
            f"SlangExpander("
            f"kannada={stats['kannada_slang_count']}, "
            f"english={stats['english_slang_count']}, "
            f"abbreviations={stats['abbreviation_count']}, "
            f"code_mix={stats['code_mix_count']})"
        )


# Convenience function
def expand_slang(text: str) -> str:
    """Quick slang expansion without creating instance."""
    expander = SlangExpander()
    return expander.expand(text)


if __name__ == "__main__":
    # Quick test
    expander = SlangExpander()
    
    test_texts = [
        "machaa naale lab ge bartiya?",
        "lol tbh idk what to say rn",
        "nee tumba irritating agthiya yaar",
        "submit madbeku assignment ivattu",
        "bro ur being sus ngl",
        "tumba sakkat waste fellow anta feel agutte",
    ]
    
    print(f"Slang Expander Stats: {expander.get_vocabulary_stats()}")
    print()
    
    for text in test_texts:
        print(f"Original:  {text}")
        print(f"Expanded:  {expander.expand(text)}")
        detected = expander.detect_slang(text)
        print(f"Detected:  {detected['total_count']} terms")
        print()
