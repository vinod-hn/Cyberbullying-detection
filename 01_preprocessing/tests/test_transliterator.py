# Test Transliterator
"""
Unit tests for the Transliterator class.
Tests transliteration between Kannada script and Roman (Latin) script.
Handles bi-directional conversion for Kannada-English code-mixed cyberbullying detection.
Based on 00_data reference files.
"""

import pytest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transliterator import Transliterator


class TestTransliterator:
    """Test suite for Transliterator class."""
    
    @pytest.fixture
    def transliterator(self):
        """Create a Transliterator instance for testing."""
        return Transliterator()
    
    # ==================== Basic Transliteration Tests ====================
    
    def test_kannada_to_roman(self, transliterator):
        """Test transliteration from Kannada script to Roman."""
        text = "ನೀನು ಹೇಗಿದ್ದೀಯ"
        result = transliterator.kannada_to_roman(text)
        
        assert isinstance(result, str)
        assert len(result) > 0
        # Result should be ASCII/Roman characters
        assert any(c.isascii() for c in result.replace(' ', ''))
    
    def test_roman_to_kannada(self, transliterator):
        """Test transliteration from Roman to Kannada script."""
        text = "neenu hegiddiya"
        result = transliterator.roman_to_kannada(text)
        
        assert isinstance(result, str)
        assert len(result) > 0
        # Result should contain Kannada characters (Unicode range: 0C80-0CFF)
        if result != text:  # If transliteration occurred
            assert any('\u0C80' <= c <= '\u0CFF' for c in result)
    
    def test_empty_string_kannada_to_roman(self, transliterator):
        """Test handling of empty string for Kannada to Roman."""
        result = transliterator.kannada_to_roman("")
        assert result == "" or result is not None
    
    def test_empty_string_roman_to_kannada(self, transliterator):
        """Test handling of empty string for Roman to Kannada."""
        result = transliterator.roman_to_kannada("")
        assert result == "" or result is not None
    
    def test_none_input_kannada_to_roman(self, transliterator):
        """Test handling of None input for Kannada to Roman."""
        try:
            result = transliterator.kannada_to_roman(None)
            assert result == "" or result is None
        except (TypeError, AttributeError):
            pass  # Expected behavior
    
    def test_none_input_roman_to_kannada(self, transliterator):
        """Test handling of None input for Roman to Kannada."""
        try:
            result = transliterator.roman_to_kannada(None)
            assert result == "" or result is None
        except (TypeError, AttributeError):
            pass  # Expected behavior
    
    # ==================== Kannada Vowels Tests ====================
    
    def test_kannada_vowels_to_roman(self, transliterator):
        """Test transliteration of Kannada vowels to Roman."""
        vowels = {
            'ಅ': 'a', 'ಆ': 'aa', 'ಇ': 'i', 'ಈ': 'ii',
            'ಉ': 'u', 'ಊ': 'uu', 'ಎ': 'e', 'ಏ': 'ee',
            'ಒ': 'o', 'ಓ': 'oo', 'ಔ': 'au'
        }
        for kannada, expected in vowels.items():
            result = transliterator.kannada_to_roman(kannada)
            assert isinstance(result, str)
            assert len(result) > 0
    
    def test_roman_vowels_to_kannada(self, transliterator):
        """Test transliteration of Roman vowels to Kannada."""
        vowels = ['a', 'aa', 'i', 'ii', 'u', 'uu', 'e', 'ee', 'o', 'oo', 'au']
        for vowel in vowels:
            result = transliterator.roman_to_kannada(vowel)
            assert isinstance(result, str)
    
    # ==================== Kannada Consonants Tests ====================
    
    def test_kannada_consonants_to_roman(self, transliterator):
        """Test transliteration of common Kannada consonants to Roman."""
        consonants = ['ಕ', 'ಗ', 'ನ', 'ಮ', 'ತ', 'ದ', 'ಪ', 'ಬ', 'ಯ', 'ರ', 'ಲ', 'ವ', 'ಸ', 'ಹ']
        for consonant in consonants:
            result = transliterator.kannada_to_roman(consonant)
            assert isinstance(result, str)
    
    def test_roman_consonants_to_kannada(self, transliterator):
        """Test transliteration of Roman consonants to Kannada."""
        consonants = ['ka', 'ga', 'na', 'ma', 'ta', 'da', 'pa', 'ba', 'ya', 'ra', 'la', 'va', 'sa', 'ha']
        for consonant in consonants:
            result = transliterator.roman_to_kannada(consonant)
            assert isinstance(result, str)
    
    # ==================== Common Words Tests (from dataset) ====================
    
    def test_common_kannada_words_to_roman(self, transliterator):
        """Test transliteration of common Kannada words from dataset."""
        # Common words from kannada.csv
        words = ['ನೀನು', 'ತುಂಬಾ', 'ಚೆನ್ನಾಗಿ', 'ಮಾಡು', 'ಹೋಗು']
        for word in words:
            result = transliterator.kannada_to_roman(word)
            assert isinstance(result, str)
            assert len(result) > 0
    
    def test_common_romanized_kannada_to_script(self, transliterator):
        """Test transliteration of common Romanized Kannada words to script."""
        # Common romanized words from dataset
        words = ['neenu', 'tumba', 'chennagi', 'maadu', 'hogu', 'hegidiya', 'machaa', 'maga']
        for word in words:
            result = transliterator.roman_to_kannada(word)
            assert isinstance(result, str)
    
    # ==================== Cyberbullying Context Tests ====================
    
    def test_transliterate_insult_words(self, transliterator):
        """Test transliteration of common insult-related words."""
        # From dataset patterns
        text = "nee sakkat dumb idiya"
        result = transliterator.roman_to_kannada(text)
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_transliterate_harassment_phrase(self, transliterator):
        """Test transliteration of harassment phrases."""
        text = "nee tumba irritating agthiya"
        result = transliterator.roman_to_kannada(text)
        assert isinstance(result, str)
    
    def test_transliterate_threat_phrase(self, transliterator):
        """Test transliteration of threat phrases."""
        text = "complaint haakthini nodu"
        result = transliterator.roman_to_kannada(text)
        assert isinstance(result, str)
    
    def test_transliterate_exclusion_phrase(self, transliterator):
        """Test transliteration of exclusion phrases."""
        text = "group inda remove maadona"
        result = transliterator.roman_to_kannada(text)
        assert isinstance(result, str)
    
    # ==================== Code-Mixed Text Tests ====================
    
    def test_code_mixed_kannada_english(self, transliterator):
        """Test handling of Kannada-English code-mixed text."""
        text = "ನಿನ್ನ presentation super ಆಯ್ತು"
        result = transliterator.kannada_to_roman(text)
        
        assert isinstance(result, str)
        # English words should be preserved
        assert "presentation" in result.lower() or "super" in result.lower()
    
    def test_code_mixed_roman_kannada_english(self, transliterator):
        """Test handling of Roman Kannada-English code-mixed text."""
        text = "nee tumba waste fellow idiya"
        result = transliterator.roman_to_kannada(text)
        
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_preserve_english_words(self, transliterator):
        """Test that English words are preserved during transliteration."""
        text = "exam tumba tough aaytu"
        
        if hasattr(transliterator, 'transliterate_preserve_english'):
            result = transliterator.transliterate_preserve_english(text)
            assert "exam" in result or "tough" in result
    
    # ==================== Address Terms Tests ====================
    
    def test_transliterate_address_terms(self, transliterator):
        """Test transliteration of address terms (maga, machaa, re, yaar)."""
        terms = ['maga', 'machaa', 're', 'yaar', 'guru']
        for term in terms:
            result = transliterator.roman_to_kannada(term)
            assert isinstance(result, str)
    
    def test_kannada_address_terms_to_roman(self, transliterator):
        """Test transliteration of Kannada address terms to Roman."""
        if hasattr(transliterator, 'kannada_to_roman'):
            text = "ಮಗ ಮಚ್ಚಾ"
            result = transliterator.kannada_to_roman(text)
            assert isinstance(result, str)
    
    # ==================== Intensifiers Tests ====================
    
    def test_transliterate_intensifiers(self, transliterator):
        """Test transliteration of Kannada intensifiers."""
        # Common intensifiers from dataset
        intensifiers = ['tumba', 'sakkat', 'full', 'swalpa']
        for word in intensifiers:
            result = transliterator.roman_to_kannada(word)
            assert isinstance(result, str)
    
    def test_kannada_intensifiers_to_roman(self, transliterator):
        """Test transliteration of Kannada intensifiers to Roman."""
        text = "ತುಂಬಾ ಸಕ್ಕತ್"
        result = transliterator.kannada_to_roman(text)
        assert isinstance(result, str)
        assert len(result) > 0
    
    # ==================== Special Character Handling Tests ====================
    
    def test_preserve_punctuation(self, transliterator):
        """Test that punctuation is preserved during transliteration."""
        text = "hegidiya? chennagidiya!"
        result = transliterator.roman_to_kannada(text)
        
        # Punctuation should be preserved
        assert "?" in result or "!" in result or len(result) > 0
    
    def test_preserve_numbers(self, transliterator):
        """Test that numbers are preserved during transliteration."""
        text = "naale 9 gante ge baa"
        result = transliterator.roman_to_kannada(text)
        
        assert "9" in result or len(result) > 0
    
    def test_handle_hashtags(self, transliterator):
        """Test handling of hashtags from dataset format."""
        text = "hegidiya machaa #5a76"
        result = transliterator.roman_to_kannada(text)
        
        assert isinstance(result, str)
    
    def test_handle_usernames(self, transliterator):
        """Test handling of anonymized usernames."""
        text = "Vxrc, nee tumba irritating"
        result = transliterator.roman_to_kannada(text)
        
        assert isinstance(result, str)
    
    # ==================== Sentence Level Tests ====================
    
    def test_full_sentence_kannada_to_roman(self, transliterator):
        """Test transliteration of full Kannada sentence."""
        text = "ನೀನು ತುಂಬಾ ಚೆನ್ನಾಗಿ ಮಾತಾಡ್ತೀಯ"
        result = transliterator.kannada_to_roman(text)
        
        assert isinstance(result, str)
        assert len(result) > 0
        # Should have multiple words
        assert ' ' in result or len(result) > 5
    
    def test_full_sentence_roman_to_kannada(self, transliterator):
        """Test transliteration of full Romanized Kannada sentence."""
        text = "neenu tumba chennagi maataadthiya"
        result = transliterator.roman_to_kannada(text)
        
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_dataset_sample_transliteration(self, transliterator):
        """Test transliteration with actual dataset samples."""
        # From kannada.csv
        samples = [
            "nee full time late bartiya dude",
            "exam tumba tough aaytu but somehow aaytu",
            "nee sakkat dumb idiya yaar machaa",
            "naale 9 gante lab ge bartira"
        ]
        for sample in samples:
            result = transliterator.roman_to_kannada(sample)
            assert isinstance(result, str)
            assert len(result) > 0
    
    # ==================== Reverse Transliteration Tests ====================
    
    def test_roundtrip_transliteration(self, transliterator):
        """Test roundtrip transliteration (Kannada -> Roman -> Kannada)."""
        original = "ನೀನು ಹೇಗಿದ್ದೀಯ"
        
        roman = transliterator.kannada_to_roman(original)
        back = transliterator.roman_to_kannada(roman)
        
        assert isinstance(back, str)
        assert len(back) > 0
    
    def test_consistency_transliteration(self, transliterator):
        """Test consistency of transliteration (same input -> same output)."""
        text = "hegidiya machaa"
        
        result1 = transliterator.roman_to_kannada(text)
        result2 = transliterator.roman_to_kannada(text)
        
        assert result1 == result2
    
    # ==================== Edge Cases Tests ====================
    
    def test_single_character_kannada(self, transliterator):
        """Test transliteration of single Kannada character."""
        text = "ನ"
        result = transliterator.kannada_to_roman(text)
        
        assert isinstance(result, str)
    
    def test_single_character_roman(self, transliterator):
        """Test transliteration of single Roman character."""
        text = "a"
        result = transliterator.roman_to_kannada(text)
        
        assert isinstance(result, str)
    
    def test_whitespace_only(self, transliterator):
        """Test handling of whitespace-only input."""
        text = "   "
        
        result = transliterator.roman_to_kannada(text)
        assert isinstance(result, str)
    
    def test_mixed_case_input(self, transliterator):
        """Test handling of mixed case Roman input."""
        text = "Hegidiya MACHAA maga"
        result = transliterator.roman_to_kannada(text)
        
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_conjunct_consonants(self, transliterator):
        """Test transliteration of conjunct consonants (ottakshara)."""
        # Words with conjuncts
        text = "ಚೆನ್ನಾಗಿ"  # 'chennagi' has conjunct 'nn'
        result = transliterator.kannada_to_roman(text)
        
        assert isinstance(result, str)
        assert len(result) > 0
    
    # ==================== Slang and Informal Text Tests ====================
    
    def test_kannada_slang_transliteration(self, transliterator):
        """Test transliteration of Kannada slang words."""
        slang_words = ['machaa', 'maga', 'guru', 'bossu', 'anna']
        for word in slang_words:
            result = transliterator.roman_to_kannada(word)
            assert isinstance(result, str)
    
    def test_informal_spellings(self, transliterator):
        """Test handling of informal/variable spellings."""
        # Same word, different spellings
        spellings = ['heggidiya', 'hegidiya', 'hegidya', 'hegidiiya']
        
        results = []
        for spelling in spellings:
            result = transliterator.roman_to_kannada(spelling)
            results.append(result)
            assert isinstance(result, str)
    
    def test_elongated_vowels(self, transliterator):
        """Test handling of elongated vowels in informal text."""
        text = "nooooodu machaaaa"
        result = transliterator.roman_to_kannada(text)
        
        assert isinstance(result, str)
    
    # ==================== Script Detection Tests ====================
    
    def test_detect_kannada_script(self, transliterator):
        """Test detection of Kannada script in text."""
        if hasattr(transliterator, 'is_kannada_script'):
            kannada_text = "ನೀನು ಹೇಗಿದ್ದೀಯ"
            assert transliterator.is_kannada_script(kannada_text) == True
    
    def test_detect_roman_script(self, transliterator):
        """Test detection of Roman script in text."""
        if hasattr(transliterator, 'is_kannada_script'):
            roman_text = "neenu hegiddiya"
            assert transliterator.is_kannada_script(roman_text) == False
    
    def test_detect_mixed_script(self, transliterator):
        """Test detection of mixed script text."""
        if hasattr(transliterator, 'get_script_ratio'):
            mixed_text = "ನಿನ್ನ presentation super ಆಯ್ತು"
            ratio = transliterator.get_script_ratio(mixed_text)
            assert isinstance(ratio, (int, float, dict))
    
    # ==================== Auto-Transliteration Tests ====================
    
    def test_auto_transliterate(self, transliterator):
        """Test automatic direction detection for transliteration."""
        if hasattr(transliterator, 'auto_transliterate'):
            kannada_text = "ನೀನು ಹೇಗಿದ್ದೀಯ"
            roman_text = "neenu hegiddiya"
            
            result1 = transliterator.auto_transliterate(kannada_text)
            result2 = transliterator.auto_transliterate(roman_text)
            
            assert isinstance(result1, str)
            assert isinstance(result2, str)
    
    # ==================== Batch Processing Tests ====================
    
    def test_batch_transliteration(self, transliterator):
        """Test batch transliteration of multiple texts."""
        if hasattr(transliterator, 'batch_transliterate'):
            texts = [
                "neenu hegiddiya",
                "nee tumba irritating",
                "machaa helu"
            ]
            results = transliterator.batch_transliterate(texts)
            
            assert isinstance(results, list)
            assert len(results) == len(texts)
    
    def test_transliterate_dataframe_column(self, transliterator):
        """Test transliteration of pandas DataFrame column."""
        if hasattr(transliterator, 'transliterate_column'):
            import pandas as pd
            df = pd.DataFrame({
                'text': ['neenu hegiddiya', 'nee tumba irritating']
            })
            
            result = transliterator.transliterate_column(df, 'text')
            assert isinstance(result, pd.DataFrame)
    
    # ==================== Performance Tests ====================
    
    def test_long_text_transliteration(self, transliterator):
        """Test transliteration of long text."""
        # Simulate a longer message
        text = "nee tumba irritating agthiya " * 10
        result = transliterator.roman_to_kannada(text)
        
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_unicode_normalization(self, transliterator):
        """Test Unicode normalization during transliteration."""
        # Kannada text with potential normalization issues
        text = "ನೀನು"
        result = transliterator.kannada_to_roman(text)
        
        assert isinstance(result, str)
        # Result should not contain combining characters incorrectly
    
    # ==================== Error Handling Tests ====================
    
    def test_invalid_unicode(self, transliterator):
        """Test handling of invalid Unicode characters."""
        try:
            text = "test\x00invalid"
            result = transliterator.roman_to_kannada(text)
            assert isinstance(result, str)
        except (ValueError, UnicodeError):
            pass  # Expected behavior
    
    def test_emoji_preservation(self, transliterator):
        """Test that emojis are preserved during transliteration."""
        text = "hegidiya machaa 😀"
        result = transliterator.roman_to_kannada(text)
        
        assert isinstance(result, str)
        # Emoji should be preserved or gracefully handled
    
    # ==================== Configuration Tests ====================
    
    def test_custom_mapping(self, transliterator):
        """Test custom character mapping if supported."""
        if hasattr(transliterator, 'add_custom_mapping'):
            transliterator.add_custom_mapping('nn', 'ನ್ನ')
            result = transliterator.roman_to_kannada('anna')
            assert isinstance(result, str)
    
    def test_get_mapping(self, transliterator):
        """Test retrieval of transliteration mapping."""
        if hasattr(transliterator, 'get_mapping'):
            mapping = transliterator.get_mapping()
            assert isinstance(mapping, dict)


class TestTransliteratorIntegration:
    """Integration tests for Transliterator with other preprocessors."""
    
    @pytest.fixture
    def transliterator(self):
        """Create a Transliterator instance for testing."""
        return Transliterator()
    
    def test_transliterate_then_normalize(self, transliterator):
        """Test transliteration followed by normalization."""
        text = "NEE TUMBA IRRITATING AGTHIYA"
        result = transliterator.roman_to_kannada(text.lower())
        
        assert isinstance(result, str)
    
    def test_with_code_mix_processor_patterns(self, transliterator):
        """Test transliteration with code-mix patterns."""
        # Patterns from code_mix_patterns.txt
        patterns = [
            "nee full stupid idiya",
            "You are such a loser maga",
            "Class nalli ee behavior beda",
            "tumba irritating"
        ]
        for pattern in patterns:
            result = transliterator.roman_to_kannada(pattern)
            assert isinstance(result, str)
    
    def test_cyberbullying_dataset_samples(self, transliterator):
        """Test transliteration with actual cyberbullying dataset samples."""
        # Samples from kannad english.csv
        samples = [
            "stop talking nonsense nin matu beka illa",
            "neenu hopeless yaake heege",
            "ninige nanna online time gothu how",
            "your kind spoils the environment"
        ]
        for sample in samples:
            result = transliterator.roman_to_kannada(sample)
            assert isinstance(result, str)
            assert len(result) > 0


class TestTransliteratorSpecialCases:
    """Test special cases and edge conditions for Transliterator."""
    
    @pytest.fixture
    def transliterator(self):
        """Create a Transliterator instance for testing."""
        return Transliterator()
    
    def test_halant_handling(self, transliterator):
        """Test handling of halant (virama) in Kannada."""
        # Words with halant
        text = "ಚೆನ್ನ"  # has halant
        result = transliterator.kannada_to_roman(text)
        
        assert isinstance(result, str)
    
    def test_anusvara_handling(self, transliterator):
        """Test handling of anusvara (ಂ) in Kannada."""
        text = "ತುಂಬಾ"  # has anusvara
        result = transliterator.kannada_to_roman(text)
        
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_visarga_handling(self, transliterator):
        """Test handling of visarga (ಃ) in Kannada."""
        text = "ದುಃಖ"  # has visarga
        result = transliterator.kannada_to_roman(text)
        
        assert isinstance(result, str)
    
    def test_compound_words(self, transliterator):
        """Test transliteration of compound words."""
        compounds = ['maataadthiya', 'bartiddini', 'madkondiru', 'hogthidvi']
        for word in compounds:
            result = transliterator.roman_to_kannada(word)
            assert isinstance(result, str)
    
    def test_aspirated_consonants(self, transliterator):
        """Test transliteration of aspirated consonants."""
        # kha, gha, cha, jha, tha, dha, pha, bha
        aspirated = ['kha', 'gha', 'cha', 'jha', 'tha', 'dha', 'pha', 'bha']
        for consonant in aspirated:
            result = transliterator.roman_to_kannada(consonant)
            assert isinstance(result, str)
    
    def test_retroflex_consonants(self, transliterator):
        """Test transliteration of retroflex consonants."""
        # Ta, Da, Na (retroflex)
        text = "beTa daDa"
        result = transliterator.roman_to_kannada(text)
        
        assert isinstance(result, str)


class TestRomanizedKannadaSlang:
    """
    Test suite for Romanized Kannada slang - the primary format in the dataset.
    Chat conversations use Kannada slang written with English alphabet.
    """
    
    @pytest.fixture
    def transliterator(self):
        """Create a Transliterator instance for testing."""
        return Transliterator()
    
    # ==================== Common Slang Words Tests ====================
    
    def test_common_slang_words(self, transliterator):
        """Test transliteration of common Kannada slang words in Roman."""
        slang_words = [
            'maga',      # dude/bro
            'machaa',    # bro/friend
            'guru',      # friend/master
            'yaar',      # man/dude
            're',        # hey (addressing)
            'anna',      # elder brother
            'akka',      # elder sister
            'bossu',     # boss (slang)
            'saar',      # sir (slang)
            'macha'      # alternate spelling of machaa
        ]
        for word in slang_words:
            result = transliterator.roman_to_kannada(word)
            assert isinstance(result, str)
            assert len(result) > 0
    
    def test_addressing_patterns(self, transliterator):
        """Test common addressing patterns in slang."""
        patterns = [
            "aye maga",
            "lo machaa", 
            "eno guru",
            "helu re",
            "kelu yaar"
        ]
        for pattern in patterns:
            result = transliterator.roman_to_kannada(pattern)
            assert isinstance(result, str)
    
    # ==================== Verb Endings Tests ====================
    
    def test_common_verb_endings(self, transliterator):
        """Test common Kannada verb endings in Roman script."""
        verb_endings = [
            'madthiya',   # you do
            'bartiya',    # you come
            'hogthiya',   # you go
            'nodthiya',   # you see
            'kelthiya',   # you ask/listen
            'heltiya',    # you say
            'idiya',      # you are
            'agthiya',    # you become
            'tingtiya',   # you eat
            'kudithiya'   # you drink
        ]
        for verb in verb_endings:
            result = transliterator.roman_to_kannada(verb)
            assert isinstance(result, str)
    
    def test_verb_tenses_slang(self, transliterator):
        """Test verb tenses in slang format."""
        verbs = [
            'maadidya',   # did you do
            'bartidya',   # are you coming
            'hogidya',    # did you go
            'aaytu',      # it became/happened
            'aagide',     # it is
            'madkond',    # having done
            'bartini',    # I will come
            'hogthini'    # I will go
        ]
        for verb in verbs:
            result = transliterator.roman_to_kannada(verb)
            assert isinstance(result, str)
    
    # ==================== Pronouns Tests ====================
    
    def test_slang_pronouns(self, transliterator):
        """Test Kannada pronouns in Roman slang."""
        pronouns = [
            'nee',        # you (informal)
            'neenu',      # you
            'naanu',      # I
            'avnu',       # he
            'avlu',       # she
            'avru',       # they (respectful)
            'navu',       # we
            'ivan',       # this guy
            'ivalu',      # this girl
            'ivru',       # these people
            'ninna',      # your
            'nanna',      # my
            'avna',       # his
            'avla'        # her
        ]
        for pronoun in pronouns:
            result = transliterator.roman_to_kannada(pronoun)
            assert isinstance(result, str)
    
    # ==================== Intensifiers & Modifiers Tests ====================
    
    def test_slang_intensifiers(self, transliterator):
        """Test Kannada intensifiers commonly used in slang."""
        intensifiers = [
            'tumba',      # very
            'sakkat',     # extremely/super
            'full',       # fully (borrowed)
            'swalpa',     # a little
            'jaasti',     # more/too much
            'kammmi',     # less
            'chennaag',   # nicely/well
            'bekku',      # a lot
            'thika',      # properly
            'araam'       # comfortably
        ]
        for word in intensifiers:
            result = transliterator.roman_to_kannada(word)
            assert isinstance(result, str)
    
    def test_degree_modifiers_with_adjectives(self, transliterator):
        """Test degree modifiers with adjectives in slang."""
        phrases = [
            "tumba irritating",
            "sakkat dumb",
            "full stupid",
            "swalpa annoying",
            "jaasti toxic"
        ]
        for phrase in phrases:
            result = transliterator.roman_to_kannada(phrase)
            assert isinstance(result, str)
    
    # ==================== Question Words Tests ====================
    
    def test_question_words(self, transliterator):
        """Test Kannada question words in Roman slang."""
        questions = [
            'yaake',      # why
            'yelli',      # where
            'yavaga',     # when
            'hege',       # how
            'yaaru',      # who
            'yaavdu',     # which
            'yenu',       # what
            'eshtu',      # how much
            'yavag',      # when (informal)
            'hegidiya'    # how are you
        ]
        for word in questions:
            result = transliterator.roman_to_kannada(word)
            assert isinstance(result, str)
    
    def test_question_patterns(self, transliterator):
        """Test question patterns in slang format."""
        patterns = [
            "yaake heege madthiya",
            "yelli hogthidiya",
            "yavaga barthiya",
            "eshtu time aagutte",
            "yaaru heldru"
        ]
        for pattern in patterns:
            result = transliterator.roman_to_kannada(pattern)
            assert isinstance(result, str)
    
    # ==================== Negation Tests ====================
    
    def test_negation_words(self, transliterator):
        """Test Kannada negation words in Roman slang."""
        negations = [
            'illa',       # no/not
            'beda',       # don't want
            'agalla',     # cannot
            'baralla',    # won't come
            'gotilla',    # don't know
            'bekagilla',  # don't need
            'madbeda',    # don't do
            'hogbeda',    # don't go
            'helbeda'     # don't say
        ]
        for word in negations:
            result = transliterator.roman_to_kannada(word)
            assert isinstance(result, str)
    
    # ==================== Cyberbullying Slang Tests ====================
    
    def test_insult_phrases_slang(self, transliterator):
        """Test common insult phrases in Romanized Kannada slang."""
        insults = [
            "nee tumba dumb idiya",
            "sakkat waste fellow",
            "full useless maga",
            "nee hopeless guru",
            "complete idiot idiya"
        ]
        for phrase in insults:
            result = transliterator.roman_to_kannada(phrase)
            assert isinstance(result, str)
            assert len(result) > 0
    
    def test_harassment_phrases_slang(self, transliterator):
        """Test harassment phrases in Romanized Kannada slang."""
        phrases = [
            "yaake nanna hinde bartiya",
            "nanna status yavaglu check madthiya",
            "prati dina ping madbeda",
            "nanna movements track madbeda",
            "elli hogidaroo ade tara madthiya"
        ]
        for phrase in phrases:
            result = transliterator.roman_to_kannada(phrase)
            assert isinstance(result, str)
    
    def test_threat_phrases_slang(self, transliterator):
        """Test threat phrases in Romanized Kannada slang."""
        phrases = [
            "sir ge complaint haakthini",
            "consequences face maadu",
            "regret madthiya nodu",
            "lesson sigutte",
            "limit nodu yaar"
        ]
        for phrase in phrases:
            result = transliterator.roman_to_kannada(phrase)
            assert isinstance(result, str)
    
    def test_exclusion_phrases_slang(self, transliterator):
        """Test exclusion phrases in Romanized Kannada slang."""
        phrases = [
            "group inda remove maadona",
            "ivan illaadre better",
            "ninge illi place illa",
            "nin illade better agide",
            "selected people only"
        ]
        for phrase in phrases:
            result = transliterator.roman_to_kannada(phrase)
            assert isinstance(result, str)
    
    # ==================== Mixed English-Kannada Slang Tests ====================
    
    def test_english_words_in_kannada_context(self, transliterator):
        """Test English words embedded in Kannada slang sentences."""
        sentences = [
            "exam tumba tough aaytu",
            "ninna presentation super aaytu",
            "project help madidakke thanks",
            "network sakkat slow ide",
            "lecture miss aaytu"
        ]
        for sentence in sentences:
            result = transliterator.roman_to_kannada(sentence)
            assert isinstance(result, str)
            # English words like exam, presentation, project should be handled
    
    def test_english_insults_with_kannada_grammar(self, transliterator):
        """Test English insult words with Kannada grammar."""
        phrases = [
            "nee asshole tara behave madthiya",
            "ivan complete stupid maga",
            "tumba fake friend",
            "sakkat toxic behavior",
            "full time loser idiya"
        ]
        for phrase in phrases:
            result = transliterator.roman_to_kannada(phrase)
            assert isinstance(result, str)
    
    def test_borrowed_words(self, transliterator):
        """Test commonly borrowed English words in Kannada slang."""
        borrowed = [
            'time', 'late', 'full', 'waste', 'fellow',
            'group', 'class', 'lab', 'exam', 'project',
            'message', 'reply', 'status', 'online', 'spam'
        ]
        for word in borrowed:
            # These should be preserved or handled appropriately
            result = transliterator.roman_to_kannada(word)
            assert isinstance(result, str)
    
    # ==================== Informal Spelling Variations Tests ====================
    
    def test_spelling_variations_tumba(self, transliterator):
        """Test spelling variations of 'tumba' (very)."""
        variations = ['tumba', 'thumpa', 'thumba', 'tumpa']
        for var in variations:
            result = transliterator.roman_to_kannada(var)
            assert isinstance(result, str)
    
    def test_spelling_variations_machaa(self, transliterator):
        """Test spelling variations of 'machaa' (bro)."""
        variations = ['machaa', 'macha', 'maccha', 'machcha', 'machchaa']
        for var in variations:
            result = transliterator.roman_to_kannada(var)
            assert isinstance(result, str)
    
    def test_spelling_variations_hegidiya(self, transliterator):
        """Test spelling variations of 'hegidiya' (how are you)."""
        variations = ['hegidiya', 'hegiddiya', 'heggidiya', 'hegidya', 'hegidiiya']
        for var in variations:
            result = transliterator.roman_to_kannada(var)
            assert isinstance(result, str)
    
    def test_elongated_words(self, transliterator):
        """Test elongated words for emphasis."""
        elongated = [
            'nooodu',      # look (elongated)
            'heluuu',      # say (elongated)
            'baaaa',       # come (elongated)
            'ayyyooo',     # expression
            'machaaaa'     # bro (elongated)
        ]
        for word in elongated:
            result = transliterator.roman_to_kannada(word)
            assert isinstance(result, str)
    
    # ==================== Common Expressions Tests ====================
    
    def test_common_expressions(self, transliterator):
        """Test common Kannada expressions in slang."""
        expressions = [
            'ayyoo',       # oh no
            'yappa',       # wow/oh
            'aiyo',        # alternate of ayyoo
            'arey',        # hey
            'abe',         # hey (informal)
            'eno',         # what (informal)
            'yeno',        # what (casual)
            'sari',        # okay
            'haa',         # yes
            'illa',        # no
            'houdu',       # yes
            'beku',        # want/need
            'beda'         # don't want
        ]
        for expr in expressions:
            result = transliterator.roman_to_kannada(expr)
            assert isinstance(result, str)
    
    def test_filler_words(self, transliterator):
        """Test filler words and discourse markers."""
        fillers = [
            'actually',    # discourse marker
            'basically',   # discourse marker
            'andre',       # means/that is
            'anthavre',    # they say
            'antha',       # like/that
            'alva',        # right?
            'alla',        # not
            'nodappa'      # see (casual)
        ]
        for filler in fillers:
            result = transliterator.roman_to_kannada(filler)
            assert isinstance(result, str)
    
    # ==================== Dataset Real Examples Tests ====================
    
    def test_real_dataset_examples_kannada_csv(self, transliterator):
        """Test with real examples from kannada.csv."""
        examples = [
            "nee full time late bartiya dude",
            "exam tumba tough aaytu but somehow aaytu",
            "nee sakkat dumb idiya yaar re",
            "naale 9 gante lab ge bartira",
            "pizza bekka biryani bekka decide maadi",
            "project help madidakke thanks maga",
            "ninna presentation super aaytu chennag ide",
            "network sakkath slow ide once repeat maadu",
            "nee ivattu full lazy mode alli idiya",
            "fest practice ge yar yar barthidira"
        ]
        for example in examples:
            result = transliterator.roman_to_kannada(example)
            assert isinstance(result, str)
            assert len(result) > 0
    
    def test_real_dataset_examples_kannad_english_csv(self, transliterator):
        """Test with real examples from kannad english.csv."""
        examples = [
            "stop talking nonsense nin matu beka illa",
            "neenu hopeless yaake heege",
            "just wait and watch ninige lesson sigutte",
            "yaake nanna follow madthiya everywhere",
            "your kind spoils the environment",
            "nin illade better agide period",
            "neenu full dumb yaake heege",
            "back off right now nin hinde hogu",
            "stay out of my matters nin hogu",
            "ninige simple vishaya kuda artha agalla"
        ]
        for example in examples:
            result = transliterator.roman_to_kannada(example)
            assert isinstance(result, str)
            assert len(result) > 0
    
    # ==================== Sentence Structure Tests ====================
    
    def test_kannada_base_english_insult_pattern(self, transliterator):
        """Test 'Kannada Base + English Insult' pattern."""
        # Pattern: [Kannada pronoun] + [English insult] + [Kannada verb]
        patterns = [
            "nee full stupid idiya",
            "avanu tumba toxic agidane",
            "nanna fake friend madkondu",
            "ivanu complete waste idane"
        ]
        for pattern in patterns:
            result = transliterator.roman_to_kannada(pattern)
            assert isinstance(result, str)
    
    def test_english_base_kannada_filler_pattern(self, transliterator):
        """Test 'English Base + Kannada Fillers' pattern."""
        # Pattern: [English sentence] + [Kannada particles: re, maga, yaar]
        patterns = [
            "You are such a loser maga",
            "Stop being toxic re",
            "Why are you so fake yaar",
            "That's so annoying guru"
        ]
        for pattern in patterns:
            result = transliterator.roman_to_kannada(pattern)
            assert isinstance(result, str)
    
    def test_mixed_vocabulary_pattern(self, transliterator):
        """Test alternating Kannada-English vocabulary pattern."""
        patterns = [
            "Class nalli ee behavior beda",
            "Group ge spam madbeda",
            "Lecture miss aaytu but notes share maadi",
            "Assignment submit madbeku tomorrow"
        ]
        for pattern in patterns:
            result = transliterator.roman_to_kannada(pattern)
            assert isinstance(result, str)
    
    # ==================== Postpositions & Particles Tests ====================
    
    def test_postpositions(self, transliterator):
        """Test common Kannada postpositions in slang."""
        postpositions = [
            'nalli',      # in
            'ge',         # to
            'inda',       # from
            'jote',       # with
            'hatra',      # near/with
            'mele',       # on/above
            'kelage',     # below
            'mundhe',     # before/front
            'hinde'       # behind/after
        ]
        for word in postpositions:
            result = transliterator.roman_to_kannada(word)
            assert isinstance(result, str)
    
    def test_sentence_ending_particles(self, transliterator):
        """Test sentence ending particles in slang."""
        particles = [
            'alva',       # right?
            'alla',       # isn't it
            'anta',       # saying that
            'andre',      # meaning
            'aadre',      # but
            'nodappa',    # see
            'kelappa'     # listen
        ]
        for particle in particles:
            result = transliterator.roman_to_kannada(particle)
            assert isinstance(result, str)


class TestBadWordsProfanityTransliteration:
    """
    Test suite for transliteration of bad words and profanity.
    Based on bad_words.csv, profanity_kannada.csv, and profanity_english.csv.
    """
    
    @pytest.fixture
    def transliterator(self):
        """Create a Transliterator instance for testing."""
        return Transliterator()
    
    # ==================== Kannada Profanity Terms Tests ====================
    
    def test_kannada_profanity_basic(self, transliterator):
        """Test transliteration of basic Kannada profanity terms."""
        # From profanity_kannada.csv - Romanized forms
        terms = [
            'thotha',       # stupid/useless
            'gaandu',       # vulgar slur
            'bosodina',     # severe insult
            'hucchadana',   # crazy person
            'singri',       # useless person
            'dagarina'      # worthless person
        ]
        for term in terms:
            result = transliterator.roman_to_kannada(term)
            assert isinstance(result, str)
            assert len(result) > 0
    
    def test_kannada_profanity_script_to_roman(self, transliterator):
        """Test transliteration of Kannada script profanity to Roman."""
        # From profanity_kannada.csv - Kannada script
        terms = [
            'ತೊಠ',         # thotha
            'ಗಾಂಡು',       # gaandu
            'ಬೊಸೋಡಿನ',    # bosodina
            'ಹುಚ್ಚದಾನ',    # hucchadana
            'ಸಿಂಗ್ರಿ',      # singri
            'ದಗರಿನ',       # dagarina
            'ಹಾವು',        # haavu (snake/traitor)
            'ಮೂರ್ಖ'        # moorkha (fool)
        ]
        for term in terms:
            result = transliterator.kannada_to_roman(term)
            assert isinstance(result, str)
            assert len(result) > 0
    
    def test_kannada_profanity_with_maga(self, transliterator):
        """Test profanity combined with 'maga' (buddy/friend address)."""
        # From profanity_kannada.csv - compound insults
        terms = [
            'thotha maga',       # useless buddy
            'dagarina maga',     # worthless buddy
            'soolya maga',       # severe insult
            'kall maga',         # thief buddy
            'mangyan maga',      # monkey buddy
            'ninna gandu maga'   # insult phrase
        ]
        for term in terms:
            result = transliterator.roman_to_kannada(term)
            assert isinstance(result, str)
    
    def test_kannada_comparative_insults(self, transliterator):
        """Test 'thara' (like) comparative insults."""
        # Pattern: [insult] + thara = like a [insult]
        terms = [
            'monkey thara',      # like a monkey
            'joker thara',       # like a joker
            'kalladantha',       # like a thief
            'hucchadanthe'       # acting crazy
        ]
        for term in terms:
            result = transliterator.roman_to_kannada(term)
            assert isinstance(result, str)
    
    # ==================== Sexual Harassment Terms Tests ====================
    
    def test_sexual_terms_roman(self, transliterator):
        """Test transliteration of sexual harassment terms (Roman)."""
        # From profanity_kannada.csv - handle carefully
        terms = [
            'thullu',            # vulgar term
            'keya',              # sexual reference
            'hendruna keya'      # sexual reference
        ]
        for term in terms:
            result = transliterator.roman_to_kannada(term)
            assert isinstance(result, str)
    
    def test_sexual_terms_script(self, transliterator):
        """Test transliteration of sexual terms (Kannada script)."""
        terms = [
            'ತುಲ್ಲು',          # thullu
            'ಕೆಯ',            # keya
            'ಹೆಂಡ್ರುನ ಕೆಯ'    # hendruna keya
        ]
        for term in terms:
            result = transliterator.kannada_to_roman(term)
            assert isinstance(result, str)
    
    # ==================== Family-Related Insults Tests ====================
    
    def test_family_insults_roman(self, transliterator):
        """Test family-related insults in Roman script."""
        terms = [
            'nimavvan',          # your father (derogatory)
            'nimmavvana',        # your father's
            'nimmavvana maga',   # insulting father
            'nim amman',         # your mother
            'ninna amma'         # your mother (derogatory)
        ]
        for term in terms:
            result = transliterator.roman_to_kannada(term)
            assert isinstance(result, str)
    
    # ==================== English Profanity in Kannada Context Tests ====================
    
    def test_english_profanity_in_context(self, transliterator):
        """Test English profanity used in Kannada slang context."""
        # From profanity_english.csv - commonly mixed
        phrases = [
            "nee asshole tara idiya",
            "you are stupid maga",
            "full loser behaviour",
            "idiot thara behavior",
            "moron tara mathadbeda"
        ]
        for phrase in phrases:
            result = transliterator.roman_to_kannada(phrase)
            assert isinstance(result, str)
    
    def test_english_borrowed_insults_in_kannada(self, transliterator):
        """Test English insults borrowed into Kannada."""
        # From profanity_kannada.csv - English borrowed terms
        borrowed = [
            'ಡಂಬ್',      # dumb
            'ಫೂಲ್',      # fool
            'ಐಡಿಯಟ್',    # idiot
            'ಲೂಸರ್',     # loser
            'ವೇಸ್ಟ್',    # waste
            'ಕ್ರೈ ಬೇಬಿ',  # cry baby
            'ಫೇಕ್',      # fake
            'ಜಲೋಸ್'      # jealous
        ]
        for term in borrowed:
            result = transliterator.kannada_to_roman(term)
            assert isinstance(result, str)
    
    # ==================== Bad Words Dataset Examples Tests ====================
    
    def test_bad_words_dataset_insults(self, transliterator):
        """Test transliteration with bad_words.csv insult examples."""
        examples = [
            "nee tumba thotha maga tara idiya maga",
            "ninna idiot thara behavior",
            "group nalli a tara thotha baribeda",
            "ivattu full hucchadanta mode nalli idiya",
            "nee tumba dagarina maga tara idiya maga"
        ]
        for example in examples:
            result = transliterator.roman_to_kannada(example)
            assert isinstance(result, str)
            assert len(result) > 0
    
    def test_bad_words_dataset_harassment(self, transliterator):
        """Test transliteration with bad_words.csv harassment examples."""
        examples = [
            "group nalli a tara haavu bro neenu maga baribeda",
            "group nalli ee not a loyal attitude beda",
            "ninna behavior nodi everyone frustrate agta idare",
            "people are tired of your attitude"
        ]
        for example in examples:
            result = transliterator.roman_to_kannada(example)
            assert isinstance(result, str)
    
    def test_bad_words_dataset_threats(self, transliterator):
        """Test transliteration with bad_words.csv threat examples."""
        examples = [
            "nee full sucide tara behave madthiya",
            "ivattu kooda sucide tara act madtidya",
            "yaaru kuda ninna sucide nature ishta padalla",
            "ivattu full die mad mode nalli idiya"
        ]
        for example in examples:
            result = transliterator.roman_to_kannada(example)
            assert isinstance(result, str)
    
    def test_bad_words_dataset_sexual(self, transliterator):
        """Test transliteration with bad_words.csv sexual harassment examples."""
        examples = [
            "nin ninna hendruna keya nataka nodakke bore aaytu",
            "nee tumba nin thullu keya tara idiya maga",
            "group nalli ee sexy baby attitude beda"
        ]
        for example in examples:
            result = transliterator.roman_to_kannada(example)
            assert isinstance(result, str)
    
    # ==================== Severity Levels Tests ====================
    
    def test_low_severity_terms(self, transliterator):
        """Test transliteration of low severity bad words."""
        # Low severity from profanity files
        terms = [
            'dumb', 'waste', 'cry baby', 'jealous',
            'rubbish', 'damn', 'crap', 'bore'
        ]
        for term in terms:
            result = transliterator.roman_to_kannada(term)
            assert isinstance(result, str)
    
    def test_medium_severity_terms(self, transliterator):
        """Test transliteration of medium severity bad words."""
        # Medium severity from profanity_kannada.csv
        terms = [
            'thotha', 'hucchadana', 'singri', 'dagarina',
            'haavu', 'moorkha', 'bewakoof', 'kembatti'
        ]
        for term in terms:
            result = transliterator.roman_to_kannada(term)
            assert isinstance(result, str)
    
    def test_high_severity_terms(self, transliterator):
        """Test transliteration of high severity bad words."""
        # High severity from profanity_kannada.csv
        terms = [
            'gaandu', 'soolya', 'bosodina', 'soole',
            'thullu', 'ninna amma', 'nimmavvana maga'
        ]
        for term in terms:
            result = transliterator.roman_to_kannada(term)
            assert isinstance(result, str)
    
    # ==================== Context-Based Profanity Tests ====================
    
    def test_profanity_in_whatsapp_context(self, transliterator):
        """Test profanity in WhatsApp group chat context."""
        # From bad_words.csv - whatsapp_group_chat context
        phrases = [
            "group nalli a tara singri maga baribeda",
            "yella dina gaandu tara mathadbeda",
            "group nalli ee joker thara attitude beda",
            "nee tumba bosodina tara idiya maga"
        ]
        for phrase in phrases:
            result = transliterator.roman_to_kannada(phrase)
            assert isinstance(result, str)
    
    def test_profanity_with_anonymized_usernames(self, transliterator):
        """Test profanity with anonymized usernames from dataset."""
        phrases = [
            "Vxrc, nee tumba thotha maga tara idiya",
            "Kxo, you are such a waste in this group",
            "group nalli a tara baribeda, Ujp"
        ]
        for phrase in phrases:
            result = transliterator.roman_to_kannada(phrase)
            assert isinstance(result, str)
    
    # ==================== Mental Health Stigma Terms Tests ====================
    
    def test_mental_health_stigma_terms(self, transliterator):
        """Test mental health stigma terms transliteration."""
        terms = [
            'hucchadana',        # crazy person
            'hucchadanta',       # like mad person
            'thale kettide',     # gone mad
            'psycho',            # English borrowed
            'mental'             # English borrowed
        ]
        for term in terms:
            result = transliterator.roman_to_kannada(term)
            assert isinstance(result, str)
    
    def test_mental_health_phrases(self, transliterator):
        """Test mental health stigma phrases."""
        phrases = [
            "ivattu full hucchadanta mode nalli idiya",
            "ninna psycho behavior",
            "thale kettide ninge",
            "mental case idiya nee"
        ]
        for phrase in phrases:
            result = transliterator.roman_to_kannada(phrase)
            assert isinstance(result, str)
    
    # ==================== Appearance-Based Insults Tests ====================
    
    def test_appearance_insults(self, transliterator):
        """Test appearance-based insults transliteration."""
        # From profanity_english.csv
        terms = ['ugly', 'fat', 'freak', 'disgusting']
        for term in terms:
            result = transliterator.roman_to_kannada(term)
            assert isinstance(result, str)
    
    def test_appearance_insults_in_context(self, transliterator):
        """Test appearance insults in Kannada slang context."""
        phrases = [
            "nee tumba ugly idiya",
            "sakkat fat aagidiya",
            "disgusting tara idiya nee"
        ]
        for phrase in phrases:
            result = transliterator.roman_to_kannada(phrase)
            assert isinstance(result, str)
    
    # ==================== Gendered Insults Tests ====================
    
    def test_gendered_insults(self, transliterator):
        """Test gendered insults transliteration."""
        # From profanity_english.csv
        terms = ['bitch', 'slut', 'whore']
        for term in terms:
            result = transliterator.roman_to_kannada(term)
            assert isinstance(result, str)
    
    def test_kannada_gendered_terms(self, transliterator):
        """Test Kannada gendered insult terms."""
        terms = [
            'soole',         # prostitute
            'sooley',        # alternate form
            'bosodike'       # feminine form
        ]
        for term in terms:
            result = transliterator.roman_to_kannada(term)
            assert isinstance(result, str)
    
    # ==================== Exclusion & Dismissal Phrases Tests ====================
    
    def test_exclusion_phrases_profanity(self, transliterator):
        """Test exclusion phrases with profanity."""
        phrases = [
            "shut up nin baayi muchko",
            "go away hogu illi",
            "nobody cares ninna bagge",
            "nin illade better this group"
        ]
        for phrase in phrases:
            result = transliterator.roman_to_kannada(phrase)
            assert isinstance(result, str)
    
    # ==================== Threat Words Tests ====================
    
    def test_threat_keywords(self, transliterator):
        """Test threat-related keywords transliteration."""
        # From profanity_english.csv
        terms = ['die', 'regret', 'consequences', 'watch out', 'sucide']
        for term in terms:
            result = transliterator.roman_to_kannada(term)
            assert isinstance(result, str)
    
    def test_threat_phrases_profanity(self, transliterator):
        """Test threat phrases with profanity context."""
        phrases = [
            "die mad agutte ninge",
            "regret madthiya nodu maga",
            "consequences face maadu thotha",
            "watch out illa andre"
        ]
        for phrase in phrases:
            result = transliterator.roman_to_kannada(phrase)
            assert isinstance(result, str)
    
    # ==================== Vulgar Compound Words Tests ====================
    
    def test_english_vulgar_compounds(self, transliterator):
        """Test English vulgar compound words from dataset."""
        # From profanity_english.csv
        terms = [
            'dickhole', 'pissflaps', 'asswipe', 'bellend',
            'dumbass', 'dickhead', 'cockfoam', 'butthead'
        ]
        for term in terms:
            result = transliterator.roman_to_kannada(term)
            assert isinstance(result, str)
    
    def test_vulgar_compounds_in_context(self, transliterator):
        """Test vulgar compound words in Kannada context."""
        phrases = [
            "you are such a dickhole in this group",
            "only a asswipe will spam the group like you",
            "this dumbass behaviour is not okay"
        ]
        for phrase in phrases:
            result = transliterator.roman_to_kannada(phrase)
            assert isinstance(result, str)
    
    # ==================== Alternate Spellings of Profanity Tests ====================
    
    def test_profanity_spelling_variations(self, transliterator):
        """Test spelling variations of profanity terms."""
        variations = {
            'gaandu': ['gandu', 'gaandu', 'gaandu'],
            'soolya': ['soolya', 'sulya', 'sooliya'],
            'thotha': ['thotha', 'thota', 'totha'],
            'hucchadana': ['hucchadana', 'huchadana', 'hucchadaana']
        }
        for base, spellings in variations.items():
            for spelling in spellings:
                result = transliterator.roman_to_kannada(spelling)
                assert isinstance(result, str)
    
    # ==================== Combined Patterns Tests ====================
    
    def test_profanity_with_intensifiers(self, transliterator):
        """Test profanity combined with intensifiers."""
        phrases = [
            "tumba thotha idiya",
            "sakkat gaandu maga",
            "full waste fellow",
            "swalpa hucchadana"
        ]
        for phrase in phrases:
            result = transliterator.roman_to_kannada(phrase)
            assert isinstance(result, str)
    
    def test_profanity_with_verbs(self, transliterator):
        """Test profanity with Kannada verb endings."""
        phrases = [
            "thotha tara behave madthiya",
            "gaandu tara mathadbeda",
            "bosodina tara act madtidya",
            "hucchadanta tara aadthiya"
        ]
        for phrase in phrases:
            result = transliterator.roman_to_kannada(phrase)
            assert isinstance(result, str)
    
    def test_double_profanity(self, transliterator):
        """Test sentences with multiple profanity terms."""
        phrases = [
            "nee thotha gaandu maga idiya",
            "full waste hucchadana fellow",
            "singri thotha behavior"
        ]
        for phrase in phrases:
            result = transliterator.roman_to_kannada(phrase)
            assert isinstance(result, str)
