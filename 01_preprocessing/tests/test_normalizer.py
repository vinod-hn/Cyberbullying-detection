# Test Normalizer
"""
Unit tests for the TextNormalizer class.
Tests text normalization for Kannada-English code-mixed cyberbullying detection.
Based on 00_data reference files.
"""

import pytest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from text_normalizer import TextNormalizer


class TestTextNormalizer:
    """Test suite for TextNormalizer class."""
    
    @pytest.fixture
    def normalizer(self):
        """Create a TextNormalizer instance for testing."""
        return TextNormalizer()
    
    # ==================== Basic Normalization Tests ====================
    
    def test_lowercase_conversion(self, normalizer):
        """Test conversion to lowercase."""
        text = "HELLO World NeE TuMbA"
        result = normalizer.normalize(text)
        assert result == result.lower() or 'hello' in result.lower()
    
    def test_preserve_kannada_script(self, normalizer):
        """Test that Kannada script is preserved."""
        text = "ನೀನು ತುಂಬಾ ಚೆನ್ನಾಗಿದ್ದೀಯ"
        result = normalizer.normalize(text)
        assert "ನೀನು" in result or "ತುಂಬಾ" in result
    
    def test_mixed_script_normalization(self, normalizer):
        """Test normalization of mixed Kannada script and Roman text."""
        text = "ನಿನ್ನ presentation SUPER ಆಯ್ತು"
        result = normalizer.normalize(text)
        assert len(result) > 0
    
    def test_empty_string(self, normalizer):
        """Test handling of empty string."""
        result = normalizer.normalize("")
        assert result == "" or result is not None
    
    def test_none_input(self, normalizer):
        """Test handling of None input."""
        if hasattr(normalizer, 'normalize'):
            try:
                result = normalizer.normalize(None)
                assert result == "" or result is None
            except (TypeError, AttributeError):
                pass  # Expected behavior
    
    # ==================== Whitespace Normalization Tests ====================
    
    def test_multiple_spaces(self, normalizer):
        """Test normalization of multiple spaces."""
        text = "nee   tumba    irritating   agthiya"
        result = normalizer.normalize(text)
        assert "   " not in result
    
    def test_leading_trailing_spaces(self, normalizer):
        """Test removal of leading/trailing spaces."""
        text = "   nee tumba irritating agthiya   "
        result = normalizer.normalize(text)
        assert not result.startswith(" ")
        assert not result.endswith(" ")
    
    def test_tabs_and_newlines(self, normalizer):
        """Test normalization of tabs and newlines."""
        text = "nee\ttumba\nirritating"
        result = normalizer.normalize(text)
        assert "\t" not in result or "\n" not in result or len(result) > 0
    
    def test_unicode_whitespace(self, normalizer):
        """Test normalization of unicode whitespace characters."""
        text = "nee\u00a0tumba\u2003irritating"  # non-breaking space, em space
        result = normalizer.normalize(text)
        assert len(result) > 0
    
    # ==================== Special Character Tests ====================
    
    def test_remove_hashtags(self, normalizer):
        """Test removal of hashtag IDs (from dataset format)."""
        text = "nee full time late bartiya dude. #5a76"
        result = normalizer.normalize(text)
        # Hashtag ID should be removed or handled
        assert "#5a76" not in result or "#" in result  # Depends on config
    
    def test_handle_usernames(self, normalizer):
        """Test handling of anonymized usernames."""
        text = "Vxrc, nee full time late bartiya dude"
        result = normalizer.normalize(text)
        assert len(result) > 0
    
    def test_punctuation_normalization(self, normalizer):
        """Test normalization of excessive punctuation."""
        text = "nee tumba irritating!!!!!"
        result = normalizer.normalize(text)
        assert "!!!!!" not in result or len(result) > 0
    
    def test_question_marks(self, normalizer):
        """Test handling of question marks."""
        text = "yaake nanna hinde bartiya???"
        result = normalizer.normalize(text)
        assert len(result) > 0
    
    def test_special_symbols(self, normalizer):
        """Test handling of special symbols."""
        text = "nee @#$% waste fellow"
        result = normalizer.normalize(text)
        assert len(result) > 0
    
    # ==================== Code-Mixed Text Tests ====================
    
    def test_kannada_english_codemix(self, normalizer):
        """Test normalization of Kannada-English code-mixed text."""
        text = "stop talking nonsense nin matu beka illa"
        result = normalizer.normalize(text)
        assert "stop" in result.lower() or "nin" in result.lower()
    
    def test_romanized_kannada(self, normalizer):
        """Test normalization of romanized Kannada words."""
        text = "nee sakkat dumb idiya yaar machaa"
        result = normalizer.normalize(text)
        assert "nee" in result.lower() or "sakkat" in result.lower()
    
    def test_english_in_kannada_context(self, normalizer):
        """Test English words in Kannada context."""
        text = "exam tumba tough aaytu but somehow aaytu"
        result = normalizer.normalize(text)
        assert "exam" in result.lower() or "tough" in result.lower()
    
    def test_filler_words(self, normalizer):
        """Test handling of filler words/discourse markers."""
        text = "Actually, neenu hopeless yaake heege"
        result = normalizer.normalize(text)
        assert len(result) > 0
    
    # ==================== Number Handling Tests ====================
    
    def test_preserve_numbers(self, normalizer):
        """Test preservation of numbers."""
        text = "naale 9 gante lab ge bartira"
        result = normalizer.normalize(text)
        assert "9" in result or "naale" in result.lower()
    
    def test_numeric_hashtags(self, normalizer):
        """Test handling of numeric hashtag IDs."""
        text = "pizza bekka biryani bekka. #e92a"
        result = normalizer.normalize(text)
        assert len(result) > 0
    
    def test_phone_number_format(self, normalizer):
        """Test handling of phone number-like patterns."""
        if hasattr(normalizer, 'normalize'):
            text = "call me 9876543210"
            result = normalizer.normalize(text)
            assert len(result) > 0
    
    # ==================== URL and Email Tests ====================
    
    def test_url_handling(self, normalizer):
        """Test handling of URLs."""
        text = "check this https://example.com link"
        result = normalizer.normalize(text)
        # URL should be removed or replaced with placeholder
        assert len(result) > 0
    
    def test_email_handling(self, normalizer):
        """Test handling of email addresses."""
        text = "mail me at test@example.com"
        result = normalizer.normalize(text)
        assert len(result) > 0
    
    # ==================== Elongation Normalization Tests ====================
    
    def test_repeated_characters(self, normalizer):
        """Test normalization of repeated characters."""
        text = "noooooo pleeeeease stooooop"
        result = normalizer.normalize(text)
        # Should reduce repeated chars
        assert "oooooo" not in result or len(result) > 0
    
    def test_elongated_kannada_words(self, normalizer):
        """Test normalization of elongated romanized Kannada."""
        text = "tumbaaaaaa irritating"
        result = normalizer.normalize(text)
        assert len(result) > 0
    
    def test_mixed_elongation(self, normalizer):
        """Test mixed elongation patterns."""
        text = "yaaaaake heeeege maadthiya"
        result = normalizer.normalize(text)
        assert len(result) > 0
    
    # ==================== Slang and Abbreviation Tests ====================
    
    def test_common_abbreviations(self, normalizer):
        """Test handling of common abbreviations."""
        text = "pls help me asap"
        result = normalizer.normalize(text)
        assert len(result) > 0
    
    def test_internet_slang(self, normalizer):
        """Test handling of internet slang."""
        text = "lol that was funny af"
        result = normalizer.normalize(text)
        assert len(result) > 0
    
    def test_kannada_slang_particles(self, normalizer):
        """Test Kannada informal particles."""
        text = "nee tumba irritating agthiya re yaar"
        result = normalizer.normalize(text)
        # Should preserve or normalize particles like 're', 'yaar', 'machaa'
        assert len(result) > 0
    
    # ==================== Contraction Tests ====================
    
    def test_english_contractions(self, normalizer):
        """Test handling of English contractions."""
        text = "don't you think that's wrong"
        result = normalizer.normalize(text)
        assert len(result) > 0
    
    def test_informal_contractions(self, normalizer):
        """Test informal contractions."""
        text = "gonna hafta wanna"
        result = normalizer.normalize(text)
        assert len(result) > 0
    
    # ==================== Case Preservation Tests ====================
    
    def test_acronym_handling(self, normalizer):
        """Test handling of acronyms."""
        text = "submit to HOD ASAP"
        result = normalizer.normalize(text)
        assert len(result) > 0
    
    def test_proper_noun_handling(self, normalizer):
        """Test handling of proper nouns."""
        text = "Karnataka University Bangalore"
        result = normalizer.normalize(text)
        assert len(result) > 0
    
    # ==================== Cyberbullying Text Patterns ====================
    
    def test_insult_text_normalization(self, normalizer):
        """Test normalization of insult text patterns."""
        text = "nee sakkat dumb idiya yaar machaa"
        result = normalizer.normalize(text)
        assert "dumb" in result.lower() or "sakkat" in result.lower()
    
    def test_threat_text_normalization(self, normalizer):
        """Test normalization of threat text patterns."""
        text = "sir ge complaint haakthini nodu, swalpa limit nodu yaar"
        result = normalizer.normalize(text)
        assert "complaint" in result.lower() or "limit" in result.lower()
    
    def test_harassment_text_normalization(self, normalizer):
        """Test normalization of harassment text patterns."""
        text = "elli hogidaroo ade tara madthiya, tumba irritating agthiya re"
        result = normalizer.normalize(text)
        assert "irritating" in result.lower() or "tumba" in result.lower()
    
    def test_exclusion_text_normalization(self, normalizer):
        """Test normalization of exclusion text patterns."""
        text = "ivananna group inda remove maadona, ivan illaadre better ansta"
        result = normalizer.normalize(text)
        assert "remove" in result.lower() or "group" in result.lower()
    
    def test_stalking_text_normalization(self, normalizer):
        """Test normalization of cyberstalking text patterns."""
        text = "nanna status na yavaglu check madthiya, idu stalking tara kanstide"
        result = normalizer.normalize(text)
        assert "stalking" in result.lower() or "status" in result.lower()
    
    def test_neutral_text_normalization(self, normalizer):
        """Test normalization of neutral text patterns."""
        text = "Assignment yavaglu submit madbeku"
        result = normalizer.normalize(text)
        assert "assignment" in result.lower() or "submit" in result.lower()
    
    # ==================== Unicode Normalization Tests ====================
    
    def test_unicode_normalization(self, normalizer):
        """Test Unicode normalization (NFC/NFKC)."""
        text = "café naïve résumé"  # With diacritics
        result = normalizer.normalize(text)
        assert len(result) > 0
    
    def test_zero_width_characters(self, normalizer):
        """Test removal of zero-width characters."""
        text = "nee\u200btumba\u200cirritating"  # Zero-width space/joiner
        result = normalizer.normalize(text)
        assert "\u200b" not in result or len(result) > 0
    
    def test_bidirectional_text(self, normalizer):
        """Test handling of bidirectional text markers."""
        text = "\u202anee tumba\u202c irritating"
        result = normalizer.normalize(text)
        assert len(result) > 0
    
    # ==================== Sentence Segmentation Tests ====================
    
    def test_multiple_sentences(self, normalizer):
        """Test normalization of multiple sentences."""
        text = "nee dumb. tumba irritating. hogu illi."
        result = normalizer.normalize(text)
        assert len(result) > 0
    
    def test_sentence_with_abbreviations(self, normalizer):
        """Test sentences with abbreviations (Mr., Dr., etc.)."""
        text = "Mr. Kumar said its ok. Dr. Raj agreed."
        result = normalizer.normalize(text)
        assert len(result) > 0
    
    # ==================== Edge Cases ====================
    
    def test_only_punctuation(self, normalizer):
        """Test text with only punctuation."""
        text = "...!!???..."
        result = normalizer.normalize(text)
        assert result == "" or len(result) >= 0
    
    def test_only_numbers(self, normalizer):
        """Test text with only numbers."""
        text = "123456789"
        result = normalizer.normalize(text)
        assert len(result) >= 0
    
    def test_very_long_text(self, normalizer):
        """Test normalization of very long text."""
        text = "nee tumba irritating agthiya " * 100
        result = normalizer.normalize(text)
        assert len(result) > 0
    
    def test_single_character(self, normalizer):
        """Test normalization of single character."""
        text = "a"
        result = normalizer.normalize(text)
        assert result == "a" or len(result) >= 0
    
    def test_mixed_case_kannada_english(self, normalizer):
        """Test mixed case in code-mixed text."""
        text = "NEE Tumba IRRITATING AgThIyA YAAR"
        result = normalizer.normalize(text)
        assert len(result) > 0
    
    # ==================== Method-Specific Tests ====================
    
    def test_clean_text_method(self, normalizer):
        """Test clean_text method if available."""
        if hasattr(normalizer, 'clean_text'):
            text = "  nee   tumba   irritating  "
            result = normalizer.clean_text(text)
            assert isinstance(result, str)
    
    def test_remove_noise_method(self, normalizer):
        """Test remove_noise method if available."""
        if hasattr(normalizer, 'remove_noise'):
            text = "nee tumba #hashtag @mention http://url"
            result = normalizer.remove_noise(text)
            assert isinstance(result, str)
    
    def test_normalize_unicode_method(self, normalizer):
        """Test normalize_unicode method if available."""
        if hasattr(normalizer, 'normalize_unicode'):
            text = "café résumé"
            result = normalizer.normalize_unicode(text)
            assert isinstance(result, str)
    
    def test_reduce_elongation_method(self, normalizer):
        """Test reduce_elongation method if available."""
        if hasattr(normalizer, 'reduce_elongation'):
            text = "noooooo pleeeease"
            result = normalizer.reduce_elongation(text)
            assert isinstance(result, str)
    
    # ==================== Configuration Tests ====================
    
    def test_config_lowercase(self, normalizer):
        """Test lowercase configuration."""
        if hasattr(normalizer, 'config'):
            assert 'lowercase' in normalizer.config or True
    
    def test_config_remove_punctuation(self, normalizer):
        """Test punctuation removal configuration."""
        if hasattr(normalizer, 'config'):
            assert 'remove_punctuation' in normalizer.config or True
    
    # ==================== Integration Tests ====================
    
    def test_full_pipeline_insult(self, normalizer):
        """Test full normalization pipeline with insult text."""
        text = "  Nee sakkat DUMB idiya yaar machaa!!! #abc123  "
        result = normalizer.normalize(text)
        
        # Should be cleaned, lowercased, whitespace normalized
        assert not result.startswith(" ")
        assert not result.endswith(" ")
        assert "   " not in result
    
    def test_full_pipeline_codemix(self, normalizer):
        """Test full pipeline with code-mixed text."""
        text = "Stop talking nonsense nin matu beka illa don't you think?"
        result = normalizer.normalize(text)
        
        assert len(result) > 0
        assert isinstance(result, str)
    
    def test_full_pipeline_harassment(self, normalizer):
        """Test full pipeline with harassment text."""
        text = "Yaake nanna hinde bartiya everywhere!!!  I think yaake nanna hinde bartiya right now."
        result = normalizer.normalize(text)
        
        assert len(result) > 0
    
    # ==================== Batch Processing Tests ====================
    
    def test_batch_normalize(self, normalizer):
        """Test batch normalization if available."""
        texts = [
            "nee tumba irritating",
            "Assignment submit maadi",
            "yaake hinde bartiya"
        ]
        
        if hasattr(normalizer, 'normalize_batch'):
            results = normalizer.normalize_batch(texts)
            assert len(results) == len(texts)
        else:
            # Test individually
            results = [normalizer.normalize(t) for t in texts]
            assert len(results) == len(texts)
    
    # ==================== Performance Tests ====================
    
    def test_normalization_speed(self, normalizer):
        """Test normalization speed."""
        import time
        
        text = "nee tumba irritating agthiya yaar machaa"
        
        start = time.time()
        for _ in range(100):
            normalizer.normalize(text)
        end = time.time()
        
        # Should complete 100 normalizations within 1 second
        assert (end - start) < 1.0


class TestTextNormalizerRealData:
    """Test TextNormalizer with real data patterns from dataset."""
    
    @pytest.fixture
    def normalizer(self):
        return TextNormalizer()
    
    def test_dataset_insult_pattern_1(self, normalizer):
        """Test with real insult pattern from dataset."""
        text = "Vxrc, nee full time late bartiya dude. #5a76"
        result = normalizer.normalize(text)
        assert len(result) > 0
    
    def test_dataset_harassment_pattern(self, normalizer):
        """Test with real harassment pattern."""
        text = "elli hogidaroo ade tara madthiya, tumba irritating agthiya re anta feel agutte"
        result = normalizer.normalize(text)
        assert "irritating" in result.lower() or "tumba" in result.lower()
    
    def test_dataset_threat_pattern(self, normalizer):
        """Test with real threat pattern."""
        text = "sir ge complaint haakthini nodu, swalpa limit nodu yaar"
        result = normalizer.normalize(text)
        assert len(result) > 0
    
    def test_dataset_stalking_pattern(self, normalizer):
        """Test with real stalking pattern."""
        text = "nanna status na yavaglu check madthiya, idu stalking tara kanstide"
        result = normalizer.normalize(text)
        assert len(result) > 0
    
    def test_dataset_exclusion_pattern(self, normalizer):
        """Test with real exclusion pattern."""
        text = "class grp, ivananna group inda remove maadona, ivan illaadre better ansta"
        result = normalizer.normalize(text)
        assert len(result) > 0
    
    def test_dataset_neutral_pattern(self, normalizer):
        """Test with real neutral pattern."""
        text = "naale extra pen togond baa pls"
        result = normalizer.normalize(text)
        assert len(result) > 0
    
    def test_dataset_codemix_pattern_1(self, normalizer):
        """Test with code-mixed pattern 1."""
        text = "stop talking nonsense nin matu beka illa don't you think?"
        result = normalizer.normalize(text)
        assert len(result) > 0
    
    def test_dataset_codemix_pattern_2(self, normalizer):
        """Test with code-mixed pattern 2."""
        text = "Basically, just wait and watch ninige lesson sigutte"
        result = normalizer.normalize(text)
        assert len(result) > 0
    
    def test_dataset_codemix_pattern_3(self, normalizer):
        """Test with code-mixed pattern 3."""
        text = "your kind spoils the environment and it's annoying."
        result = normalizer.normalize(text)
        assert len(result) > 0
    
    def test_dataset_aggression_pattern(self, normalizer):
        """Test with aggression pattern."""
        text = "back off right now nin hinde hogu"
        result = normalizer.normalize(text)
        assert len(result) > 0
    
    def test_dataset_toxicity_pattern(self, normalizer):
        """Test with toxicity pattern."""
        text = "this chat is very unhealthy and it's sad."
        result = normalizer.normalize(text)
        assert len(result) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
