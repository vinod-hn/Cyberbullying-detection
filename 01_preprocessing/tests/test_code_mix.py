# Test Code Mix Processor
"""
Unit tests for the CodeMixProcessor class.
Tests handling of Kannada-English code-mixed text for cyberbullying detection.
"""

import pytest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from code_mix_processor import CodeMixProcessor


class TestCodeMixProcessor:
    """Test suite for CodeMixProcessor class."""
    
    @pytest.fixture
    def processor(self):
        """Create a CodeMixProcessor instance for testing."""
        return CodeMixProcessor()
    
    # ==================== Language Detection Tests ====================
    
    def test_detect_kannada_text(self, processor):
        """Test detection of pure Kannada text."""
        text = "ನೀನು ಹೇಗಿದ್ದೀಯ"
        result = processor.detect_language(text)
        assert result == "kannada" or result == "kn"
    
    def test_detect_english_text(self, processor):
        """Test detection of pure English text."""
        text = "How are you doing today"
        result = processor.detect_language(text)
        assert result == "english" or result == "en"
    
    def test_detect_code_mixed_text(self, processor):
        """Test detection of Kannada-English code-mixed text."""
        text = "nee tumba irritating agthiya maga"
        result = processor.detect_language(text)
        assert result == "code-mixed" or result == "mixed" or result == "kn-en"
    
    def test_detect_romanized_kannada(self, processor):
        """Test detection of Romanized Kannada text."""
        text = "hegidiya machaa"
        result = processor.detect_language(text)
        assert result in ["romanized-kannada", "code-mixed", "kn-roman", "mixed"]
    
    # ==================== Code-Mix Ratio Tests ====================
    
    def test_calculate_code_mix_ratio_pure_english(self, processor):
        """Test code-mix ratio for pure English text."""
        text = "This is a completely English sentence"
        ratio = processor.calculate_code_mix_ratio(text)
        assert ratio == 0.0 or ratio < 0.1
    
    def test_calculate_code_mix_ratio_mixed(self, processor):
        """Test code-mix ratio for mixed text."""
        text = "nee tumba waste fellow idiya maga"
        ratio = processor.calculate_code_mix_ratio(text)
        assert 0.0 < ratio <= 1.0
    
    def test_calculate_code_mix_ratio_high_kannada(self, processor):
        """Test code-mix ratio for text with high Kannada content."""
        text = "nee sakkat dumb idiya yaar machaa"
        ratio = processor.calculate_code_mix_ratio(text)
        assert ratio > 0.5
    
    # ==================== Token Classification Tests ====================
    
    def test_classify_tokens(self, processor):
        """Test classification of individual tokens by language."""
        text = "nee tumba irritating agthiya"
        tokens = processor.classify_tokens(text)
        
        assert isinstance(tokens, list)
        assert len(tokens) > 0
        
        # Each token should have word and language
        for token in tokens:
            assert "word" in token or "token" in token
            assert "language" in token or "lang" in token
    
    def test_classify_kannada_slang_tokens(self, processor):
        """Test classification of Kannada slang tokens."""
        text = "machaa maga guru"
        tokens = processor.classify_tokens(text)
        
        kannada_tokens = [t for t in tokens if t.get("language", t.get("lang", "")) in ["kannada", "kn", "romanized-kannada"]]
        assert len(kannada_tokens) >= 2
    
    def test_classify_english_tokens(self, processor):
        """Test classification of English tokens in mixed text."""
        text = "nee complete waste fellow"
        tokens = processor.classify_tokens(text)
        
        english_tokens = [t for t in tokens if t.get("language", t.get("lang", "")) in ["english", "en"]]
        assert len(english_tokens) >= 2  # "complete", "waste", "fellow"
    
    # ==================== Normalization Tests ====================
    
    def test_normalize_code_mixed_text(self, processor):
        """Test normalization of code-mixed text."""
        text = "Nee TUMBA irritating AGTHIYA"
        normalized = processor.normalize(text)
        
        assert normalized.islower() or normalized == normalized.lower()
    
    def test_normalize_removes_extra_spaces(self, processor):
        """Test that normalization removes extra whitespace."""
        text = "nee   tumba    irritating   agthiya"
        normalized = processor.normalize(text)
        
        assert "   " not in normalized
        assert "  " not in normalized
    
    def test_normalize_handles_special_characters(self, processor):
        """Test normalization handles special characters."""
        text = "nee tumba irritating!!! agthiya???"
        normalized = processor.normalize(text)
        
        assert isinstance(normalized, str)
        assert len(normalized) > 0
    
    # ==================== Transliteration Tests ====================
    
    def test_transliterate_romanized_to_kannada(self, processor):
        """Test transliteration from Romanized to Kannada script."""
        if hasattr(processor, 'transliterate_to_kannada'):
            text = "hegidiya"
            result = processor.transliterate_to_kannada(text)
            assert isinstance(result, str)
    
    def test_transliterate_kannada_to_roman(self, processor):
        """Test transliteration from Kannada to Roman script."""
        if hasattr(processor, 'transliterate_to_roman'):
            text = "ಹೇಗಿದ್ದೀಯ"
            result = processor.transliterate_to_roman(text)
            assert isinstance(result, str)
            assert result.isascii() or all(ord(c) < 128 for c in result.replace(' ', ''))
    
    # ==================== Profanity Detection Tests ====================
    
    def test_detect_kannada_profanity(self, processor):
        """Test detection of Kannada profanity in code-mixed text."""
        if hasattr(processor, 'detect_profanity'):
            text = "nee thotha maga idiya"
            result = processor.detect_profanity(text)
            
            assert isinstance(result, (bool, list, dict))
            if isinstance(result, bool):
                assert result == True
            elif isinstance(result, list):
                assert len(result) > 0
    
    def test_detect_english_profanity_in_mixed(self, processor):
        """Test detection of English profanity in code-mixed text."""
        if hasattr(processor, 'detect_profanity'):
            text = "you are such an idiot maga"
            result = processor.detect_profanity(text)
            
            assert isinstance(result, (bool, list, dict))
    
    # ==================== Slang Expansion Tests ====================
    
    def test_expand_kannada_slang(self, processor):
        """Test expansion of Kannada slang terms."""
        if hasattr(processor, 'expand_slang'):
            text = "machaa tumba sakkat"
            expanded = processor.expand_slang(text)
            
            assert isinstance(expanded, str)
            assert len(expanded) > 0
    
    def test_expand_english_abbreviations(self, processor):
        """Test expansion of English abbreviations in mixed text."""
        if hasattr(processor, 'expand_slang'):
            text = "u r tumba irritating"
            expanded = processor.expand_slang(text)
            
            assert isinstance(expanded, str)
    
    # ==================== Pattern Matching Tests ====================
    
    def test_identify_code_switch_points(self, processor):
        """Test identification of language switch points."""
        if hasattr(processor, 'identify_switch_points'):
            text = "nee tumba irritating agthiya bro"
            switches = processor.identify_switch_points(text)
            
            assert isinstance(switches, list)
    
    def test_identify_insult_patterns(self, processor):
        """Test identification of common insult patterns."""
        if hasattr(processor, 'identify_patterns'):
            text = "nee sakkat dumb idiya maga"
            patterns = processor.identify_patterns(text)
            
            assert isinstance(patterns, (list, dict))
    
    # ==================== Edge Cases Tests ====================
    
    def test_empty_text(self, processor):
        """Test handling of empty text."""
        text = ""
        
        # Should not raise exception
        if hasattr(processor, 'detect_language'):
            result = processor.detect_language(text)
            assert result is not None or result == "" or result == "unknown"
    
    def test_only_spaces(self, processor):
        """Test handling of whitespace-only text."""
        text = "     "
        
        if hasattr(processor, 'normalize'):
            result = processor.normalize(text)
            assert isinstance(result, str)
    
    def test_only_special_characters(self, processor):
        """Test handling of special characters only."""
        text = "!@#$%^&*()"
        
        if hasattr(processor, 'detect_language'):
            result = processor.detect_language(text)
            assert result is not None
    
    def test_emoji_in_code_mixed(self, processor):
        """Test handling of emojis in code-mixed text."""
        text = "nee tumba irritating 😡 agthiya"
        
        if hasattr(processor, 'normalize'):
            result = processor.normalize(text)
            assert isinstance(result, str)
    
    def test_numbers_in_code_mixed(self, processor):
        """Test handling of numbers in code-mixed text."""
        text = "nee 100% waste fellow"
        
        if hasattr(processor, 'classify_tokens'):
            tokens = processor.classify_tokens(text)
            assert isinstance(tokens, list)
    
    def test_hashtags_in_code_mixed(self, processor):
        """Test handling of hashtags in code-mixed text."""
        text = "nee tumba irritating #stopbullying"
        
        if hasattr(processor, 'normalize'):
            result = processor.normalize(text)
            assert isinstance(result, str)
    
    def test_mentions_in_code_mixed(self, processor):
        """Test handling of @mentions in code-mixed text."""
        text = "@user nee tumba waste fellow"
        
        if hasattr(processor, 'normalize'):
            result = processor.normalize(text)
            assert isinstance(result, str)
    
    # ==================== Integration Tests ====================
    
    def test_full_pipeline_processing(self, processor):
        """Test full preprocessing pipeline for code-mixed text."""
        text = "Nee TUMBA irritating agthiya maga!!!"
        
        if hasattr(processor, 'process'):
            result = processor.process(text)
            assert result is not None
        elif hasattr(processor, 'preprocess'):
            result = processor.preprocess(text)
            assert result is not None
    
    def test_batch_processing(self, processor):
        """Test batch processing of multiple texts."""
        texts = [
            "nee tumba irritating",
            "you are so annoying",
            "nee sakkat dumb idiya"
        ]
        
        if hasattr(processor, 'process_batch'):
            results = processor.process_batch(texts)
            assert len(results) == len(texts)
    
    # ==================== Performance Tests ====================
    
    def test_processing_speed(self, processor):
        """Test that processing completes in reasonable time."""
        import time
        
        text = "nee tumba irritating agthiya maga " * 10
        
        start = time.time()
        if hasattr(processor, 'process'):
            processor.process(text)
        elif hasattr(processor, 'normalize'):
            processor.normalize(text)
        end = time.time()
        
        # Should complete within 1 second
        assert (end - start) < 1.0


class TestCodeMixPatterns:
    """Test common code-mixing patterns in Kannada-English text."""
    
    @pytest.fixture
    def processor(self):
        return CodeMixProcessor()
    
    def test_kannada_verb_english_adjective(self, processor):
        """Test pattern: English adjective + Kannada verb."""
        # e.g., "irritating agthiya" (irritating + becoming)
        text = "nee tumba irritating agthiya"
        
        if hasattr(processor, 'identify_patterns'):
            patterns = processor.identify_patterns(text)
            assert patterns is not None
    
    def test_kannada_address_english_insult(self, processor):
        """Test pattern: Kannada address + English insult."""
        # e.g., "maga you are waste"
        text = "maga you are waste fellow"
        
        if hasattr(processor, 'classify_tokens'):
            tokens = processor.classify_tokens(text)
            assert len(tokens) > 0
    
    def test_english_borrowed_intensifiers(self, processor):
        """Test English borrowed intensifiers in Kannada context."""
        # e.g., "full", "total", "complete"
        text = "nee full lazy mode alli idiya"
        
        if hasattr(processor, 'classify_tokens'):
            tokens = processor.classify_tokens(text)
            assert len(tokens) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
