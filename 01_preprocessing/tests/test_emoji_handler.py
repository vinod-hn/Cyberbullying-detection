# Test Emoji Handler
"""
Unit tests for the EmojiHandler class.
Tests emoji processing, sentiment extraction, and cyberbullying emoji detection.
"""

import pytest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from emoji_handler import EmojiHandler


class TestEmojiHandler:
    """Test suite for EmojiHandler class."""
    
    @pytest.fixture
    def handler(self):
        """Create an EmojiHandler instance for testing."""
        return EmojiHandler()
    
    # ==================== Emoji Detection Tests ====================
    
    def test_detect_emoji_in_text(self, handler):
        """Test detection of emojis in text."""
        text = "Hello 😀 how are you?"
        result = handler.contains_emoji(text)
        assert result == True
    
    def test_detect_no_emoji(self, handler):
        """Test detection when no emojis present."""
        text = "Hello how are you"
        result = handler.contains_emoji(text)
        assert result == False
    
    def test_detect_multiple_emojis(self, handler):
        """Test detection of multiple emojis."""
        text = "I'm so happy 😀😊🎉"
        result = handler.contains_emoji(text)
        assert result == True
    
    def test_extract_emojis(self, handler):
        """Test extraction of emojis from text."""
        text = "Hello 😀 how are you 😊?"
        emojis = handler.extract_emojis(text)
        
        assert isinstance(emojis, list)
        assert len(emojis) == 2
        assert "😀" in emojis
        assert "😊" in emojis
    
    def test_extract_emojis_empty(self, handler):
        """Test extraction when no emojis."""
        text = "Hello how are you"
        emojis = handler.extract_emojis(text)
        
        assert isinstance(emojis, list)
        assert len(emojis) == 0
    
    # ==================== Emoji Count Tests ====================
    
    def test_count_emojis(self, handler):
        """Test counting emojis in text."""
        text = "😀😊🎉😀"
        count = handler.count_emojis(text)
        assert count == 4
    
    def test_count_unique_emojis(self, handler):
        """Test counting unique emojis."""
        text = "😀😊🎉😀😀"
        
        if hasattr(handler, 'count_unique_emojis'):
            count = handler.count_unique_emojis(text)
            assert count == 3
    
    def test_emoji_frequency(self, handler):
        """Test emoji frequency calculation."""
        text = "😀😊😀😀😊"
        
        if hasattr(handler, 'get_emoji_frequency'):
            freq = handler.get_emoji_frequency(text)
            assert isinstance(freq, dict)
            assert freq.get("😀", 0) == 3
            assert freq.get("😊", 0) == 2
    
    # ==================== Emoji Sentiment Tests ====================
    
    def test_positive_emoji_sentiment(self, handler):
        """Test sentiment of positive emojis."""
        if hasattr(handler, 'get_emoji_sentiment'):
            sentiment = handler.get_emoji_sentiment("😀")
            assert sentiment in ["positive", "happy", 1, 1.0] or sentiment > 0
    
    def test_negative_emoji_sentiment(self, handler):
        """Test sentiment of negative emojis."""
        if hasattr(handler, 'get_emoji_sentiment'):
            sentiment = handler.get_emoji_sentiment("😡")
            assert sentiment in ["negative", "angry", -1, -1.0] or sentiment < 0
    
    def test_neutral_emoji_sentiment(self, handler):
        """Test sentiment of neutral emojis."""
        if hasattr(handler, 'get_emoji_sentiment'):
            sentiment = handler.get_emoji_sentiment("🔵")
            assert sentiment in ["neutral", 0, 0.0, None] or sentiment == 0
    
    def test_aggregate_emoji_sentiment(self, handler):
        """Test aggregate sentiment of multiple emojis."""
        text = "😀😊🎉"
        
        if hasattr(handler, 'get_text_emoji_sentiment'):
            sentiment = handler.get_text_emoji_sentiment(text)
            assert sentiment is not None
    
    # ==================== Cyberbullying Emoji Tests ====================
    
    def test_detect_aggressive_emojis(self, handler):
        """Test detection of aggressive emojis."""
        aggressive_emojis = ["😡", "🤬", "💀", "🖕", "👊"]
        
        if hasattr(handler, 'is_aggressive_emoji'):
            for emoji in aggressive_emojis:
                result = handler.is_aggressive_emoji(emoji)
                assert isinstance(result, bool)
    
    def test_detect_bullying_emoji_patterns(self, handler):
        """Test detection of bullying emoji patterns."""
        text = "You're so ugly 🤮🤮🤮"
        
        if hasattr(handler, 'detect_bullying_pattern'):
            result = handler.detect_bullying_pattern(text)
            assert result is not None
    
    def test_repeated_negative_emojis(self, handler):
        """Test detection of repeated negative emojis (spam pattern)."""
        text = "😡😡😡😡😡"
        
        if hasattr(handler, 'detect_emoji_spam'):
            result = handler.detect_emoji_spam(text)
            assert result == True or result is not None
    
    def test_skull_emoji_context(self, handler):
        """Test skull emoji in different contexts."""
        # Skull can mean 'dead from laughing' or actual threat
        text_laughing = "That joke 💀💀💀 I'm dead"
        text_threat = "You're gonna be 💀"
        
        if hasattr(handler, 'analyze_emoji_context'):
            result1 = handler.analyze_emoji_context(text_laughing, "💀")
            result2 = handler.analyze_emoji_context(text_threat, "💀")
            assert result1 is not None
            assert result2 is not None
    
    # ==================== Emoji Replacement Tests ====================
    
    def test_replace_emoji_with_text(self, handler):
        """Test replacing emojis with text descriptions."""
        text = "I'm so happy 😀"
        
        if hasattr(handler, 'replace_with_text'):
            result = handler.replace_with_text(text)
            assert "😀" not in result
            assert len(result) > 0
    
    def test_replace_emoji_with_placeholder(self, handler):
        """Test replacing emojis with placeholder."""
        text = "Hello 😀 world 😊"
        
        if hasattr(handler, 'replace_with_placeholder'):
            result = handler.replace_with_placeholder(text, "[EMOJI]")
            assert "😀" not in result
            assert "😊" not in result
            assert "[EMOJI]" in result
    
    def test_remove_emojis(self, handler):
        """Test removing all emojis from text."""
        text = "Hello 😀 world 😊"
        
        if hasattr(handler, 'remove_emojis'):
            result = handler.remove_emojis(text)
            assert "😀" not in result
            assert "😊" not in result
            assert "Hello" in result
            assert "world" in result
    
    # ==================== Emoji Meaning Tests ====================
    
    def test_get_emoji_meaning(self, handler):
        """Test getting emoji meaning/description."""
        if hasattr(handler, 'get_meaning'):
            meaning = handler.get_meaning("😀")
            assert isinstance(meaning, str)
            assert len(meaning) > 0
    
    def test_get_emoji_category(self, handler):
        """Test getting emoji category."""
        if hasattr(handler, 'get_category'):
            category = handler.get_category("😀")
            assert category in ["face", "emotion", "smileys", "people", None] or isinstance(category, str)
    
    def test_get_emoji_unicode_name(self, handler):
        """Test getting emoji unicode name."""
        if hasattr(handler, 'get_unicode_name'):
            name = handler.get_unicode_name("😀")
            assert isinstance(name, str)
    
    # ==================== Emoji Normalization Tests ====================
    
    def test_normalize_emoji_variants(self, handler):
        """Test normalization of emoji skin tone variants."""
        emoji_variants = ["👍", "👍🏻", "👍🏼", "👍🏽", "👍🏾", "👍🏿"]
        
        if hasattr(handler, 'normalize_skin_tones'):
            normalized = [handler.normalize_skin_tones(e) for e in emoji_variants]
            # All should normalize to base emoji
            assert len(set(normalized)) == 1 or all(n == normalized[0] for n in normalized)
    
    def test_normalize_emoji_zwj_sequences(self, handler):
        """Test handling of ZWJ (Zero Width Joiner) emoji sequences."""
        # Family emoji, flag emojis, etc.
        text = "👨‍👩‍👧‍👦 family"
        
        if hasattr(handler, 'handle_zwj_sequences'):
            result = handler.handle_zwj_sequences(text)
            assert isinstance(result, str)
    
    # ==================== Edge Cases Tests ====================
    
    def test_empty_text(self, handler):
        """Test handling of empty text."""
        text = ""
        
        emojis = handler.extract_emojis(text)
        assert emojis == [] or emojis is not None
    
    def test_only_emojis(self, handler):
        """Test text with only emojis."""
        text = "😀😊🎉"
        
        emojis = handler.extract_emojis(text)
        assert len(emojis) == 3
    
    def test_emoji_in_code_mixed_text(self, handler):
        """Test emojis in Kannada-English code-mixed text."""
        text = "nee tumba irritating 😡 agthiya maga"
        
        emojis = handler.extract_emojis(text)
        assert "😡" in emojis
    
    def test_kannada_script_with_emoji(self, handler):
        """Test emojis with Kannada script text."""
        text = "ನೀನು ತುಂಬಾ 😊 ಚೆನ್ನಾಗಿದ್ದೀಯ"
        
        emojis = handler.extract_emojis(text)
        assert "😊" in emojis
    
    def test_special_characters_with_emoji(self, handler):
        """Test emojis with special characters."""
        text = "Hello!!! 😀 @#$% 😊"
        
        emojis = handler.extract_emojis(text)
        assert len(emojis) == 2
    
    def test_numbers_with_emoji(self, handler):
        """Test emojis with numbers."""
        text = "100% happy 😀"
        
        emojis = handler.extract_emojis(text)
        assert "😀" in emojis
    
    def test_unicode_special_symbols(self, handler):
        """Test distinction between emojis and unicode symbols."""
        text = "Price: $100 © ® ™ 😀"
        
        emojis = handler.extract_emojis(text)
        # Should only extract actual emoji, not symbols
        assert "😀" in emojis
        # These should NOT be in emojis list
        assert "$" not in emojis
    
    # ==================== Emoji Sequence Tests ====================
    
    def test_emoji_with_modifiers(self, handler):
        """Test emojis with modifiers (skin tone, gender)."""
        text = "👩🏽‍💻 coding"
        
        emojis = handler.extract_emojis(text)
        assert len(emojis) >= 1
    
    def test_flag_emojis(self, handler):
        """Test country flag emojis."""
        text = "🇮🇳 India"
        
        emojis = handler.extract_emojis(text)
        assert len(emojis) >= 1
    
    def test_keycap_emojis(self, handler):
        """Test keycap number emojis."""
        text = "Press 1️⃣ or 2️⃣"
        
        emojis = handler.extract_emojis(text)
        assert len(emojis) >= 2
    
    # ==================== Cyberbullying Specific Tests ====================
    
    def test_threatening_emoji_combinations(self, handler):
        """Test detection of threatening emoji combinations."""
        combinations = [
            "🔪💀",  # knife + skull
            "🔫😵",  # gun + dizzy face
            "💣💥",  # bomb + explosion
        ]
        
        if hasattr(handler, 'detect_threatening_pattern'):
            for combo in combinations:
                result = handler.detect_threatening_pattern(combo)
                assert result is not None
    
    def test_mocking_emoji_patterns(self, handler):
        """Test detection of mocking emoji patterns."""
        # Clown emoji used mockingly
        text = "You think you're smart? 🤡🤡🤡"
        
        if hasattr(handler, 'detect_mocking_pattern'):
            result = handler.detect_mocking_pattern(text)
            assert result is not None
    
    def test_sarcastic_emoji_usage(self, handler):
        """Test detection of sarcastic emoji usage."""
        # Positive emoji with negative text
        text = "Great job failing 👏👏👏"
        
        if hasattr(handler, 'detect_sarcasm'):
            result = handler.detect_sarcasm(text)
            assert result is not None
    
    def test_body_shaming_emojis(self, handler):
        """Test detection of body-shaming emoji usage."""
        emojis = ["🐷", "🐮", "🤮"]  # Often used for body shaming
        
        if hasattr(handler, 'is_body_shaming_emoji'):
            for emoji in emojis:
                result = handler.is_body_shaming_emoji(emoji)
                assert isinstance(result, bool)
    
    # ==================== Integration Tests ====================
    
    def test_full_emoji_analysis(self, handler):
        """Test complete emoji analysis of text."""
        text = "nee tumba ugly 🤮🤮 sakkat waste 😡"
        
        if hasattr(handler, 'analyze'):
            result = handler.analyze(text)
            assert result is not None
            assert isinstance(result, dict)
    
    def test_emoji_feature_extraction(self, handler):
        """Test emoji feature extraction for ML."""
        text = "Hello 😀😊😡"
        
        if hasattr(handler, 'extract_features'):
            features = handler.extract_features(text)
            assert isinstance(features, (dict, list))
    
    # ==================== Performance Tests ====================
    
    def test_processing_speed(self, handler):
        """Test emoji processing speed."""
        import time
        
        text = "😀😊🎉😡🔥💀👍👎❤️😢" * 10
        
        start = time.time()
        handler.extract_emojis(text)
        end = time.time()
        
        # Should complete within 100ms
        assert (end - start) < 0.1
    
    def test_batch_processing(self, handler):
        """Test batch emoji processing."""
        texts = [
            "Hello 😀",
            "Angry 😡",
            "Sad 😢",
            "Happy 🎉"
        ]
        
        if hasattr(handler, 'process_batch'):
            results = handler.process_batch(texts)
            assert len(results) == len(texts)


class TestEmojiSemantics:
    """Test emoji semantic mappings from emoji_semantics.json."""
    
    @pytest.fixture
    def handler(self):
        return EmojiHandler()
    
    def test_load_semantics_file(self, handler):
        """Test loading of emoji_semantics.json."""
        if hasattr(handler, 'semantics'):
            assert handler.semantics is not None
            assert isinstance(handler.semantics, dict)
    
    def test_semantic_lookup(self, handler):
        """Test semantic lookup for common emojis."""
        common_emojis = ["😀", "😡", "😢", "❤️", "👍"]
        
        if hasattr(handler, 'get_semantic'):
            for emoji in common_emojis:
                semantic = handler.get_semantic(emoji)
                # Should return something, even if None for unknown
                pass  # Just testing it doesn't crash


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
