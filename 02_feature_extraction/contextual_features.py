# Contextual Features
"""
ContextualFeatures: Conversation and context-aware feature extraction.
Extracts features from message context, conversation history, and user interactions.
Optimized for cyberbullying detection in Kannada-English code-mixed text.
"""

import re
import logging
from typing import List, Dict, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np

logger = logging.getLogger(__name__)


class ContextualFeatures:
    """
    Contextual feature extractor for cyberbullying detection.
    
    Extracts features based on:
    - Conversation history (previous messages)
    - Message position in thread
    - User mention patterns
    - Response patterns
    - Escalation indicators
    
    Attributes:
        config: Configuration dictionary
        window_size: Number of previous messages to consider
    """
    
    # Patterns for detecting targets
    MENTION_PATTERN = re.compile(r'@\w+')
    HASHTAG_PATTERN = re.compile(r'#[a-f0-9]{4}\b')  # Dataset anonymized IDs
    NAME_PATTERN = re.compile(r'\b[A-Z][a-z]+\b')
    
    # Escalation keywords
    ESCALATION_WORDS = {
        'warning': ['warning', 'warn', 'careful', 'watch out', 'limit', 'swalpa limit'],
        'threat': ['kill', 'die', 'hurt', 'destroy', 'end', 'finish', 'complaint', 'report'],
        'ultimatum': ['last time', 'final', 'never again', 'or else', 'mundina sari'],
        'aggression': ['hate', 'angry', 'furious', 'rage', 'mad', 'irritating', 'annoying']
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize ContextualFeatures.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or self._default_config()
        self.window_size = self.config.get('window_size', 5)
    
    def _default_config(self) -> Dict[str, Any]:
        """Return default configuration."""
        return {
            'window_size': 5,
            'include_position': True,
            'include_mentions': True,
            'include_response': True,
            'include_escalation': True,
            'include_temporal': True
        }
    
    # =========================================================================
    # Position Features
    # =========================================================================
    def extract_position_features(
        self,
        message_idx: int,
        total_messages: int
    ) -> Dict[str, Any]:
        """
        Extract position-based features.
        
        Args:
            message_idx: Index of current message (0-based)
            total_messages: Total messages in conversation
            
        Returns:
            Position features dictionary
        """
        if total_messages == 0:
            return {
                'is_first_message': False,
                'is_last_message': False,
                'position_ratio': 0.0,
                'messages_before': 0,
                'messages_after': 0
            }
        
        position_ratio = message_idx / max(total_messages - 1, 1)
        
        return {
            'is_first_message': message_idx == 0,
            'is_last_message': message_idx == total_messages - 1,
            'position_ratio': round(position_ratio, 3),
            'messages_before': message_idx,
            'messages_after': total_messages - message_idx - 1,
            'is_early': message_idx < total_messages * 0.25,
            'is_late': message_idx > total_messages * 0.75
        }
    
    # =========================================================================
    # Mention Features
    # =========================================================================
    def extract_mention_features(self, text: str) -> Dict[str, Any]:
        """
        Extract mention and targeting features.
        
        Args:
            text: Input text
            
        Returns:
            Mention features dictionary
        """
        # Find @mentions
        mentions = self.MENTION_PATTERN.findall(text)
        
        # Find anonymized IDs (dataset format)
        anon_ids = self.HASHTAG_PATTERN.findall(text)
        
        # Find potential names (capitalized words)
        names = self.NAME_PATTERN.findall(text)
        # Filter common English words
        common_words = {'The', 'This', 'That', 'What', 'Why', 'How', 'When', 'Where'}
        names = [n for n in names if n not in common_words]
        
        # Check for direct address
        direct_address_patterns = [
            r'\b(you|your|u|ur)\b',
            r'\b(nee|neenu|ninna|nimma)\b',  # Kannada "you"
        ]
        has_direct_address = any(
            re.search(pattern, text.lower())
            for pattern in direct_address_patterns
        )
        
        return {
            'mention_count': len(mentions),
            'mentions': mentions,
            'anon_id_count': len(anon_ids),
            'anon_ids': anon_ids,
            'potential_name_count': len(names),
            'has_direct_address': has_direct_address,
            'has_target': len(mentions) > 0 or len(anon_ids) > 0 or has_direct_address
        }
    
    # =========================================================================
    # Response Features
    # =========================================================================
    def extract_response_features(
        self,
        current_message: str,
        previous_messages: List[str],
        current_user: Optional[str] = None,
        previous_users: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Extract response pattern features.
        
        Args:
            current_message: Current message text
            previous_messages: List of previous messages in conversation
            current_user: Current message sender
            previous_users: Senders of previous messages
            
        Returns:
            Response features dictionary
        """
        if not previous_messages:
            return {
                'is_response': False,
                'response_length_ratio': 0.0,
                'is_escalation': False,
                'same_user_repeat': False,
                'context_messages': 0
            }
        
        # Length comparison
        current_length = len(current_message.split())
        prev_lengths = [len(m.split()) for m in previous_messages]
        avg_prev_length = np.mean(prev_lengths) if prev_lengths else 1
        
        length_ratio = current_length / max(avg_prev_length, 1)
        
        # Check for word overlap (likely response)
        current_words = set(current_message.lower().split())
        prev_words = set()
        for msg in previous_messages:
            prev_words.update(msg.lower().split())
        
        word_overlap = len(current_words & prev_words)
        overlap_ratio = word_overlap / max(len(current_words), 1)
        
        # Same user repeat
        same_user = False
        if current_user and previous_users:
            same_user = current_user in previous_users
        
        return {
            'is_response': overlap_ratio > 0.1,
            'response_length_ratio': round(length_ratio, 3),
            'word_overlap_ratio': round(overlap_ratio, 3),
            'same_user_repeat': same_user,
            'context_messages': len(previous_messages),
            'is_longer_response': length_ratio > 1.5
        }
    
    # =========================================================================
    # Escalation Features
    # =========================================================================
    def extract_escalation_features(
        self,
        current_message: str,
        previous_messages: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Extract escalation pattern features.
        
        Args:
            current_message: Current message text
            previous_messages: Previous messages for comparison
            
        Returns:
            Escalation features dictionary
        """
        text_lower = current_message.lower()
        
        # Check for escalation keywords
        escalation_scores = {}
        total_escalation = 0
        
        for category, keywords in self.ESCALATION_WORDS.items():
            count = sum(1 for kw in keywords if kw in text_lower)
            escalation_scores[f'escalation_{category}'] = count
            total_escalation += count
        
        # Intensity indicators
        exclamation_count = current_message.count('!')
        caps_ratio = sum(1 for c in current_message if c.isupper()) / max(len(current_message), 1)
        
        # Compare with previous messages
        is_intensifying = False
        if previous_messages:
            prev_text = ' '.join(previous_messages).lower()
            prev_escalation = sum(
                1 for keywords in self.ESCALATION_WORDS.values()
                for kw in keywords if kw in prev_text
            )
            is_intensifying = total_escalation > prev_escalation / max(len(previous_messages), 1)
        
        return {
            **escalation_scores,
            'total_escalation_score': total_escalation,
            'exclamation_count': exclamation_count,
            'caps_ratio': round(caps_ratio, 3),
            'is_intensifying': is_intensifying,
            'has_escalation': total_escalation > 0,
            'escalation_level': self._calculate_escalation_level(total_escalation, caps_ratio)
        }
    
    def _calculate_escalation_level(
        self,
        escalation_score: int,
        caps_ratio: float
    ) -> str:
        """Calculate overall escalation level."""
        score = escalation_score + (10 * caps_ratio)
        
        if score < 1:
            return 'none'
        elif score < 3:
            return 'low'
        elif score < 5:
            return 'medium'
        else:
            return 'high'
    
    # =========================================================================
    # Temporal Features
    # =========================================================================
    def extract_temporal_features(
        self,
        current_timestamp: Optional[datetime] = None,
        previous_timestamps: Optional[List[datetime]] = None
    ) -> Dict[str, Any]:
        """
        Extract temporal features.
        
        Args:
            current_timestamp: Timestamp of current message
            previous_timestamps: Timestamps of previous messages
            
        Returns:
            Temporal features dictionary
        """
        if current_timestamp is None:
            return {
                'has_temporal_info': False,
                'time_gap_seconds': 0,
                'is_rapid_response': False,
                'is_delayed_response': False
            }
        
        features = {
            'has_temporal_info': True,
            'hour_of_day': current_timestamp.hour,
            'day_of_week': current_timestamp.weekday(),
            'is_weekend': current_timestamp.weekday() >= 5,
            'is_night': current_timestamp.hour < 6 or current_timestamp.hour >= 22
        }
        
        if previous_timestamps and len(previous_timestamps) > 0:
            last_timestamp = previous_timestamps[-1]
            time_gap = (current_timestamp - last_timestamp).total_seconds()
            
            features['time_gap_seconds'] = time_gap
            features['is_rapid_response'] = time_gap < 60  # Less than 1 minute
            features['is_delayed_response'] = time_gap > 3600  # More than 1 hour
            
            # Message frequency
            if len(previous_timestamps) > 1:
                gaps = [
                    (previous_timestamps[i] - previous_timestamps[i-1]).total_seconds()
                    for i in range(1, len(previous_timestamps))
                ]
                features['avg_response_time'] = np.mean(gaps)
        else:
            features['time_gap_seconds'] = 0
            features['is_rapid_response'] = False
            features['is_delayed_response'] = False
        
        return features
    
    # =========================================================================
    # Conversation-Level Features
    # =========================================================================
    def extract_conversation_features(
        self,
        messages: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Extract conversation-level features.
        
        Args:
            messages: List of message dictionaries with 'text', 'user', 'timestamp'
            
        Returns:
            Conversation features dictionary
        """
        if not messages:
            return {
                'conversation_length': 0,
                'unique_participants': 0,
                'avg_message_length': 0,
                'escalation_trend': 'none'
            }
        
        texts = [m.get('text', '') for m in messages]
        users = [m.get('user', f'user_{i}') for i, m in enumerate(messages)]
        
        # Basic stats
        conversation_length = len(messages)
        unique_participants = len(set(users))
        avg_message_length = np.mean([len(t.split()) for t in texts])
        
        # User dominance
        user_counts = defaultdict(int)
        for user in users:
            user_counts[user] += 1
        
        if user_counts:
            max_user = max(user_counts.values())
            dominance_ratio = max_user / conversation_length
        else:
            dominance_ratio = 0
        
        # Escalation trend
        escalation_trend = self._detect_escalation_trend(texts)
        
        # Back-and-forth detection
        is_back_and_forth = self._is_back_and_forth(users)
        
        return {
            'conversation_length': conversation_length,
            'unique_participants': unique_participants,
            'avg_message_length': round(avg_message_length, 2),
            'user_dominance_ratio': round(dominance_ratio, 3),
            'escalation_trend': escalation_trend,
            'is_back_and_forth': is_back_and_forth,
            'is_one_sided': unique_participants == 1 or dominance_ratio > 0.8
        }
    
    def _detect_escalation_trend(self, texts: List[str]) -> str:
        """Detect if conversation is escalating."""
        if len(texts) < 3:
            return 'none'
        
        escalation_scores = []
        for text in texts:
            score = 0
            text_lower = text.lower()
            for keywords in self.ESCALATION_WORDS.values():
                score += sum(1 for kw in keywords if kw in text_lower)
            escalation_scores.append(score)
        
        # Check trend
        first_half = np.mean(escalation_scores[:len(texts)//2])
        second_half = np.mean(escalation_scores[len(texts)//2:])
        
        if second_half > first_half * 1.5:
            return 'escalating'
        elif second_half < first_half * 0.5:
            return 'de-escalating'
        else:
            return 'stable'
    
    def _is_back_and_forth(self, users: List[str]) -> bool:
        """Check if conversation is back-and-forth between two users."""
        unique_users = list(set(users))
        if len(unique_users) != 2:
            return False
        
        alternations = 0
        for i in range(1, len(users)):
            if users[i] != users[i-1]:
                alternations += 1
        
        return alternations >= len(users) * 0.7
    
    # =========================================================================
    # Combined Extraction
    # =========================================================================
    def extract(
        self,
        texts: Union[str, List[str]],
        context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract contextual features from texts.
        
        Args:
            texts: Text or list of texts
            context: Optional context with 'previous_messages', 'users', 'timestamps'
            
        Returns:
            List of feature dictionaries
        """
        if isinstance(texts, str):
            texts = [texts]
        
        context = context or {}
        previous_messages = context.get('previous_messages', [])
        previous_users = context.get('previous_users', [])
        current_user = context.get('current_user')
        
        results = []
        
        for i, text in enumerate(texts):
            features = {}
            
            # Position features
            if self.config.get('include_position', True):
                total = len(texts) + len(previous_messages)
                pos = len(previous_messages) + i
                features.update(self.extract_position_features(pos, total))
            
            # Mention features
            if self.config.get('include_mentions', True):
                features.update(self.extract_mention_features(text))
            
            # Response features
            if self.config.get('include_response', True):
                prev = previous_messages + texts[:i]
                features.update(self.extract_response_features(
                    text, prev[-self.window_size:], current_user, previous_users
                ))
            
            # Escalation features
            if self.config.get('include_escalation', True):
                prev = previous_messages + texts[:i]
                features.update(self.extract_escalation_features(
                    text, prev[-self.window_size:]
                ))
            
            results.append(features)
        
        return results
    
    def extract_numeric_features(
        self,
        texts: Union[str, List[str]],
        context: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """
        Extract numeric contextual features.
        
        Returns:
            Feature matrix (n_samples, n_features)
        """
        all_features = self.extract(texts, context)
        
        numeric_keys = [
            'position_ratio', 'messages_before', 'messages_after',
            'mention_count', 'anon_id_count', 'has_direct_address',
            'response_length_ratio', 'word_overlap_ratio', 'context_messages',
            'total_escalation_score', 'exclamation_count', 'caps_ratio'
        ]
        
        matrix = []
        for features in all_features:
            row = []
            for key in numeric_keys:
                val = features.get(key, 0)
                if isinstance(val, bool):
                    val = int(val)
                row.append(val)
            matrix.append(row)
        
        return np.array(matrix)
    
    def get_feature_names(self) -> List[str]:
        """Get list of numeric feature names."""
        return [
            'position_ratio', 'messages_before', 'messages_after',
            'mention_count', 'anon_id_count', 'has_direct_address',
            'response_length_ratio', 'word_overlap_ratio', 'context_messages',
            'total_escalation_score', 'exclamation_count', 'caps_ratio'
        ]
    
    def __repr__(self) -> str:
        """String representation."""
        return f"ContextualFeatures(window_size={self.window_size})"


if __name__ == "__main__":
    print("ContextualFeatures Test")
    print("=" * 50)
    
    extractor = ContextualFeatures()
    
    # Test with context
    previous = [
        "Hey what's up?",
        "Nothing much, studying for exam",
        "Same here, it's tough"
    ]
    
    current = "nee tumba irritating agthiya! Stop messaging me!!"
    
    context = {
        'previous_messages': previous,
        'current_user': 'user_a'
    }
    
    features = extractor.extract(current, context)[0]
    
    print(f"Message: {current}")
    print(f"Previous messages: {len(previous)}")
    print(f"\nExtracted features:")
    print(f"  Has target: {features.get('has_target')}")
    print(f"  Has direct address: {features.get('has_direct_address')}")
    print(f"  Escalation level: {features.get('escalation_level')}")
    print(f"  Total escalation score: {features.get('total_escalation_score')}")
    print(f"  Is response: {features.get('is_response')}")
