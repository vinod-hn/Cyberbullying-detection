# Behavioral Features
"""
BehavioralFeatures: User behavior pattern extraction for cyberbullying detection.
Analyzes temporal patterns, interaction history, and behavioral indicators.
Optimized for Kannada-English code-mixed text environments.
"""

import re
import logging
from typing import List, Dict, Optional, Any, Union, Set
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import numpy as np

logger = logging.getLogger(__name__)


class BehavioralFeatures:
    """
    Behavioral feature extractor for cyberbullying detection.
    
    Extracts features based on:
    - User messaging patterns
    - Temporal behavior
    - Target selection patterns
    - Harassment campaign indicators
    - Repeat offense patterns
    
    Attributes:
        config: Configuration dictionary
        user_history: Storage for user behavior history
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize BehavioralFeatures.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or self._default_config()
        
        # User history storage
        self.user_history: Dict[str, Dict[str, Any]] = {}
        self.target_history: Dict[str, Dict[str, Any]] = {}
    
    def _default_config(self) -> Dict[str, Any]:
        """Return default configuration."""
        return {
            'include_temporal': True,
            'include_targeting': True,
            'include_frequency': True,
            'include_persistence': True,
            'history_window_days': 30,
            'min_messages_for_pattern': 3
        }
    
    # =========================================================================
    # User Profile Building
    # =========================================================================
    def build_user_profile(
        self,
        user_id: str,
        messages: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Build a behavioral profile for a user.
        
        Args:
            user_id: User identifier
            messages: List of messages from this user
            
        Returns:
            User behavioral profile
        """
        if not messages:
            return self._empty_profile(user_id)
        
        profile = {
            'user_id': user_id,
            'total_messages': len(messages),
            'message_texts': [m.get('text', '') for m in messages]
        }
        
        # Temporal patterns
        timestamps = [m.get('timestamp') for m in messages if m.get('timestamp')]
        if timestamps:
            profile.update(self._analyze_temporal_patterns(timestamps))
        
        # Content patterns
        texts = [m.get('text', '') for m in messages]
        profile.update(self._analyze_content_patterns(texts))
        
        # Target patterns
        targets = [m.get('target') for m in messages if m.get('target')]
        profile.update(self._analyze_target_patterns(targets))
        
        # Severity patterns
        severities = [m.get('severity', 'low') for m in messages]
        profile.update(self._analyze_severity_patterns(severities))
        
        # Store in history
        self.user_history[user_id] = profile
        
        return profile
    
    def _empty_profile(self, user_id: str) -> Dict[str, Any]:
        """Create empty profile for user with no messages."""
        return {
            'user_id': user_id,
            'total_messages': 0,
            'is_new_user': True,
            'risk_score': 0.0
        }
    
    def _analyze_temporal_patterns(
        self,
        timestamps: List[datetime]
    ) -> Dict[str, Any]:
        """Analyze temporal patterns from timestamps."""
        if not timestamps:
            return {}
        
        timestamps = sorted(timestamps)
        
        # Activity hours
        hours = [ts.hour for ts in timestamps]
        hour_counts = Counter(hours)
        peak_hour = max(hour_counts, key=hour_counts.get)
        
        # Day patterns
        days = [ts.weekday() for ts in timestamps]
        day_counts = Counter(days)
        
        # Time gaps between messages
        gaps = []
        for i in range(1, len(timestamps)):
            gap = (timestamps[i] - timestamps[i-1]).total_seconds()
            gaps.append(gap)
        
        avg_gap = np.mean(gaps) if gaps else 0
        min_gap = min(gaps) if gaps else 0
        
        # Burst detection (multiple messages in short time)
        burst_count = sum(1 for g in gaps if g < 60)  # Less than 1 minute
        
        return {
            'peak_activity_hour': peak_hour,
            'activity_hours': list(hour_counts.keys()),
            'is_night_active': any(h in [22, 23, 0, 1, 2, 3, 4, 5] for h in hours),
            'avg_message_gap_seconds': round(avg_gap, 2),
            'min_message_gap_seconds': round(min_gap, 2),
            'burst_message_count': burst_count,
            'has_burst_behavior': burst_count > 2,
            'weekend_ratio': sum(1 for d in days if d >= 5) / len(days)
        }
    
    def _analyze_content_patterns(self, texts: List[str]) -> Dict[str, Any]:
        """Analyze content patterns from message texts."""
        if not texts:
            return {}
        
        # Message lengths
        lengths = [len(t.split()) for t in texts]
        avg_length = np.mean(lengths)
        
        # Aggressive word frequency
        aggressive_words = [
            'stupid', 'idiot', 'hate', 'kill', 'die', 'ugly', 'fat', 'loser',
            'irritating', 'annoying', 'thotha', 'singri', 'dagarina', 'saayi'
        ]
        
        aggressive_count = 0
        for text in texts:
            text_lower = text.lower()
            aggressive_count += sum(1 for w in aggressive_words if w in text_lower)
        
        aggression_ratio = aggressive_count / len(texts)
        
        # Emoji usage
        emoji_pattern = re.compile(
            r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F900-\U0001F9FF]+'
        )
        emoji_count = sum(len(emoji_pattern.findall(t)) for t in texts)
        
        # Caps usage
        caps_chars = sum(sum(1 for c in t if c.isupper()) for t in texts)
        total_chars = sum(len(t) for t in texts)
        caps_ratio = caps_chars / max(total_chars, 1)
        
        return {
            'avg_message_length': round(avg_length, 2),
            'total_aggressive_words': aggressive_count,
            'aggression_per_message': round(aggression_ratio, 3),
            'total_emoji_count': emoji_count,
            'emoji_per_message': round(emoji_count / len(texts), 2),
            'caps_ratio': round(caps_ratio, 3),
            'is_aggressive_user': aggression_ratio > 0.3
        }
    
    def _analyze_target_patterns(self, targets: List[str]) -> Dict[str, Any]:
        """Analyze target selection patterns."""
        if not targets:
            return {
                'unique_targets': 0,
                'is_focused_targeting': False
            }
        
        target_counts = Counter(targets)
        unique_targets = len(target_counts)
        total_targeted = len(targets)
        
        # Focused targeting (repeatedly targeting same person)
        max_target_count = max(target_counts.values())
        focus_ratio = max_target_count / total_targeted
        
        return {
            'unique_targets': unique_targets,
            'total_targeted_messages': total_targeted,
            'max_target_count': max_target_count,
            'targeting_focus_ratio': round(focus_ratio, 3),
            'is_focused_targeting': focus_ratio > 0.5,
            'primary_target': max(target_counts, key=target_counts.get) if target_counts else None
        }
    
    def _analyze_severity_patterns(self, severities: List[str]) -> Dict[str, Any]:
        """Analyze severity patterns."""
        if not severities:
            return {}
        
        severity_map = {'low': 1, 'medium': 2, 'high': 3, 'critical': 4}
        severity_scores = [severity_map.get(s.lower(), 1) for s in severities]
        
        avg_severity = np.mean(severity_scores)
        max_severity = max(severity_scores)
        
        # Trend detection
        if len(severity_scores) >= 3:
            first_half = np.mean(severity_scores[:len(severity_scores)//2])
            second_half = np.mean(severity_scores[len(severity_scores)//2:])
            trend = 'escalating' if second_half > first_half else 'de-escalating' if second_half < first_half else 'stable'
        else:
            trend = 'unknown'
        
        return {
            'avg_severity_score': round(avg_severity, 2),
            'max_severity_score': max_severity,
            'high_severity_count': sum(1 for s in severity_scores if s >= 3),
            'severity_trend': trend,
            'is_high_risk': avg_severity >= 2.5
        }
    
    # =========================================================================
    # Feature Extraction
    # =========================================================================
    def extract_frequency_features(
        self,
        user_id: str,
        time_window: Optional[timedelta] = None
    ) -> Dict[str, Any]:
        """
        Extract message frequency features for a user.
        
        Args:
            user_id: User identifier
            time_window: Time window to consider
            
        Returns:
            Frequency features dictionary
        """
        profile = self.user_history.get(user_id, {})
        
        total_messages = profile.get('total_messages', 0)
        
        if total_messages == 0:
            return {
                'message_count': 0,
                'messages_per_day': 0,
                'is_high_volume': False
            }
        
        # Calculate frequency metrics
        avg_gap = profile.get('avg_message_gap_seconds', 0)
        
        if avg_gap > 0:
            messages_per_hour = 3600 / avg_gap
            messages_per_day = messages_per_hour * 24
        else:
            messages_per_day = total_messages  # Assume all in one day
        
        return {
            'message_count': total_messages,
            'messages_per_day': round(messages_per_day, 2),
            'is_high_volume': messages_per_day > 50,
            'burst_count': profile.get('burst_message_count', 0),
            'has_burst_behavior': profile.get('has_burst_behavior', False)
        }
    
    def extract_persistence_features(
        self,
        user_id: str,
        target_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extract persistence features (repeated behavior).
        
        Args:
            user_id: User identifier
            target_id: Optional specific target to check
            
        Returns:
            Persistence features dictionary
        """
        profile = self.user_history.get(user_id, {})
        
        features = {
            'total_incidents': profile.get('total_messages', 0),
            'unique_targets': profile.get('unique_targets', 0),
            'is_repeat_offender': profile.get('total_messages', 0) > 5,
        }
        
        # Target-specific persistence
        if target_id:
            max_target = profile.get('primary_target')
            features['targets_specific_user'] = max_target == target_id
            features['target_focus_ratio'] = profile.get('targeting_focus_ratio', 0)
        
        # Severity persistence
        features['has_high_severity_history'] = profile.get('is_high_risk', False)
        features['severity_trend'] = profile.get('severity_trend', 'unknown')
        
        return features
    
    def extract_harassment_campaign_features(
        self,
        messages: List[Dict[str, Any]],
        target_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Detect harassment campaign patterns.
        
        Args:
            messages: List of messages (potentially from multiple users)
            target_id: Target to check for campaign against
            
        Returns:
            Campaign detection features
        """
        if not messages:
            return {
                'is_campaign': False,
                'campaign_score': 0.0
            }
        
        # Group by user
        user_messages = defaultdict(list)
        for msg in messages:
            user = msg.get('user', 'unknown')
            user_messages[user].append(msg)
        
        # Multiple users targeting same person
        if target_id:
            targeting_users = set()
            for user, msgs in user_messages.items():
                for msg in msgs:
                    if msg.get('target') == target_id:
                        targeting_users.add(user)
            
            is_coordinated = len(targeting_users) > 2
        else:
            is_coordinated = False
        
        # Calculate campaign score
        campaign_indicators = 0
        
        # Multiple users
        if len(user_messages) > 2:
            campaign_indicators += 1
        
        # High frequency
        total_messages = len(messages)
        if total_messages > 10:
            campaign_indicators += 1
        
        # Similar content (copy-paste detection)
        texts = [m.get('text', '').lower() for m in messages]
        unique_ratio = len(set(texts)) / max(len(texts), 1)
        if unique_ratio < 0.7:  # Many similar messages
            campaign_indicators += 1
        
        # Short time span
        timestamps = [m.get('timestamp') for m in messages if m.get('timestamp')]
        if len(timestamps) >= 2:
            time_span = (max(timestamps) - min(timestamps)).total_seconds()
            if time_span < 3600 and total_messages > 5:  # Many messages in 1 hour
                campaign_indicators += 1
        
        campaign_score = campaign_indicators / 4.0
        
        return {
            'is_campaign': campaign_score >= 0.5,
            'campaign_score': round(campaign_score, 3),
            'unique_attackers': len(user_messages),
            'is_coordinated': is_coordinated,
            'content_similarity': round(1 - unique_ratio, 3),
            'message_frequency': total_messages
        }
    
    # =========================================================================
    # Risk Scoring
    # =========================================================================
    def calculate_user_risk_score(self, user_id: str) -> Dict[str, Any]:
        """
        Calculate overall risk score for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            Risk assessment dictionary
        """
        profile = self.user_history.get(user_id, {})
        
        if not profile or profile.get('total_messages', 0) == 0:
            return {
                'risk_score': 0.0,
                'risk_level': 'unknown',
                'risk_factors': []
            }
        
        risk_factors = []
        risk_score = 0.0
        
        # Aggression factor
        aggression = profile.get('aggression_per_message', 0)
        if aggression > 0.3:
            risk_factors.append('high_aggression')
            risk_score += 0.25
        elif aggression > 0.1:
            risk_factors.append('moderate_aggression')
            risk_score += 0.1
        
        # Targeting factor
        if profile.get('is_focused_targeting', False):
            risk_factors.append('focused_targeting')
            risk_score += 0.2
        
        # Persistence factor
        if profile.get('total_messages', 0) > 10:
            risk_factors.append('high_volume')
            risk_score += 0.15
        
        # Burst behavior
        if profile.get('has_burst_behavior', False):
            risk_factors.append('burst_behavior')
            risk_score += 0.15
        
        # Severity factor
        if profile.get('is_high_risk', False):
            risk_factors.append('high_severity_history')
            risk_score += 0.2
        
        # Escalation
        if profile.get('severity_trend') == 'escalating':
            risk_factors.append('escalating_behavior')
            risk_score += 0.15
        
        # Determine risk level
        if risk_score >= 0.7:
            risk_level = 'critical'
        elif risk_score >= 0.5:
            risk_level = 'high'
        elif risk_score >= 0.3:
            risk_level = 'medium'
        else:
            risk_level = 'low'
        
        return {
            'risk_score': round(min(risk_score, 1.0), 3),
            'risk_level': risk_level,
            'risk_factors': risk_factors,
            'requires_intervention': risk_level in ['high', 'critical']
        }
    
    # =========================================================================
    # Combined Extraction
    # =========================================================================
    def extract(
        self,
        texts: Union[str, List[str]],
        user_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract behavioral features.
        
        Args:
            texts: Text or list of texts
            user_id: Optional user identifier
            context: Optional context with user history
            
        Returns:
            List of feature dictionaries
        """
        if isinstance(texts, str):
            texts = [texts]
        
        results = []
        
        # Get or create user profile
        if user_id and user_id in self.user_history:
            profile = self.user_history[user_id]
        else:
            profile = {}
        
        for text in texts:
            features = {}
            
            # Add profile-based features
            features['has_user_history'] = bool(profile)
            features['user_message_count'] = profile.get('total_messages', 0)
            features['user_aggression_ratio'] = profile.get('aggression_per_message', 0)
            features['user_is_high_risk'] = profile.get('is_high_risk', False)
            
            # Current message analysis
            text_lower = text.lower()
            
            # Quick aggression check
            aggressive_words = ['stupid', 'idiot', 'hate', 'kill', 'die', 'ugly', 
                              'thotha', 'singri', 'irritating']
            aggression = sum(1 for w in aggressive_words if w in text_lower)
            features['current_aggression_count'] = aggression
            
            # Intensity indicators
            features['current_exclamation_count'] = text.count('!')
            features['current_caps_ratio'] = sum(1 for c in text if c.isupper()) / max(len(text), 1)
            
            # Risk indicators
            if user_id:
                risk = self.calculate_user_risk_score(user_id)
                features['user_risk_score'] = risk.get('risk_score', 0)
                features['user_risk_level'] = risk.get('risk_level', 'unknown')
            
            results.append(features)
        
        return results
    
    def extract_numeric_features(
        self,
        texts: Union[str, List[str]],
        user_id: Optional[str] = None
    ) -> np.ndarray:
        """
        Extract numeric behavioral features.
        
        Returns:
            Feature matrix (n_samples, n_features)
        """
        all_features = self.extract(texts, user_id)
        
        numeric_keys = [
            'user_message_count', 'user_aggression_ratio',
            'current_aggression_count', 'current_exclamation_count',
            'current_caps_ratio', 'user_risk_score'
        ]
        
        matrix = []
        for features in all_features:
            row = []
            for key in numeric_keys:
                val = features.get(key, 0)
                if isinstance(val, bool):
                    val = int(val)
                elif not isinstance(val, (int, float)):
                    val = 0
                row.append(val)
            matrix.append(row)
        
        return np.array(matrix)
    
    def get_feature_names(self) -> List[str]:
        """Get list of numeric feature names."""
        return [
            'user_message_count', 'user_aggression_ratio',
            'current_aggression_count', 'current_exclamation_count',
            'current_caps_ratio', 'user_risk_score'
        ]
    
    def __repr__(self) -> str:
        """String representation."""
        return f"BehavioralFeatures(users_tracked={len(self.user_history)})"


if __name__ == "__main__":
    print("BehavioralFeatures Test")
    print("=" * 50)
    
    extractor = BehavioralFeatures()
    
    # Simulate user history
    user_messages = [
        {'text': 'nee tumba irritating agthiya!', 'severity': 'medium'},
        {'text': 'Stop talking to me idiot', 'severity': 'high'},
        {'text': 'I hate you so much!!', 'severity': 'high'},
        {'text': 'You are such a loser', 'severity': 'medium'},
    ]
    
    # Build profile
    profile = extractor.build_user_profile('user_123', user_messages)
    
    print(f"User Profile:")
    print(f"  Total messages: {profile.get('total_messages')}")
    print(f"  Aggression ratio: {profile.get('aggression_per_message')}")
    print(f"  Is aggressive user: {profile.get('is_aggressive_user')}")
    
    # Calculate risk
    risk = extractor.calculate_user_risk_score('user_123')
    print(f"\nRisk Assessment:")
    print(f"  Risk score: {risk.get('risk_score')}")
    print(f"  Risk level: {risk.get('risk_level')}")
    print(f"  Risk factors: {risk.get('risk_factors')}")
