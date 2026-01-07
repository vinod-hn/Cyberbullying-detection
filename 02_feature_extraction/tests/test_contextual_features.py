# Test Contextual Features
"""
Unit tests for contextual features extraction module.
"""

import unittest
import sys
import os
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contextual_features import ContextualFeatures


class TestContextualFeatures(unittest.TestCase):
    """Test cases for ContextualFeatures class."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.extractor = ContextualFeatures()
        
        # Sample conversation
        base_time = datetime.now()
        cls.conversation = [
            {
                'message': 'hey everyone',
                'user_id': 'user1',
                'timestamp': (base_time - timedelta(hours=2)).isoformat(),
                'severity_score': 0.1
            },
            {
                'message': '@user2 you are so stupid',
                'user_id': 'user1',
                'timestamp': (base_time - timedelta(hours=1)).isoformat(),
                'severity_score': 0.6
            },
            {
                'message': 'leave me alone',
                'user_id': 'user2',
                'timestamp': (base_time - timedelta(minutes=30)).isoformat(),
                'severity_score': 0.2
            },
            {
                'message': '@user2 I will find you',
                'user_id': 'user1',
                'timestamp': base_time.isoformat(),
                'severity_score': 0.9
            }
        ]
    
    def test_initialization(self):
        """Test ContextualFeatures initialization."""
        extractor = ContextualFeatures()
        self.assertIsNotNone(extractor)
        self.assertIsNotNone(extractor.config)
    
    def test_position_features(self):
        """Test position feature extraction."""
        message = self.conversation[0]
        features = self.extractor.extract_position_features(
            message, self.conversation, position=0
        )
        
        self.assertIn('position_ratio', features)
        self.assertIn('is_first_message', features)
        self.assertEqual(features['position_ratio'], 0.0)
        self.assertTrue(features['is_first_message'])
    
    def test_last_message_position(self):
        """Test position features for last message."""
        message = self.conversation[-1]
        features = self.extractor.extract_position_features(
            message, self.conversation, position=len(self.conversation) - 1
        )
        
        self.assertEqual(features['position_ratio'], 1.0)
        self.assertFalse(features['is_first_message'])
    
    def test_mention_features(self):
        """Test mention feature extraction."""
        message = {'message': '@user2 @user3 you both are losers'}
        features = self.extractor.extract_mention_features(message)
        
        self.assertIn('mention_count', features)
        self.assertIn('mentions', features)
        self.assertEqual(features['mention_count'], 2)
        self.assertIn('user2', features['mentions'])
        self.assertIn('user3', features['mentions'])
    
    def test_no_mentions(self):
        """Test message with no mentions."""
        message = {'message': 'hello everyone'}
        features = self.extractor.extract_mention_features(message)
        
        self.assertEqual(features['mention_count'], 0)
        self.assertEqual(len(features['mentions']), 0)
    
    def test_response_features(self):
        """Test response feature extraction."""
        current_msg = self.conversation[2]
        prev_msg = self.conversation[1]
        
        features = self.extractor.extract_response_features(
            current_msg, prev_msg, thread_depth=2
        )
        
        self.assertIn('has_reply_context', features)
        self.assertIn('thread_depth', features)
        self.assertTrue(features['has_reply_context'])
        self.assertEqual(features['thread_depth'], 2)
    
    def test_escalation_features(self):
        """Test escalation feature extraction."""
        features = self.extractor.extract_escalation_features(
            self.conversation, current_position=3
        )
        
        self.assertIn('escalation_score', features)
        self.assertIn('is_escalating', features)
        # Severity goes from 0.1 to 0.9, should be escalating
        self.assertTrue(features['is_escalating'])
    
    def test_no_escalation(self):
        """Test non-escalating conversation."""
        calm_conversation = [
            {'message': 'hi', 'severity_score': 0.1},
            {'message': 'hello', 'severity_score': 0.1},
            {'message': 'how are you', 'severity_score': 0.05},
        ]
        
        features = self.extractor.extract_escalation_features(
            calm_conversation, current_position=2
        )
        
        self.assertFalse(features['is_escalating'])
    
    def test_temporal_features(self):
        """Test temporal feature extraction."""
        message = {
            'timestamp': datetime(2024, 1, 15, 23, 30, 0).isoformat()
        }
        
        features = self.extractor.extract_temporal_features(message)
        
        self.assertIn('hour_of_day', features)
        self.assertIn('is_late_night', features)
        self.assertEqual(features['hour_of_day'], 23)
        self.assertTrue(features['is_late_night'])
    
    def test_daytime_message(self):
        """Test daytime temporal features."""
        message = {
            'timestamp': datetime(2024, 1, 15, 14, 30, 0).isoformat()
        }
        
        features = self.extractor.extract_temporal_features(message)
        
        self.assertEqual(features['hour_of_day'], 14)
        self.assertFalse(features['is_late_night'])
    
    def test_all_features(self):
        """Test extracting all contextual features."""
        message = self.conversation[1]
        features = self.extractor.extract_all_features(
            message, self.conversation, position=1
        )
        
        # Should have position features
        self.assertIn('position_ratio', features)
        
        # Should have mention features
        self.assertIn('mention_count', features)
        
        # Should have escalation features
        self.assertIn('escalation_score', features)
    
    def test_numeric_features(self):
        """Test numeric feature extraction."""
        import numpy as np
        
        messages = [m['message'] for m in self.conversation]
        features = self.extractor.extract_numeric_features(
            messages, self.conversation
        )
        
        self.assertIsInstance(features, np.ndarray)
        self.assertEqual(len(features), len(messages))


if __name__ == '__main__':
    unittest.main(verbosity=2)
