# Conversation Threader
"""
ConversationThreader: Comprehensive conversation threading and context analysis
for cyberbullying detection in social media and messaging platforms.

Features:
- Message threading and grouping
- Conversation context tracking
- User interaction graph building
- Temporal pattern analysis
- Escalation detection
- Reply chain reconstruction
- Harassment campaign identification
- Target identification across threads
"""

import re
import os
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any, Union, Set
from collections import defaultdict, Counter
from dataclasses import dataclass, field
import hashlib


@dataclass
class Message:
    """
    Represents a single message in a conversation.
    
    Attributes:
        message_id: Unique identifier for the message
        sender_id: Identifier of the message sender
        content: Text content of the message
        timestamp: When the message was sent
        reply_to: ID of the message this is replying to (if any)
        mentions: List of user IDs mentioned in the message
        thread_id: Identifier of the conversation thread
        metadata: Additional message metadata
    """
    message_id: str
    sender_id: str
    content: str
    timestamp: datetime
    reply_to: Optional[str] = None
    mentions: List[str] = field(default_factory=list)
    thread_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary."""
        return {
            'message_id': self.message_id,
            'sender_id': self.sender_id,
            'content': self.content,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'reply_to': self.reply_to,
            'mentions': self.mentions,
            'thread_id': self.thread_id,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """Create Message from dictionary."""
        timestamp = data.get('timestamp')
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        elif timestamp is None:
            timestamp = datetime.now()
        
        return cls(
            message_id=data.get('message_id', ''),
            sender_id=data.get('sender_id', ''),
            content=data.get('content', ''),
            timestamp=timestamp,
            reply_to=data.get('reply_to'),
            mentions=data.get('mentions', []),
            thread_id=data.get('thread_id'),
            metadata=data.get('metadata', {})
        )


@dataclass
class ConversationThread:
    """
    Represents a conversation thread.
    
    Attributes:
        thread_id: Unique identifier for the thread
        messages: List of messages in the thread
        participants: Set of user IDs participating
        start_time: When the thread started
        end_time: When the last message was sent
        topic: Detected topic/subject of conversation
        metadata: Additional thread metadata
    """
    thread_id: str
    messages: List[Message] = field(default_factory=list)
    participants: Set[str] = field(default_factory=set)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    topic: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_message(self, message: Message) -> None:
        """Add a message to the thread."""
        self.messages.append(message)
        self.participants.add(message.sender_id)
        
        if self.start_time is None or message.timestamp < self.start_time:
            self.start_time = message.timestamp
        if self.end_time is None or message.timestamp > self.end_time:
            self.end_time = message.timestamp
    
    def get_duration(self) -> Optional[timedelta]:
        """Get the duration of the thread."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None
    
    def get_message_count(self) -> int:
        """Get total number of messages."""
        return len(self.messages)
    
    def get_participant_count(self) -> int:
        """Get number of unique participants."""
        return len(self.participants)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert thread to dictionary."""
        return {
            'thread_id': self.thread_id,
            'messages': [m.to_dict() for m in self.messages],
            'participants': list(self.participants),
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'topic': self.topic,
            'message_count': self.get_message_count(),
            'participant_count': self.get_participant_count(),
            'duration_seconds': self.get_duration().total_seconds() if self.get_duration() else None,
            'metadata': self.metadata
        }


class ConversationThreader:
    """
    Processor for conversation threading and context analysis.
    
    Handles message grouping, reply chain reconstruction, and
    detection of harassment patterns across conversations.
    
    Attributes:
        config: Configuration dictionary
        threads: Dictionary of conversation threads
        user_interactions: Graph of user interactions
        message_index: Index of all messages by ID
    """
    
    # Default time window for thread grouping (in minutes)
    DEFAULT_THREAD_WINDOW = 30
    
    # Maximum gap between messages in same thread (in minutes)
    DEFAULT_MAX_GAP = 60
    
    # Minimum messages for a valid thread
    DEFAULT_MIN_THREAD_MESSAGES = 2
    
    # Patterns indicating reply/mention
    MENTION_PATTERNS = [
        r'@(\w+)',                    # @username
        r'@\[([^\]]+)\]',             # @[Full Name]
        r'(?:^|\s)(\w+),\s',          # Username, at start
        r'(?:hey|hi|yo|arey|abe)\s+(\w+)',  # Greeting + name
    ]
    
    # Patterns for detecting targeting in Kannada-English
    TARGET_PATTERNS = {
        'direct_address': [
            r'^(\w+),\s',              # Name at start
            r'(?:nee|neenu|ninna)\b',  # Kannada "you/your"
            r'\byou\b',                # English "you"
            r'(?:nin|nimma)\b',        # Kannada possessive
        ],
        'accusation': [
            r'(?:nee|you)\s+(?:tumba|full|always)',
            r'(?:ninna|your)\s+\w+\s+(?:behavior|behaviour|nature|attitude)',
            r'(?:yaaru|everyone)\s+(?:kuda|also)',
        ],
        'group_reference': [
            r'group\s+nalli',           # "in the group"
            r'(?:yella|everyone)\s+',   # Everyone
            r'people\s+are\s+tired',    # Group frustration
        ]
    }
    
    # Escalation indicators
    ESCALATION_INDICATORS = {
        'threat_words': [
            'sucide', 'suicide', 'die', 'kill', 'saayi',
            'hodbitta', 'haaku', 'threat', 'warning'
        ],
        'intensity_words': [
            'tumba', 'full', 'sakkat', 'very', 'totally',
            'always', 'never', 'yella dina', 'every day'
        ],
        'group_pressure': [
            'everyone', 'yaaru kuda', 'people are',
            'group nalli', 'all of us', 'nobody likes'
        ]
    }
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        data_path: Optional[str] = None
    ):
        """
        Initialize ConversationThreader.
        
        Args:
            config: Optional configuration dictionary
            data_path: Path to data directory for loading context
        """
        self.config = config or self._default_config()
        self.data_path = data_path or self._get_default_data_path()
        
        # Thread storage
        self.threads: Dict[str, ConversationThread] = {}
        
        # Message index for quick lookup
        self.message_index: Dict[str, Message] = {}
        
        # User interaction tracking
        self.user_interactions: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        
        # Target tracking (who gets targeted)
        self.target_counts: Dict[str, int] = defaultdict(int)
        
        # Sender tracking (who sends hostile messages)
        self.sender_hostility: Dict[str, int] = defaultdict(int)
        
        # Compile patterns
        self._compile_patterns()
    
    def _default_config(self) -> Dict[str, Any]:
        """Return default configuration."""
        return {
            'thread_window_minutes': self.DEFAULT_THREAD_WINDOW,
            'max_gap_minutes': self.DEFAULT_MAX_GAP,
            'min_thread_messages': self.DEFAULT_MIN_THREAD_MESSAGES,
            'detect_mentions': True,
            'detect_replies': True,
            'track_escalation': True,
            'build_interaction_graph': True,
            'identify_targets': True,
            'identify_harassers': True,
            'temporal_analysis': True,
        }
    
    def _get_default_data_path(self) -> str:
        """Get default data directory path."""
        return os.path.join(
            os.path.dirname(__file__),
            '..', '00_data'
        )
    
    def _compile_patterns(self) -> None:
        """Compile regex patterns for efficiency."""
        # Mention patterns
        self.mention_patterns_compiled = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in self.MENTION_PATTERNS
        ]
        
        # Target patterns
        self.target_patterns_compiled = {
            category: [re.compile(p, re.IGNORECASE) for p in patterns]
            for category, patterns in self.TARGET_PATTERNS.items()
        }
        
        # Hashtag ID pattern (dataset format)
        self.hashtag_id_pattern = re.compile(r'#[a-f0-9]{4}\b')
        
        # Anonymized username pattern
        self.anon_username_pattern = re.compile(r'^[A-Z][a-z]+(?:[A-Z][a-z]+)*$')
        
        # Timestamp pattern for parsing
        self.timestamp_pattern = re.compile(
            r'(\d{1,2}[:/]\d{2}(?:[:/]\d{2})?(?:\s*[AP]M)?)',
            re.IGNORECASE
        )
    
    def create_message(
        self,
        content: str,
        sender_id: str,
        message_id: Optional[str] = None,
        timestamp: Optional[datetime] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Message:
        """
        Create a Message object from raw data.
        
        Args:
            content: Message text content
            sender_id: Identifier of sender
            message_id: Optional message ID (auto-generated if not provided)
            timestamp: Optional timestamp (current time if not provided)
            reply_to: Optional ID of message being replied to
            metadata: Optional additional metadata
            
        Returns:
            Message object
        """
        # Generate message ID if not provided
        if not message_id:
            hash_input = f"{sender_id}:{content}:{timestamp or datetime.now()}"
            message_id = hashlib.md5(hash_input.encode()).hexdigest()[:12]
        
        # Extract mentions from content
        mentions = self._extract_mentions(content)
        
        # Extract sender from content if available (dataset format)
        extracted_sender = self._extract_sender_from_content(content)
        if extracted_sender and not sender_id:
            sender_id = extracted_sender
        
        return Message(
            message_id=message_id,
            sender_id=sender_id,
            content=content,
            timestamp=timestamp or datetime.now(),
            reply_to=reply_to,
            mentions=mentions,
            metadata=metadata or {}
        )
    
    def _extract_mentions(self, content: str) -> List[str]:
        """Extract mentioned usernames from content."""
        mentions = []
        
        for pattern in self.mention_patterns_compiled:
            matches = pattern.findall(content)
            mentions.extend(matches)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_mentions = []
        for m in mentions:
            if m.lower() not in seen:
                seen.add(m.lower())
                unique_mentions.append(m)
        
        return unique_mentions
    
    def _extract_sender_from_content(self, content: str) -> Optional[str]:
        """
        Extract sender name from dataset format.
        Dataset format often has: "Username, message text. #hash"
        """
        # Check for comma-separated name at start
        match = re.match(r'^([A-Z][a-z]+(?:[a-z]+)*),\s', content)
        if match:
            return match.group(1)
        
        # Check for name after certain patterns
        match = re.search(r',\s+([A-Z][a-z]+(?:[a-z]+)*)\.\s*#', content)
        if match:
            return match.group(1)
        
        return None
    
    def _extract_target_from_content(self, content: str) -> Optional[str]:
        """
        Extract target (victim) from content.
        Identifies who is being addressed/attacked.
        """
        # Pattern: "Username, message" - Username is target
        match = re.match(r'^([A-Z][a-z]+(?:[a-z]+)*),\s', content)
        if match:
            return match.group(1)
        
        # Pattern: "message, Username. #hash" - Username is target
        match = re.search(r',\s+([A-Z][a-z]+(?:[a-z]+)*)\.?\s*#[a-f0-9]{4}', content)
        if match:
            return match.group(1)
        
        # Pattern: addressing in middle
        match = re.search(r'(?:nee|you)\s+(\w+)', content, re.IGNORECASE)
        if match and self.anon_username_pattern.match(match.group(1)):
            return match.group(1)
        
        return None
    
    def add_message(
        self,
        message: Union[Message, Dict[str, Any], str],
        sender_id: Optional[str] = None,
        thread_id: Optional[str] = None,
        timestamp: Optional[datetime] = None
    ) -> Message:
        """
        Add a message to the threader.
        
        Args:
            message: Message object, dictionary, or raw text
            sender_id: Sender ID (required if message is string)
            thread_id: Optional thread ID to add to
            timestamp: Optional timestamp
            
        Returns:
            The added Message object
        """
        # Convert to Message object if needed
        if isinstance(message, str):
            message = self.create_message(
                content=message,
                sender_id=sender_id or 'unknown',
                timestamp=timestamp
            )
        elif isinstance(message, dict):
            message = Message.from_dict(message)
        
        # Add to message index
        self.message_index[message.message_id] = message
        
        # Determine thread
        if thread_id:
            message.thread_id = thread_id
        elif message.thread_id is None:
            message.thread_id = self._determine_thread(message)
        
        # Add to or create thread
        if message.thread_id not in self.threads:
            self.threads[message.thread_id] = ConversationThread(
                thread_id=message.thread_id
            )
        
        self.threads[message.thread_id].add_message(message)
        
        # Track interactions
        if self.config.get('build_interaction_graph', True):
            self._track_interaction(message)
        
        # Track targeting
        if self.config.get('identify_targets', True):
            self._track_targeting(message)
        
        return message
    
    def _determine_thread(self, message: Message) -> str:
        """
        Determine which thread a message belongs to.
        
        Uses time-based grouping and reply chain analysis.
        """
        # If it's a reply, use the parent's thread
        if message.reply_to and message.reply_to in self.message_index:
            parent = self.message_index[message.reply_to]
            if parent.thread_id:
                return parent.thread_id
        
        # Find recent threads within time window
        window_minutes = self.config.get('thread_window_minutes', self.DEFAULT_THREAD_WINDOW)
        window = timedelta(minutes=window_minutes)
        
        best_thread = None
        min_time_diff = None
        
        for thread_id, thread in self.threads.items():
            if thread.end_time and message.timestamp:
                time_diff = abs((message.timestamp - thread.end_time).total_seconds())
                
                # Check if within window and from same conversation context
                if time_diff <= window.total_seconds():
                    # Check for participant overlap or mentions
                    if (message.sender_id in thread.participants or
                        any(m in thread.participants for m in message.mentions)):
                        if min_time_diff is None or time_diff < min_time_diff:
                            min_time_diff = time_diff
                            best_thread = thread_id
        
        if best_thread:
            return best_thread
        
        # Create new thread
        return self._generate_thread_id(message)
    
    def _generate_thread_id(self, message: Message) -> str:
        """Generate a unique thread ID."""
        hash_input = f"{message.sender_id}:{message.timestamp}:{message.content[:20]}"
        return f"thread_{hashlib.md5(hash_input.encode()).hexdigest()[:8]}"
    
    def _track_interaction(self, message: Message) -> None:
        """Track user-to-user interactions."""
        sender = message.sender_id
        
        # Track interactions with mentioned users
        for mention in message.mentions:
            self.user_interactions[sender][mention] += 1
        
        # Track reply interactions
        if message.reply_to and message.reply_to in self.message_index:
            replied_to = self.message_index[message.reply_to].sender_id
            self.user_interactions[sender][replied_to] += 1
    
    def _track_targeting(self, message: Message) -> None:
        """Track who is being targeted in messages."""
        target = self._extract_target_from_content(message.content)
        
        if target:
            self.target_counts[target] += 1
            
            # Check for hostile content
            if self._is_hostile_message(message.content):
                self.sender_hostility[message.sender_id] += 1
    
    def _is_hostile_message(self, content: str) -> bool:
        """Check if message content appears hostile."""
        content_lower = content.lower()
        
        # Check for hostile indicators
        hostile_words = [
            'thotha', 'gaandu', 'gandu', 'singri', 'dagarina',
            'bosodina', 'hucchadana', 'idiot', 'stupid', 'fool',
            'loser', 'hate', 'worst', 'die', 'kill'
        ]
        
        for word in hostile_words:
            if word in content_lower:
                return True
        
        return False
    
    def process_conversation(
        self,
        messages: List[Union[Message, Dict[str, Any], str]],
        context_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a full conversation for analysis.
        
        Args:
            messages: List of messages to process
            context_id: Optional context identifier
            
        Returns:
            Dictionary with conversation analysis
        """
        processed_messages = []
        
        for i, msg in enumerate(messages):
            if isinstance(msg, str):
                processed = self.add_message(
                    msg,
                    sender_id=f"user_{i}",
                    timestamp=datetime.now() + timedelta(seconds=i * 30)
                )
            else:
                processed = self.add_message(msg)
            processed_messages.append(processed)
        
        # Analyze the conversation
        analysis = self.analyze_threads()
        analysis['processed_messages'] = [m.to_dict() for m in processed_messages]
        analysis['context_id'] = context_id
        
        return analysis
    
    def analyze_threads(self) -> Dict[str, Any]:
        """
        Analyze all threads for patterns.
        
        Returns:
            Dictionary with thread analysis
        """
        thread_summaries = []
        
        for thread_id, thread in self.threads.items():
            summary = self._analyze_thread(thread)
            thread_summaries.append(summary)
        
        # Overall statistics
        total_messages = sum(t.get_message_count() for t in self.threads.values())
        total_participants = len(set().union(*[t.participants for t in self.threads.values()])) if self.threads else 0
        
        return {
            'thread_count': len(self.threads),
            'total_messages': total_messages,
            'total_participants': total_participants,
            'threads': thread_summaries,
            'escalation_detected': self._detect_overall_escalation(),
            'top_targets': self._get_top_targets(),
            'top_hostile_senders': self._get_top_hostile_senders(),
            'interaction_density': self._calculate_interaction_density()
        }
    
    def _analyze_thread(self, thread: ConversationThread) -> Dict[str, Any]:
        """Analyze a single thread."""
        messages = thread.messages
        
        # Message frequency analysis
        sender_counts = Counter(m.sender_id for m in messages)
        
        # Temporal analysis
        if len(messages) >= 2:
            timestamps = sorted(m.timestamp for m in messages if m.timestamp)
            if len(timestamps) >= 2:
                gaps = [(timestamps[i+1] - timestamps[i]).total_seconds() 
                        for i in range(len(timestamps)-1)]
                avg_gap = sum(gaps) / len(gaps) if gaps else 0
            else:
                avg_gap = 0
        else:
            avg_gap = 0
        
        # Escalation analysis
        escalation = self._detect_thread_escalation(messages)
        
        # Target analysis
        targets = [self._extract_target_from_content(m.content) for m in messages]
        targets = [t for t in targets if t]
        target_counts = Counter(targets)
        
        return {
            'thread_id': thread.thread_id,
            'message_count': len(messages),
            'participant_count': len(thread.participants),
            'participants': list(thread.participants),
            'sender_distribution': dict(sender_counts),
            'start_time': thread.start_time.isoformat() if thread.start_time else None,
            'end_time': thread.end_time.isoformat() if thread.end_time else None,
            'duration_seconds': thread.get_duration().total_seconds() if thread.get_duration() else 0,
            'avg_message_gap_seconds': avg_gap,
            'escalation_detected': escalation['detected'],
            'escalation_level': escalation['level'],
            'escalation_indicators': escalation['indicators'],
            'targets': dict(target_counts),
            'primary_target': max(target_counts, key=target_counts.get) if target_counts else None
        }
    
    def _detect_thread_escalation(self, messages: List[Message]) -> Dict[str, Any]:
        """Detect escalation within a thread."""
        if len(messages) < 2:
            return {'detected': False, 'level': 'none', 'indicators': []}
        
        indicators = []
        intensity_scores = []
        
        for i, msg in enumerate(messages):
            content_lower = msg.content.lower()
            score = 0
            
            # Check threat words
            for word in self.ESCALATION_INDICATORS['threat_words']:
                if word in content_lower:
                    indicators.append(f"threat_word:{word}")
                    score += 3
            
            # Check intensity words
            for word in self.ESCALATION_INDICATORS['intensity_words']:
                if word in content_lower:
                    score += 1
            
            # Check group pressure
            for phrase in self.ESCALATION_INDICATORS['group_pressure']:
                if phrase in content_lower:
                    indicators.append(f"group_pressure:{phrase}")
                    score += 2
            
            intensity_scores.append((i, score))
        
        # Check for escalation pattern (increasing intensity)
        escalating = False
        if len(intensity_scores) >= 3:
            recent_avg = sum(s for _, s in intensity_scores[-3:]) / 3
            early_avg = sum(s for _, s in intensity_scores[:3]) / 3
            if recent_avg > early_avg * 1.5:
                escalating = True
        
        # Determine level
        max_score = max(s for _, s in intensity_scores) if intensity_scores else 0
        
        if max_score >= 5 or escalating:
            level = 'high'
            detected = True
        elif max_score >= 3:
            level = 'medium'
            detected = True
        elif max_score >= 1:
            level = 'low'
            detected = True
        else:
            level = 'none'
            detected = False
        
        return {
            'detected': detected,
            'level': level,
            'indicators': list(set(indicators)),
            'escalating_pattern': escalating
        }
    
    def _detect_overall_escalation(self) -> Dict[str, Any]:
        """Detect escalation across all threads."""
        all_escalation = []
        
        for thread in self.threads.values():
            esc = self._detect_thread_escalation(thread.messages)
            if esc['detected']:
                all_escalation.append({
                    'thread_id': thread.thread_id,
                    **esc
                })
        
        if not all_escalation:
            return {'detected': False, 'threads': [], 'overall_level': 'none'}
        
        # Determine overall level
        levels = [e['level'] for e in all_escalation]
        if 'high' in levels:
            overall = 'high'
        elif 'medium' in levels:
            overall = 'medium'
        else:
            overall = 'low'
        
        return {
            'detected': True,
            'threads': all_escalation,
            'overall_level': overall,
            'escalating_thread_count': len(all_escalation)
        }
    
    def _get_top_targets(self, n: int = 5) -> List[Dict[str, Any]]:
        """Get the most frequently targeted users."""
        sorted_targets = sorted(
            self.target_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:n]
        
        return [
            {'user_id': user, 'target_count': count}
            for user, count in sorted_targets
        ]
    
    def _get_top_hostile_senders(self, n: int = 5) -> List[Dict[str, Any]]:
        """Get users sending most hostile messages."""
        sorted_senders = sorted(
            self.sender_hostility.items(),
            key=lambda x: x[1],
            reverse=True
        )[:n]
        
        return [
            {'user_id': user, 'hostile_message_count': count}
            for user, count in sorted_senders
        ]
    
    def _calculate_interaction_density(self) -> float:
        """Calculate the density of user interactions."""
        if not self.user_interactions:
            return 0.0
        
        total_interactions = sum(
            sum(targets.values())
            for targets in self.user_interactions.values()
        )
        
        unique_users = len(set(self.user_interactions.keys()).union(
            *[set(targets.keys()) for targets in self.user_interactions.values()]
        ))
        
        if unique_users < 2:
            return 0.0
        
        max_interactions = unique_users * (unique_users - 1)
        return total_interactions / max_interactions if max_interactions > 0 else 0.0
    
    def get_user_profile(self, user_id: str) -> Dict[str, Any]:
        """
        Get profile analysis for a specific user.
        
        Args:
            user_id: User identifier
            
        Returns:
            User profile with interaction patterns
        """
        messages_sent = []
        messages_received = []
        threads_participated = []
        
        for thread in self.threads.values():
            user_in_thread = False
            for msg in thread.messages:
                if msg.sender_id == user_id:
                    messages_sent.append(msg)
                    user_in_thread = True
                elif user_id in msg.mentions:
                    messages_received.append(msg)
                    user_in_thread = True
            
            if user_in_thread:
                threads_participated.append(thread.thread_id)
        
        # Interaction analysis
        interactions_with = dict(self.user_interactions.get(user_id, {}))
        
        # Check if user is a target
        target_count = self.target_counts.get(user_id, 0)
        
        # Check if user sends hostile messages
        hostile_count = self.sender_hostility.get(user_id, 0)
        
        # Determine role
        if hostile_count > target_count and hostile_count > 3:
            role = 'potential_harasser'
        elif target_count > hostile_count and target_count > 3:
            role = 'potential_victim'
        elif target_count > 0 and hostile_count > 0:
            role = 'mixed_involvement'
        else:
            role = 'regular_participant'
        
        return {
            'user_id': user_id,
            'messages_sent_count': len(messages_sent),
            'messages_received_count': len(messages_received),
            'threads_participated': threads_participated,
            'thread_count': len(threads_participated),
            'interactions_with': interactions_with,
            'times_targeted': target_count,
            'hostile_messages_sent': hostile_count,
            'role_assessment': role
        }
    
    def get_conversation_context(
        self,
        message: Union[Message, str],
        context_window: int = 5
    ) -> Dict[str, Any]:
        """
        Get conversation context around a specific message.
        
        Args:
            message: Message to get context for
            context_window: Number of messages before/after
            
        Returns:
            Context including surrounding messages
        """
        if isinstance(message, str):
            # Find message by ID or content
            found = None
            for msg in self.message_index.values():
                if msg.message_id == message or message in msg.content:
                    found = msg
                    break
            if not found:
                return {'error': 'Message not found', 'context': []}
            message = found
        
        if message.thread_id not in self.threads:
            return {'error': 'Thread not found', 'context': []}
        
        thread = self.threads[message.thread_id]
        messages = sorted(thread.messages, key=lambda m: m.timestamp)
        
        # Find message index
        msg_idx = None
        for i, m in enumerate(messages):
            if m.message_id == message.message_id:
                msg_idx = i
                break
        
        if msg_idx is None:
            return {'error': 'Message not found in thread', 'context': []}
        
        # Get context window
        start = max(0, msg_idx - context_window)
        end = min(len(messages), msg_idx + context_window + 1)
        
        context_messages = messages[start:end]
        
        # Identify relationships
        context = []
        for i, m in enumerate(context_messages):
            rel = 'context'
            if m.message_id == message.message_id:
                rel = 'target'
            elif i < msg_idx - start:
                rel = 'before'
            else:
                rel = 'after'
            
            context.append({
                **m.to_dict(),
                'relationship': rel,
                'position': i - (msg_idx - start)
            })
        
        return {
            'target_message': message.to_dict(),
            'thread_id': message.thread_id,
            'context': context,
            'context_size': len(context_messages),
            'messages_before': msg_idx - start,
            'messages_after': end - msg_idx - 1
        }
    
    def detect_harassment_campaign(
        self,
        target_user: str,
        time_window_hours: int = 24
    ) -> Dict[str, Any]:
        """
        Detect if there's a harassment campaign against a user.
        
        Args:
            target_user: User ID to check
            time_window_hours: Time window to analyze
            
        Returns:
            Campaign analysis results
        """
        targeted_messages = []
        senders = Counter()
        threads_involved = set()
        
        window = timedelta(hours=time_window_hours)
        now = datetime.now()
        
        for thread in self.threads.values():
            for msg in thread.messages:
                # Check if message targets the user
                target = self._extract_target_from_content(msg.content)
                if target == target_user:
                    # Check time window
                    if msg.timestamp and (now - msg.timestamp) <= window:
                        targeted_messages.append(msg)
                        senders[msg.sender_id] += 1
                        threads_involved.add(thread.thread_id)
        
        # Analyze for campaign characteristics
        is_campaign = False
        campaign_type = 'none'
        severity = 'none'
        
        if len(targeted_messages) >= 5:
            if len(senders) >= 3:
                # Multiple senders = coordinated campaign
                is_campaign = True
                campaign_type = 'coordinated'
                severity = 'high'
            elif len(senders) >= 1:
                # Single sender = individual harassment
                is_campaign = True
                campaign_type = 'individual'
                severity = 'medium'
        elif len(targeted_messages) >= 3:
            is_campaign = True
            campaign_type = 'emerging'
            severity = 'low'
        
        # Check for threat escalation
        has_threats = False
        for msg in targeted_messages:
            content_lower = msg.content.lower()
            for word in self.ESCALATION_INDICATORS['threat_words']:
                if word in content_lower:
                    has_threats = True
                    severity = 'critical'
                    break
        
        return {
            'target_user': target_user,
            'is_campaign_detected': is_campaign,
            'campaign_type': campaign_type,
            'severity': severity,
            'message_count': len(targeted_messages),
            'unique_senders': len(senders),
            'sender_breakdown': dict(senders),
            'threads_involved': list(threads_involved),
            'thread_count': len(threads_involved),
            'has_threats': has_threats,
            'time_window_hours': time_window_hours,
            'messages': [m.to_dict() for m in targeted_messages[:10]]  # Limit for response size
        }
    
    def process_dataset_row(
        self,
        row: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process a single row from the dataset format.
        
        Expected format: message, label, target_type, severity_score, 
                        severity_level, context
        
        Args:
            row: Dictionary with dataset fields
            
        Returns:
            Processed message with context analysis
        """
        content = row.get('message', '')
        label = row.get('label', 'Unknown')
        target_type = row.get('target_type', 'Unknown')
        severity = row.get('severity_level', 'unknown')
        context = row.get('context', 'unknown')
        
        # Extract components
        target = self._extract_target_from_content(content)
        
        # Clean message (remove hashtag ID)
        clean_content = self.hashtag_id_pattern.sub('', content).strip()
        
        # Create message
        message = self.create_message(
            content=clean_content,
            sender_id='dataset',
            metadata={
                'label': label,
                'target_type': target_type,
                'severity_level': severity,
                'original_context': context
            }
        )
        
        # Analyze for patterns
        targeting_analysis = self._analyze_targeting(content)
        
        return {
            'message': message.to_dict(),
            'target_identified': target,
            'targeting_analysis': targeting_analysis,
            'label': label,
            'severity': severity,
            'is_hostile': self._is_hostile_message(content)
        }
    
    def _analyze_targeting(self, content: str) -> Dict[str, Any]:
        """Analyze targeting patterns in content."""
        patterns_found = defaultdict(list)
        
        for category, patterns in self.target_patterns_compiled.items():
            for pattern in patterns:
                matches = pattern.findall(content)
                if matches:
                    patterns_found[category].extend(matches)
        
        return {
            'patterns_found': dict(patterns_found),
            'has_direct_address': bool(patterns_found.get('direct_address')),
            'has_accusation': bool(patterns_found.get('accusation')),
            'has_group_reference': bool(patterns_found.get('group_reference')),
            'targeting_score': (
                len(patterns_found.get('direct_address', [])) * 1 +
                len(patterns_found.get('accusation', [])) * 2 +
                len(patterns_found.get('group_reference', [])) * 1.5
            )
        }
    
    def get_thread_by_id(self, thread_id: str) -> Optional[ConversationThread]:
        """Get a thread by its ID."""
        return self.threads.get(thread_id)
    
    def get_all_threads(self) -> List[ConversationThread]:
        """Get all threads."""
        return list(self.threads.values())
    
    def get_thread_count(self) -> int:
        """Get total number of threads."""
        return len(self.threads)
    
    def get_message_count(self) -> int:
        """Get total number of messages."""
        return len(self.message_index)
    
    def clear(self) -> None:
        """Clear all stored data."""
        self.threads.clear()
        self.message_index.clear()
        self.user_interactions.clear()
        self.target_counts.clear()
        self.sender_hostility.clear()
    
    def export_threads(self, output_path: str) -> bool:
        """
        Export threads to JSON file.
        
        Args:
            output_path: Path to output file
            
        Returns:
            True if successful
        """
        try:
            data = {
                'threads': [t.to_dict() for t in self.threads.values()],
                'statistics': self.analyze_threads()
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            return True
        except Exception as e:
            return False
    
    def import_threads(self, input_path: str) -> bool:
        """
        Import threads from JSON file.
        
        Args:
            input_path: Path to input file
            
        Returns:
            True if successful
        """
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for thread_data in data.get('threads', []):
                thread = ConversationThread(
                    thread_id=thread_data['thread_id'],
                    topic=thread_data.get('topic')
                )
                
                for msg_data in thread_data.get('messages', []):
                    message = Message.from_dict(msg_data)
                    thread.add_message(message)
                    self.message_index[message.message_id] = message
                
                self.threads[thread.thread_id] = thread
            
            return True
        except Exception as e:
            return False
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"ConversationThreader("
            f"threads={len(self.threads)}, "
            f"messages={len(self.message_index)}, "
            f"users={len(self.user_interactions)})"
        )
    
    def __len__(self) -> int:
        """Return number of threads."""
        return len(self.threads)
