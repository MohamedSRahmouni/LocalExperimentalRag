"""
Conversation Memory Pipeline
LangChain-powered conversation memory management

New component (not in original code)
Replaces manual conversation_history[-6:] with proper memory management
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from langchain.memory import (
    ConversationBufferMemory,
    ConversationBufferWindowMemory,
    ConversationSummaryMemory
)
from langchain.schema import BaseMemory, HumanMessage, AIMessage

logger = logging.getLogger(__name__)


class ConversationMemoryManager:
    """
    Centralized conversation memory management
    
    New features:
        - Multiple memory strategies
        - Per-user session isolation
        - Automatic cleanup
        - Memory persistence (future)
        
    Replaces:
        - Manual conversation_history[-6:] in your RAGService
        - No persistent storage in original code
    """
    
    def __init__(
        self,
        memory_type: str = "buffer_window",
        window_size: int = 6,           # Same as your history[-6:]
        max_token_limit: int = 2000,
        return_messages: bool = True,
        save_context: bool = True
    ):
        """
        Initialize Memory Manager
        
        Args:
            memory_type: Type of memory
                - 'buffer': Keep all history (unlimited)
                - 'buffer_window': Keep last N messages (same as your code)
                - 'summary': Summarize old messages (advanced)
            window_size: Number of messages to keep (total user + assistant)
            max_token_limit: Max tokens for summary memory
            return_messages: Return as Message objects (True) or strings (False)
            save_context: Save conversation context
        """
        self.memory_type = memory_type
        self.window_size = window_size
        self.max_token_limit = max_token_limit
        self.return_messages = return_messages
        self.save_context = save_context
        
        # Session storage (per user/session ID)
        self.sessions: Dict[str, BaseMemory] = {}
        
        # Statistics
        self.stats = {
            'total_sessions': 0,
            'active_sessions': 0,
            'total_messages': 0,
            'created_at': datetime.now().isoformat()
        }
        
        logger.info("="*70)
        logger.info("✅ ConversationMemoryManager initialized")
        logger.info(f"   Memory type : {memory_type}")
        logger.info(f"   Window size : {window_size} messages")
        logger.info(f"   Return type : {'Messages' if return_messages else 'Strings'}")
        logger.info("="*70)
    
    # ============================================================
    # SESSION MANAGEMENT
    # ============================================================
    
    def get_memory(
        self,
        session_id: str = "default"
    ) -> BaseMemory:
        """
        Get or create memory for a session
        
        Args:
            session_id: Unique session identifier
                - Can be user ID
                - Can be conversation ID
                - Default: "default" (single user mode)
                
        Returns:
            Memory instance for this session
        """
        if session_id not in self.sessions:
            self.sessions[session_id] = self._create_memory()
            self.stats['total_sessions'] += 1
            self.stats['active_sessions'] = len(self.sessions)
            
            logger.info(f"🆕 New session: {session_id}")
            logger.debug(f"   Active sessions: {self.stats['active_sessions']}")
        
        return self.sessions[session_id]
    
    def _create_memory(self) -> BaseMemory:
        """Create a new memory instance based on type"""
        
        if self.memory_type == "buffer":
            # Keep ALL history (no limit)
            logger.debug("Creating buffer memory (unlimited)")
            return ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=self.return_messages,
                output_key="answer"
            )
        
        elif self.memory_type == "buffer_window":
            # Keep last N messages (same as your history[-6:])
            # window_size = 6 means 3 exchanges (3 user + 3 assistant)
            k = self.window_size // 2
            
            logger.debug(
                f"Creating buffer window memory "
                f"(k={k} exchanges, {self.window_size} messages total)"
            )
            
            return ConversationBufferWindowMemory(
                memory_key="chat_history",
                k=k,  # Number of exchanges to keep
                return_messages=self.return_messages,
                output_key="answer"
            )
        
        elif self.memory_type == "summary":
            # Summarize old messages (requires LLM)
            logger.debug("Creating summary memory")
            logger.warning(
                "⚠️  Summary memory requires LLM - "
                "falling back to buffer_window"
            )
            # Fallback to buffer_window (summary needs LLM init)
            return ConversationBufferWindowMemory(
                memory_key="chat_history",
                k=3,
                return_messages=self.return_messages,
                output_key="answer"
            )
        
        else:
            logger.warning(
                f"⚠️  Unknown memory type '{self.memory_type}', "
                f"using buffer_window"
            )
            return ConversationBufferWindowMemory(
                memory_key="chat_history",
                k=3,
                return_messages=self.return_messages,
                output_key="answer"
            )
    
    def clear_session(self, session_id: str):
        """
        Clear memory for a specific session
        
        Args:
            session_id: Session to clear
        """
        if session_id in self.sessions:
            self.sessions[session_id].clear()
            logger.info(f"🗑️  Session cleared: {session_id}")
        else:
            logger.warning(f"⚠️  Session not found: {session_id}")
    
    def delete_session(self, session_id: str):
        """
        Delete a session completely
        
        Args:
            session_id: Session to delete
        """
        if session_id in self.sessions:
            del self.sessions[session_id]
            self.stats['active_sessions'] = len(self.sessions)
            logger.info(f"🗑️  Session deleted: {session_id}")
        else:
            logger.warning(f"⚠️  Session not found: {session_id}")
    
    def clear_all(self):
        """Clear all sessions"""
        self.sessions.clear()
        self.stats['active_sessions'] = 0
        logger.info("🗑️  All sessions cleared")
    
    # ============================================================
    # MEMORY ACCESS (for backward compatibility)
    # ============================================================
    
    def get_history(
        self,
        session_id: str = "default"
    ) -> List[Dict[str, str]]:
        """
        Get conversation history in your original format
        
        Returns format compatible with your RAGService:
        [
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."},
            ...
        ]
        
        This allows backward compatibility with your existing code
        """
        memory = self.get_memory(session_id)
        messages = memory.chat_memory.messages
        
        # Convert to your format
        history = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                role = "user"
            elif isinstance(msg, AIMessage):
                role = "assistant"
            else:
                role = "system"
            
            history.append({
                "role": role,
                "content": msg.content
            })
        
        return history
    
    def add_message(
        self,
        session_id: str,
        user_message: str,
        assistant_message: str
    ):
        """
        Add a message exchange to memory
        
        Args:
            session_id: Session ID
            user_message: User's message
            assistant_message: Assistant's response
        """
        memory = self.get_memory(session_id)
        
        # Add to memory
        memory.save_context(
            {"input": user_message},
            {"answer": assistant_message}
        )
        
        self.stats['total_messages'] += 2
        
        logger.debug(
            f"💬 Message added to session '{session_id}' "
            f"({len(memory.chat_memory.messages)} total)"
        )
    
    def load_history(
        self,
        session_id: str,
        history: List[Dict[str, str]]
    ):
        """
        Load existing conversation history
        
        Useful for:
            - Restoring sessions
            - Migrating from old format
            - Testing
            
        Args:
            session_id: Session ID
            history: History in your original format
        """
        memory = self.get_memory(session_id)
        
        # Clear existing
        memory.clear()
        
        # Load messages in pairs
        for i in range(0, len(history) - 1, 2):
            if i + 1 < len(history):
                user_msg = history[i]
                assistant_msg = history[i + 1]
                
                if (user_msg.get('role') == 'user' and
                    assistant_msg.get('role') == 'assistant'):
                    
                    memory.save_context(
                        {"input": user_msg['content']},
                        {"answer": assistant_msg['content']}
                    )
        
        logger.info(
            f"📥 Loaded {len(history)} messages into session '{session_id}'"
        )
    
    # ============================================================
    # STATISTICS & MONITORING
    # ============================================================
    
    def get_session_count(self) -> int:
        """Get number of active sessions"""
        return len(self.sessions)
    
    def get_session_message_count(self, session_id: str) -> int:
        """Get message count for a session"""
        if session_id not in self.sessions:
            return 0
        
        return len(self.sessions[session_id].chat_memory.messages)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get memory statistics
        
        Returns:
            Statistics about all sessions
        """
        session_details = {}
        for session_id, memory in self.sessions.items():
            session_details[session_id] = {
                'message_count': len(memory.chat_memory.messages),
                'memory_type': self.memory_type
            }
        
        return {
            'memory_type': self.memory_type,
            'window_size': self.window_size,
            'total_sessions_created': self.stats['total_sessions'],
            'active_sessions': self.stats['active_sessions'],
            'total_messages_processed': self.stats['total_messages'],
            'created_at': self.stats['created_at'],
            'sessions': session_details
        }
    
    def get_session_info(self, session_id: str) -> Dict[str, Any]:
        """
        Get detailed info about a specific session
        
        Args:
            session_id: Session ID
            
        Returns:
            Session details
        """
        if session_id not in self.sessions:
            return {
                'exists': False,
                'session_id': session_id
            }
        
        memory = self.sessions[session_id]
        messages = memory.chat_memory.messages
        
        return {
            'exists': True,
            'session_id': session_id,
            'message_count': len(messages),
            'memory_type': self.memory_type,
            'window_size': self.window_size,
            'messages': [
                {
                    'role': (
                        'user' if isinstance(msg, HumanMessage)
                        else 'assistant'
                    ),
                    'content': msg.content[:100] + '...'
                    if len(msg.content) > 100
                    else msg.content,
                    'length': len(msg.content)
                }
                for msg in messages
            ]
        }
    
    # ============================================================
    # CLEANUP & MAINTENANCE
    # ============================================================
    
    def cleanup_inactive_sessions(self, max_age_hours: int = 24):
        """
        Cleanup old sessions
        
        Note: Requires timestamp tracking (future enhancement)
        
        Args:
            max_age_hours: Max age before cleanup
        """
        logger.info(
            f"🧹 Cleanup not implemented yet "
            f"(requires timestamp tracking)"
        )
        # TODO: Add timestamp tracking to sessions
        # TODO: Implement age-based cleanup
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """
        Estimate memory usage
        
        Returns:
            Memory usage statistics
        """
        total_messages = sum(
            len(memory.chat_memory.messages)
            for memory in self.sessions.values()
        )
        
        # Rough estimate: ~500 bytes per message
        estimated_bytes = total_messages * 500
        
        return {
            'active_sessions': len(self.sessions),
            'total_messages': total_messages,
            'estimated_memory_mb': estimated_bytes / (1024 * 1024),
            'avg_messages_per_session': (
                total_messages / len(self.sessions)
                if self.sessions else 0
            )
        }


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def create_default_memory() -> ConversationMemoryManager:
    """
    Create default memory manager
    Same configuration as your original history[-6:]
    """
    return ConversationMemoryManager(
        memory_type="buffer_window",
        window_size=6,  # Same as your code
        return_messages=True
    )


def migrate_history_to_memory(
    old_history: List[Dict[str, str]],
    session_id: str = "default"
) -> ConversationMemoryManager:
    """
    Migrate your old conversation_history format to LangChain memory
    
    Args:
        old_history: Your original history format
        session_id: Session ID for the new memory
        
    Returns:
        Memory manager with loaded history
    """
    manager = create_default_memory()
    manager.load_history(session_id, old_history)
    return manager


# ============================================================
# USAGE EXAMPLES (for documentation)
# ============================================================

if __name__ == "__main__":
    """
    Example usage of ConversationMemoryManager
    """
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("\n" + "="*70)
    print("CONVERSATION MEMORY MANAGER - EXAMPLES")
    print("="*70)
    
    # Example 1: Basic usage
    print("\n1️⃣  Basic Usage (single user)")
    print("-"*70)
    
    manager = ConversationMemoryManager(
        memory_type="buffer_window",
        window_size=6
    )
    
    # Add conversation
    manager.add_message(
        session_id="default",
        user_message="What is machine learning?",
        assistant_message="Machine learning is a subset of AI..."
    )
    
    manager.add_message(
        session_id="default",
        user_message="Give me an example",
        assistant_message="A common example is spam detection..."
    )
    
    # Get history
    history = manager.get_history("default")
    print(f"History: {len(history)} messages")
    for msg in history:
        print(f"  {msg['role']}: {msg['content'][:50]}...")
    
    # Example 2: Multi-user sessions
    print("\n2️⃣  Multi-User Sessions")
    print("-"*70)
    
    # User 1
    manager.add_message(
        session_id="user_123",
        user_message="Hello!",
        assistant_message="Hi! How can I help?"
    )
    
    # User 2
    manager.add_message(
        session_id="user_456",
        user_message="Bonjour!",
        assistant_message="Bonjour! Comment puis-je vous aider?"
    )
    
    stats = manager.get_stats()
    print(f"Active sessions: {stats['active_sessions']}")
    print(f"Total messages: {stats['total_messages_processed']}")
    
    # Example 3: Migration from old format
    print("\n3️⃣  Migration from Old Format")
    print("-"*70)
    
    old_history = [
        {"role": "user", "content": "Question 1"},
        {"role": "assistant", "content": "Answer 1"},
        {"role": "user", "content": "Question 2"},
        {"role": "assistant", "content": "Answer 2"}
    ]
    
    migrated = migrate_history_to_memory(old_history, "migrated_session")
    migrated_history = migrated.get_history("migrated_session")
    print(f"Migrated: {len(migrated_history)} messages")
    
    # Example 4: Session info
    print("\n4️⃣  Session Information")
    print("-"*70)
    
    info = manager.get_session_info("user_123")
    print(f"Session exists: {info['exists']}")
    print(f"Message count: {info['message_count']}")
    print(f"Messages:")
    for msg in info['messages']:
        print(f"  - {msg['role']}: {msg['content']}")
    
    # Example 5: Cleanup
    print("\n5️⃣  Cleanup")
    print("-"*70)
    
    print(f"Before cleanup: {manager.get_session_count()} sessions")
    manager.clear_session("user_456")
    print(f"After clear one: {manager.get_session_count()} sessions")
    manager.clear_all()
    print(f"After clear all: {manager.get_session_count()} sessions")
    
    print("\n" + "="*70)