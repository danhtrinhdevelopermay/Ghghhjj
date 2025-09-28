import streamlit as st
from typing import List, Dict, Any, Optional
from datetime import datetime

class ChatInterface:
    def __init__(self):
        self.conversation_history = []
    
    def add_message(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None):
        """Add a message to the conversation history."""
        message = {
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        self.conversation_history.append(message)
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get the full conversation history."""
        return self.conversation_history
    
    def clear_conversation(self):
        """Clear the conversation history."""
        self.conversation_history = []
    
    def format_conversation_for_model(self, include_system: bool = True) -> str:
        """Format the conversation history for model input."""
        formatted_messages = []
        
        if include_system:
            formatted_messages.append("You are a helpful AI assistant.")
        
        for message in self.conversation_history:
            role = message['role']
            content = message['content']
            
            if role == 'user':
                formatted_messages.append(f"User: {content}")
            elif role == 'assistant':
                formatted_messages.append(f"Assistant: {content}")
        
        return "\n\n".join(formatted_messages)
    
    def get_last_user_message(self) -> str:
        """Get the last user message from the conversation."""
        for message in reversed(self.conversation_history):
            if message['role'] == 'user':
                return message['content']
        return ""
    
    def display_chat_messages(self):
        """Display chat messages using Streamlit's chat interface."""
        for message in self.conversation_history:
            role = message['role']
            content = message['content']
            
            if role == 'user':
                st.chat_message("user").write(content)
            elif role == 'assistant':
                st.chat_message("assistant").write(content)
            elif role == 'system':
                with st.chat_message("assistant"):
                    st.info(content)
    
    def get_conversation_stats(self) -> Dict[str, Any]:
        """Get statistics about the current conversation."""
        total_messages = len(self.conversation_history)
        user_messages = sum(1 for msg in self.conversation_history if msg['role'] == 'user')
        assistant_messages = sum(1 for msg in self.conversation_history if msg['role'] == 'assistant')
        
        total_chars = sum(len(msg['content']) for msg in self.conversation_history)
        
        return {
            'total_messages': total_messages,
            'user_messages': user_messages,
            'assistant_messages': assistant_messages,
            'total_characters': total_chars,
            'avg_message_length': total_chars / total_messages if total_messages > 0 else 0
        }
    
    def export_conversation(self, format_type: str = 'json') -> str:
        """Export the conversation in the specified format."""
        if format_type.lower() == 'json':
            import json
            return json.dumps(self.conversation_history, indent=2, ensure_ascii=False)
        
        elif format_type.lower() == 'txt':
            lines = []
            for message in self.conversation_history:
                timestamp = message.get('timestamp', '')
                role = message['role'].title()
                content = message['content']
                lines.append(f"[{timestamp}] {role}: {content}")
            return "\n\n".join(lines)
        
        elif format_type.lower() == 'md':
            lines = ["# Conversation Export", ""]
            for message in self.conversation_history:
                timestamp = message.get('timestamp', '')
                role = message['role'].title()
                content = message['content']
                
                lines.append(f"## {role} ({timestamp})")
                lines.append(content)
                lines.append("")
            
            return "\n".join(lines)
        
        else:
            raise ValueError(f"Unsupported export format: {format_type}")
    
    def search_conversation(self, query: str) -> List[Dict[str, Any]]:
        """Search for messages containing the query string."""
        results = []
        query_lower = query.lower()
        
        for i, message in enumerate(self.conversation_history):
            if query_lower in message['content'].lower():
                results.append({
                    'index': i,
                    'message': message,
                    'context': self._get_message_context(i)
                })
        
        return results
    
    def _get_message_context(self, index: int, context_size: int = 2) -> List[Dict[str, Any]]:
        """Get context messages around a specific message index."""
        start = max(0, index - context_size)
        end = min(len(self.conversation_history), index + context_size + 1)
        return self.conversation_history[start:end]
