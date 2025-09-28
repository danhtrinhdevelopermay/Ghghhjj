import json
import os
from datetime import datetime
from typing import List, Dict, Any

def format_file_size(size_bytes: int) -> str:
    """Convert bytes to human readable file size."""
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    size = float(size_bytes)
    while size >= 1024 and i < len(size_names) - 1:
        size /= 1024.0
        i += 1
    
    return f"{size:.1f} {size_names[i]}"

def export_conversation(conversation_history: List[Dict[str, Any]], model_name: str) -> str:
    """Export conversation history as JSON string."""
    export_data = {
        'model_name': model_name,
        'export_timestamp': datetime.now().isoformat(),
        'conversation_count': len(conversation_history),
        'conversation': conversation_history
    }
    
    return json.dumps(export_data, indent=2, ensure_ascii=False)

def validate_gguf_file(file_path: str) -> bool:
    """Basic validation to check if a file might be a valid GGUF file."""
    try:
        if not os.path.exists(file_path):
            return False
        
        # Check file size (should be at least a few MB for a valid model)
        file_size = os.path.getsize(file_path)
        if file_size < 1024 * 1024:  # Less than 1MB is probably not a valid model
            return False
        
        # Check if file has .gguf extension
        if not file_path.lower().endswith('.gguf'):
            return False
        
        # Try to read the first few bytes to check for GGUF magic
        with open(file_path, 'rb') as f:
            header = f.read(4)
            # GGUF files should start with 'GGUF' magic bytes
            if header == b'GGUF':
                return True
        
        return False
        
    except Exception:
        return False

def get_model_cache_dir() -> str:
    """Get the directory for storing model cache."""
    cache_dir = os.path.join(os.getcwd(), 'models')
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir

def clean_model_cache():
    """Clean up old model files from cache directory."""
    cache_dir = get_model_cache_dir()
    try:
        for filename in os.listdir(cache_dir):
            file_path = os.path.join(cache_dir, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
    except Exception as e:
        print(f"Error cleaning cache: {e}")

def format_timestamp(timestamp_str: str) -> str:
    """Format ISO timestamp string to readable format."""
    try:
        dt = datetime.fromisoformat(timestamp_str)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except:
        return timestamp_str

def get_conversation_summary(conversation_history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate a summary of the conversation."""
    if not conversation_history:
        return {
            'total_messages': 0,
            'user_messages': 0,
            'assistant_messages': 0,
            'total_words': 0,
            'duration': '0 minutes'
        }
    
    user_messages = [msg for msg in conversation_history if msg['role'] == 'user']
    assistant_messages = [msg for msg in conversation_history if msg['role'] == 'assistant']
    
    total_words = sum(len(msg['content'].split()) for msg in conversation_history)
    
    # Calculate duration
    try:
        start_time = datetime.fromisoformat(conversation_history[0]['timestamp'])
        end_time = datetime.fromisoformat(conversation_history[-1]['timestamp'])
        duration = end_time - start_time
        duration_minutes = int(duration.total_seconds() / 60)
        duration_str = f"{duration_minutes} minutes"
    except:
        duration_str = "Unknown"
    
    return {
        'total_messages': len(conversation_history),
        'user_messages': len(user_messages),
        'assistant_messages': len(assistant_messages),
        'total_words': total_words,
        'duration': duration_str
    }

def sanitize_filename(filename: str) -> str:
    """Sanitize filename to remove invalid characters."""
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    return filename

def get_available_models() -> List[str]:
    """Get list of available model files in the models directory."""
    models_dir = get_model_cache_dir()
    model_files = []
    
    try:
        for filename in os.listdir(models_dir):
            if filename.lower().endswith('.gguf'):
                file_path = os.path.join(models_dir, filename)
                if validate_gguf_file(file_path):
                    model_files.append(filename)
    except Exception as e:
        print(f"Error listing models: {e}")
    
    return sorted(model_files)

def estimate_model_memory_usage(file_size: int) -> str:
    """Estimate memory usage based on model file size."""
    # Rough estimation: model in memory is typically 1.2-1.5x file size
    estimated_memory = file_size * 1.3
    return format_file_size(int(estimated_memory))
