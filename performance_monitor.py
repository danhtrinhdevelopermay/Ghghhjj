import time
import psutil
from typing import Dict, Any, List, Optional
from datetime import datetime
import threading
import streamlit as st

class PerformanceMonitor:
    def __init__(self):
        self.metrics_history: List[Dict[str, Any]] = []
        self.current_session_metrics: Dict[str, Any] = {}
        self.start_time: Optional[float] = None
        self.memory_monitoring = False
        self.memory_samples: List[float] = []
        
    def start_generation_timing(self):
        """Start timing for generation performance."""
        self.start_time = time.time()
        self.current_session_metrics = {
            'start_time': datetime.now().isoformat(),
            'memory_before': self._get_memory_usage(),
            'cpu_before': self._get_cpu_usage()
        }
        
        # Start memory monitoring in background
        self._start_memory_monitoring()
    
    def end_generation_timing(self, response: str = "", prompt: str = "", response_metadata: Dict[str, Any] = None, **generation_params):
        """End timing and calculate performance metrics."""
        if self.start_time is None:
            return None
            
        end_time = time.time()
        generation_time = end_time - self.start_time
        
        # Stop memory monitoring
        self._stop_memory_monitoring()
        
        # Extract token information from response_metadata or estimate
        if response_metadata:
            prompt_tokens = response_metadata.get('prompt_tokens', 0)
            response_tokens = response_metadata.get('completion_tokens', 0)
        elif generation_params and any(key in generation_params for key in ['prompt_tokens', 'completion_tokens']):
            # Try to extract from generation_params if available
            prompt_tokens = generation_params.get('prompt_tokens', len(prompt.split()))
            response_tokens = generation_params.get('completion_tokens', len(response.split()) if response else 0)
        else:
            # Fallback to rough estimation by word count (approximately 0.75 words per token)
            prompt_tokens = int(len(prompt.split()) * 0.75) if prompt else 0
            response_tokens = int(len(response.split()) * 0.75) if response else 0
        
        total_tokens = prompt_tokens + response_tokens
        
        # Calculate metrics
        tokens_per_second = response_tokens / generation_time if generation_time > 0 else 0
        
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'generation_time': generation_time,
            'prompt_length': len(prompt),
            'prompt_tokens': prompt_tokens,
            'response_length': len(response) if response else 0,
            'response_tokens': response_tokens,
            'total_tokens': total_tokens,
            'tokens_per_second': tokens_per_second,
            'memory_before': self.current_session_metrics.get('memory_before', 0),
            'memory_after': self._get_memory_usage(),
            'memory_peak': max(self.memory_samples) if self.memory_samples else 0,
            'cpu_before': self.current_session_metrics.get('cpu_before', 0),
            'cpu_after': self._get_cpu_usage(),
            'generation_params': generation_params
        }
        
        # Calculate memory delta
        metrics['memory_delta'] = metrics['memory_after'] - metrics['memory_before']
        
        # Add to history
        self.metrics_history.append(metrics)
        
        # Reset for next generation
        self.start_time = None
        self.memory_samples = []
        
        return metrics
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get aggregated statistics for the current session."""
        if not self.metrics_history:
            return {}
        
        # Calculate aggregated metrics
        total_generations = len(self.metrics_history)
        avg_generation_time = sum(m['generation_time'] for m in self.metrics_history) / total_generations
        avg_tokens_per_second = sum(m['tokens_per_second'] for m in self.metrics_history) / total_generations
        total_tokens_generated = sum(m['response_tokens'] for m in self.metrics_history)
        total_time = sum(m['generation_time'] for m in self.metrics_history)
        
        fastest_generation = min(self.metrics_history, key=lambda x: x['generation_time'])
        slowest_generation = max(self.metrics_history, key=lambda x: x['generation_time'])
        
        return {
            'total_generations': total_generations,
            'total_tokens_generated': total_tokens_generated,
            'total_generation_time': total_time,
            'average_generation_time': avg_generation_time,
            'average_tokens_per_second': avg_tokens_per_second,
            'fastest_generation_time': fastest_generation['generation_time'],
            'slowest_generation_time': slowest_generation['generation_time'],
            'current_memory_usage': self._get_memory_usage(),
            'current_cpu_usage': self._get_cpu_usage()
        }
    
    def get_recent_metrics(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get the most recent performance metrics."""
        return self.metrics_history[-count:] if self.metrics_history else []
    
    def clear_metrics(self):
        """Clear all stored metrics."""
        self.metrics_history = []
        self.current_session_metrics = {}
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # Convert to MB
        except:
            return 0.0
    
    def _get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        try:
            return psutil.cpu_percent(interval=0.1)
        except:
            return 0.0
    
    def _start_memory_monitoring(self):
        """Start monitoring memory usage in background."""
        self.memory_monitoring = True
        self.memory_samples = []
        threading.Thread(target=self._monitor_memory, daemon=True).start()
    
    def _stop_memory_monitoring(self):
        """Stop monitoring memory usage."""
        self.memory_monitoring = False
    
    def _monitor_memory(self):
        """Monitor memory usage during generation."""
        while self.memory_monitoring:
            self.memory_samples.append(self._get_memory_usage())
            time.sleep(0.1)  # Sample every 100ms
    
    def export_metrics(self) -> str:
        """Export all metrics as JSON string."""
        import json
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'session_stats': self.get_session_stats(),
            'detailed_metrics': self.metrics_history
        }
        return json.dumps(export_data, indent=2, ensure_ascii=False)
    
    def get_performance_summary(self) -> Dict[str, str]:
        """Get a formatted performance summary for display."""
        stats = self.get_session_stats()
        if not stats:
            return {"status": "No performance data available"}
        
        return {
            "Total Generations": f"{stats['total_generations']}",
            "Average Response Time": f"{stats['average_generation_time']:.2f}s",
            "Average Speed": f"{stats['average_tokens_per_second']:.1f} tokens/sec",
            "Total Tokens Generated": f"{stats['total_tokens_generated']}",
            "Fastest Generation": f"{stats['fastest_generation_time']:.2f}s",
            "Slowest Generation": f"{stats['slowest_generation_time']:.2f}s",
            "Current Memory Usage": f"{stats['current_memory_usage']:.1f} MB",
            "Current CPU Usage": f"{stats['current_cpu_usage']:.1f}%"
        }