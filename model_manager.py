import os
import streamlit as st
import json
from typing import Optional, Dict, Any, Union

# Try to import llama_cpp, handle gracefully if not available
try:
    from llama_cpp import Llama  # type: ignore
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    Llama = None

class ModelManager:
    def __init__(self):
        self.current_model: Optional[Any] = None  # Type as Any since Llama may not be available
        self.current_model_path: Optional[str] = None
        self.model_type: str = "text"  # Default model type: text, code, multimodal
        self.system_prompts = {
            "text": "You are a helpful AI assistant. Please provide clear, accurate, and helpful responses.",
            "code": "You are an expert programming assistant. Provide clean, well-commented code with explanations. Focus on best practices and efficient solutions.",
            "multimodal": "You are a multimodal AI assistant capable of understanding and reasoning about text, images, and other media. Provide comprehensive and contextual responses."
        }
    
    def load_model(self, model_path: str, hardware_config: Optional[Dict[str, Any]] = None, **kwargs) -> bool:
        """Load a GGUF model from the specified path with hardware acceleration settings."""
        if not LLAMA_CPP_AVAILABLE:
            st.error("⚠️ llama-cpp-python is not installed. Please install it to use GGUF models.")
            st.info("To install, try running: `pip install llama-cpp-python` in your terminal or use the Dependencies tab in Replit.")
            return False
            
        try:
            # Unload current model if any
            self.unload_model()
            
            # Use hardware config if provided
            if hardware_config is None:
                hardware_config = {
                    'acceleration_mode': 'cpu_only',
                    'n_gpu_layers': 0,
                    'n_threads': os.cpu_count() or 4,
                    'use_server_cpu': True
                }
            
            # Default parameters for model loading
            default_params = {
                'model_path': model_path,
                'n_ctx': 2048,  # Context length
                'n_threads': hardware_config.get('n_threads', os.cpu_count() or 4),
                'n_gpu_layers': hardware_config.get('n_gpu_layers', 0),
                'verbose': False
            }
            
            # Adjust parameters based on acceleration mode
            mode = hardware_config.get('acceleration_mode', 'cpu_only')
            
            if mode == 'server_cpu':
                # Conservative settings for server environments
                default_params.update({
                    'n_threads': min(4, hardware_config.get('n_threads', 4)),
                    'n_gpu_layers': 0,
                    'n_batch': 128,  # Smaller batch size
                    'use_mlock': False,  # Don't lock memory
                    'use_mmap': True   # Use memory mapping for efficiency
                })
            elif mode == 'gpu_full':
                # GPU-optimized settings
                default_params.update({
                    'n_gpu_layers': -1,  # All layers on GPU
                    'n_threads': 1,      # Minimal CPU threads when using GPU
                })
            elif mode in ['gpu_partial', 'cpu_gpu_hybrid']:
                # Hybrid CPU+GPU settings
                default_params.update({
                    'n_gpu_layers': hardware_config.get('n_gpu_layers', 10),
                    'n_threads': hardware_config.get('n_threads', 4)
                })
            
            # Add additional GPU-specific parameters if using GPU
            if hardware_config.get('n_gpu_layers', 0) > 0:
                default_params.update({
                    'main_gpu': 0,           # Use primary GPU
                    'split_mode': 1,         # Layer-wise split
                    'tensor_split': None     # Auto-split tensors
                })
            
            # Override with any custom parameters
            default_params.update(kwargs)
            
            # Display loading info
            if mode != 'cpu_only':
                st.info(f"🔧 Loading model with {mode} acceleration: {hardware_config.get('n_gpu_layers', 0)} GPU layers, {default_params['n_threads']} CPU threads")
            else:
                st.info(f"🔧 Loading model with CPU-only: {default_params['n_threads']} threads")
            
            # Load the model
            if Llama is not None:
                self.current_model = Llama(**default_params)
            else:
                return False
            self.current_model_path = model_path
            
            return True
            
        except Exception as e:
            error_msg = str(e)
            st.error(f"Error loading model: {error_msg}")
            
            # Provide specific guidance for GPU-related errors
            if 'CUDA' in error_msg or 'GPU' in error_msg or 'cuBLAS' in error_msg:
                st.warning("🎮 GPU Error: Try switching to 'CPU Only' mode or reducing GPU layers.")
            elif 'memory' in error_msg.lower() or 'oom' in error_msg.lower():
                st.warning("💾 Memory Error: Try reducing context length or using a smaller model.")
            
            return False
    
    def unload_model(self):
        """Unload the current model to free memory."""
        if self.current_model:
            del self.current_model
            self.current_model = None
            self.current_model_path = None
    
    def generate_response(self, prompt: str, temperature: float = 0.7, 
                         max_tokens: int = 512, top_p: float = 0.9, 
                         model_type: str = "text", repeat_penalty: float = 1.1,
                         top_k: int = 40, seed: int = -1, **kwargs) -> Dict[str, Any]:
        """Generate a response using the loaded model."""
        if not LLAMA_CPP_AVAILABLE:
            return {
                'text': "⚠️ llama-cpp-python is not installed. Cannot generate responses without the package.",
                'usage': {}
            }
            
        if not self.current_model:
            st.error("No model loaded. Please load a model first.")
            return {'text': None, 'usage': {}}
        
        try:
            # Prepare context-aware prompt based on model type
            contextual_prompt = self._prepare_contextual_prompt(prompt, model_type)
            
            # Adjust parameters based on model type
            adjusted_params = self._adjust_parameters_for_type(
                model_type, temperature, max_tokens, top_p
            )
            
            # Generate response
            if hasattr(self.current_model, '__call__'):
                # Prepare generation parameters
                gen_params = {
                    'max_tokens': adjusted_params['max_tokens'],
                    'temperature': adjusted_params['temperature'],
                    'top_p': adjusted_params['top_p'],
                    'top_k': top_k,
                    'repeat_penalty': repeat_penalty,
                    'echo': False,
                    'stop': self._get_stop_tokens_for_type(model_type)
                }
                
                # Add seed if specified
                if seed != -1:
                    gen_params['seed'] = seed
                
                # Merge any additional kwargs
                gen_params.update(kwargs)
                
                response = self.current_model(contextual_prompt, **gen_params)
            else:
                return {'text': "Error: Model is not callable", 'usage': {}}
            
            # Extract the generated text and response metadata
            if response and 'choices' in response and len(response['choices']) > 0:
                generated_text = response['choices'][0]['text'].strip()
                
                # Extract response metadata if available
                usage_info = {}
                if 'usage' in response:
                    usage_info = {
                        'prompt_tokens': response['usage'].get('prompt_tokens', 0),
                        'completion_tokens': response['usage'].get('completion_tokens', 0),
                        'total_tokens': response['usage'].get('total_tokens', 0)
                    }
                else:
                    # Estimate token usage if not provided
                    prompt_tokens = int(len(contextual_prompt.split()) * 0.75)
                    completion_tokens = int(len(generated_text.split()) * 0.75)
                    usage_info = {
                        'prompt_tokens': prompt_tokens,
                        'completion_tokens': completion_tokens,
                        'total_tokens': prompt_tokens + completion_tokens
                    }
                
                return {
                    'text': generated_text,
                    'usage': usage_info
                }
            else:
                return {'text': None, 'usage': {}}
                
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
            return {'text': None, 'usage': {}}
    
    def get_basic_model_info(self, model_path: str) -> Dict[str, Any]:
        """Get basic file information about a GGUF model without loading it."""
        info = {
            'file_size': 0,
            'parameters': 'Unknown',
            'model_type': 'GGUF Model',
            'architecture': 'Unknown'
        }
        
        try:
            # Get file size
            if os.path.exists(model_path):
                info['file_size'] = os.path.getsize(model_path)
                
            # Basic GGUF file validation
            if self._validate_gguf_file(model_path):
                info['model_type'] = 'Valid GGUF Model'
            else:
                info['model_type'] = 'Invalid GGUF File'
                
        except Exception as e:
            st.warning(f"Could not extract basic model info: {str(e)}")
        
        return info
    
    def get_model_info(self, model_path: str) -> Dict[str, Any]:
        """Extract basic information about a GGUF model file."""
        info = {
            'file_size': 0,
            'parameters': 'Unknown',
            'model_type': 'Unknown',
            'architecture': 'Unknown'
        }
        
        try:
            # Get file size
            if os.path.exists(model_path):
                info['file_size'] = os.path.getsize(model_path)
            
            # Try to load model temporarily to get metadata (only if llama_cpp is available)
            if LLAMA_CPP_AVAILABLE and Llama is not None:
                try:
                    temp_model = Llama(
                        model_path=model_path,
                        n_ctx=256,  # Minimal context for info extraction
                        verbose=False
                    )
                    
                    # Try to extract model metadata if available
                    if hasattr(temp_model, 'metadata'):
                        metadata = temp_model.metadata
                        if 'general.parameter_count' in metadata:
                            param_count = metadata['general.parameter_count']
                            info['parameters'] = self._format_parameter_count(param_count)
                        
                        if 'general.architecture' in metadata:
                            info['architecture'] = metadata['general.architecture']
                        
                        if 'general.name' in metadata:
                            info['model_type'] = metadata['general.name']
                    
                    # Clean up temporary model
                    del temp_model
                    
                except Exception:
                    # If we can't load the model for metadata, that's okay
                    # We'll just return basic file info
                    pass
            else:
                info['model_type'] = 'GGUF Model (llama-cpp-python required for detailed info)'
                
        except Exception as e:
            st.warning(f"Could not extract full model info: {str(e)}")
        
        return info
    
    def _format_parameter_count(self, count) -> str:
        """Format parameter count in a human-readable way."""
        try:
            count = int(count)
            if count >= 1_000_000_000:
                return f"{count / 1_000_000_000:.1f}B"
            elif count >= 1_000_000:
                return f"{count / 1_000_000:.1f}M"
            elif count >= 1_000:
                return f"{count / 1_000:.1f}K"
            else:
                return str(count)
        except:
            return str(count)
    
    def is_model_loaded(self) -> bool:
        """Check if a model is currently loaded."""
        return self.current_model is not None
    
    def get_current_model_path(self) -> Optional[str]:
        """Get the path of the currently loaded model."""
        return self.current_model_path
    
    def _validate_gguf_file(self, file_path: str) -> bool:
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
    
    def _prepare_contextual_prompt(self, prompt: str, model_type: str) -> str:
        """Prepare prompt with appropriate context for the model type."""
        system_prompt = self.system_prompts.get(model_type, self.system_prompts["text"])
        
        if model_type == "code":
            # Add code-specific formatting
            return f"{system_prompt}\n\nUser request: {prompt}\n\nPlease provide a complete solution with code examples and explanations:"
        elif model_type == "multimodal":
            # Add multimodal context
            return f"{system_prompt}\n\nUser input: {prompt}\n\nResponse:"
        else:
            # Default text model formatting
            return f"{system_prompt}\n\nUser: {prompt}\n\nAssistant:"
    
    def _adjust_parameters_for_type(self, model_type: str, temperature: float, 
                                   max_tokens: int, top_p: float) -> Dict[str, Any]:
        """Adjust generation parameters based on model type."""
        if model_type == "code":
            # Lower temperature for more deterministic code generation
            return {
                'temperature': min(temperature, 0.3),
                'max_tokens': max(max_tokens, 1024),  # More tokens for code
                'top_p': min(top_p, 0.85)
            }
        elif model_type == "multimodal":
            # Balanced parameters for multimodal
            return {
                'temperature': temperature,
                'max_tokens': max(max_tokens, 768),
                'top_p': top_p
            }
        else:
            # Default text parameters
            return {
                'temperature': temperature,
                'max_tokens': max_tokens,
                'top_p': top_p
            }
    
    def _get_stop_tokens_for_type(self, model_type: str) -> list:
        """Get appropriate stop tokens for the model type."""
        base_stops = ["</s>", "<|endoftext|>"]
        
        if model_type == "code":
            return base_stops + ["\n\n# ", "\n\n```", "\n\nUser:", "```\n\n"]
        elif model_type == "multimodal":
            return base_stops + ["\n\nUser:", "\n\nHuman:", "\n\nImage:"]
        else:
            return base_stops + ["\n\nUser:", "\n\nHuman:"]
    
    def set_model_type(self, model_type: str):
        """Set the current model type."""
        if model_type in self.system_prompts:
            self.model_type = model_type
        else:
            st.warning(f"Unknown model type: {model_type}. Using 'text' as default.")
            self.model_type = "text"
    
    def get_model_type(self) -> str:
        """Get the current model type."""
        return self.model_type
    
    def get_supported_model_types(self) -> list:
        """Get list of supported model types."""
        return list(self.system_prompts.keys())
